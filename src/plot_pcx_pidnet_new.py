import os
import sys
import gc
import time
import h5py
import copy
import shutil
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.utils import draw_segmentation_masks, make_grid
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture

import zennit.image as zimage
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from LCRP.utils.render import vis_opaque_img_border
#from src.pcx_helper import plot_gmm_umap_2d, plot_gmm_umap_3d, plot_prototype_n_nearest


logging.disable(logging.ERROR)
# -------------------------
# Small helpers
# -------------------------
def _layer_candidates(layer: str):
    cands = [layer]
    heads = ("final_layer", "seghead_p", "seghead_d")
    for h in heads:
        if layer.startswith(f"{h}.sequential."):
            cands.append(layer.replace(f"{h}.sequential.", f"{h}.", 1))
        elif layer.startswith(f"{h}."):
            cands.append(layer.replace(f"{h}.", f"{h}.sequential.", 1))
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _resolve_attr_layer(attr, requested_layer: str):
    keys = set(attr.relevances.keys()) | set(attr.activations.keys())
    for cand in _layer_candidates(requested_layer):
        if cand in keys:
            return cand
    return requested_layer


def _resolve_pcx_attr_path(output_dir_pcx: str, layer: str):
    for cand in _layer_candidates(layer):
        p = os.path.join(output_dir_pcx, cand, "attributions.npy")
        if os.path.exists(p):
            return cand, p
    return layer, os.path.join(output_dir_pcx, layer, "attributions.npy")


def _resolve_crp_relmax_layer(output_dir_crp: str, layer: str):
    for cand in _layer_candidates(layer):
        p = os.path.join(output_dir_crp, "RelMax_sum_normed", f"{cand}_data.npy")
        if os.path.exists(p):
            return cand, p
    return layer, os.path.join(output_dir_crp, "RelMax_sum_normed", f"{layer}_data.npy")


def _ensure_crp_relmax_alias(output_dir_crp: str, src_layer: str, dst_layer: str):
    if src_layer == dst_layer:
        return
    base = os.path.join(output_dir_crp, "RelMax_sum_normed")
    os.makedirs(base, exist_ok=True)
    for suf in ("_data.npy", "_rel.npy", "_rf.npy"):
        src = os.path.join(base, f"{src_layer}{suf}")
        dst = os.path.join(base, f"{dst_layer}{suf}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)


def _as_uint8_array(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _show_message_box(
    ax: plt.Axes,
    message: str,
    facecolor: str = "#eef2ff",
    edgecolor: str = "#1f3fff",
    textcolor: str = "#1f3fff",
) -> None:
    ax.imshow(np.ones((150, 150, 3), dtype=np.float32))
    rect = patches.Rectangle((0, 0), 149, 149, linewidth=2.5, edgecolor=edgecolor, facecolor=facecolor, alpha=0.95)
    ax.add_patch(rect)
    ax.text(
        75, 75, message,
        ha="center", va="center",
        fontsize=9, color=textcolor, fontweight="bold",
        wrap=True
    )
    ax.set_xlim([0, 150])
    ax.set_ylim([149, 0])
    ax.set_xticks([])
    ax.set_yticks([])


def _select_diverse_concepts(
    channel_rels_vec: torch.Tensor,
    attributions_np: np.ndarray,
    max_concepts: int = 4,
    candidate_pool: int = 12,
    corr_penalty: float = 0.35,
    corr_skip_threshold: float = 0.92,
) -> Tuple[np.ndarray, np.ndarray]:
    rels = np.asarray(channel_rels_vec.detach().cpu(), dtype=np.float32).reshape(-1)
    if rels.size == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    max_concepts = max(1, min(int(max_concepts), rels.size))
    ranked = np.argsort(-rels)
    pool = ranked[: min(max(candidate_pool, max_concepts), ranked.size)]
    if pool.size <= max_concepts or attributions_np.ndim != 2 or attributions_np.shape[0] < 2:
        chosen = pool[:max_concepts]
        return chosen.astype(int), rels[chosen]

    bank = np.nan_to_num(attributions_np[:, pool], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    bank -= bank.mean(axis=0, keepdims=True)
    bank_std = bank.std(axis=0, keepdims=True)
    bank_std[bank_std < 1e-8] = 1.0
    bank /= bank_std
    corr = np.abs(np.corrcoef(bank, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)

    chosen_pos = [0]
    remaining = list(range(1, pool.size))
    while remaining and len(chosen_pos) < max_concepts:
        best_pos = None
        best_score = -np.inf
        slots_left = max_concepts - len(chosen_pos)
        for pos in remaining:
            max_corr = float(corr[pos, chosen_pos].max()) if chosen_pos else 0.0
            if max_corr >= corr_skip_threshold and len(remaining) > slots_left:
                continue
            score = float(rels[pool[pos]]) - corr_penalty * max_corr
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_pos is None:
            best_pos = remaining[0]
        chosen_pos.append(best_pos)
        remaining.remove(best_pos)

    chosen = pool[np.array(chosen_pos[:max_concepts], dtype=int)]
    order = np.argsort(-rels[chosen])
    chosen = chosen[order]
    return chosen.astype(int), rels[chosen]


def _select_diverse_reference_images(
    images: List[Image.Image],
    limit: int,
    hash_size: int = 24,
    duplicate_threshold: float = 2.0,
) -> List[Image.Image]:
    if limit <= 0 or not images:
        return []

    selected: List[Image.Image] = []
    fingerprints: List[np.ndarray] = []
    for image in images:
        arr = np.asarray(image.resize((hash_size, hash_size), Image.BILINEAR), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr -= arr.mean()
        norm = np.linalg.norm(arr)
        if norm > 1e-8:
            arr /= norm

        is_duplicate = False
        for prev in fingerprints:
            if np.linalg.norm(arr - prev) <= duplicate_threshold / hash_size:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(image)
            fingerprints.append(arr)
        if len(selected) >= limit:
            break

    if len(selected) < min(limit, len(images)):
        for image in images:
            if len(selected) >= limit:
                break
            if all(image is not chosen for chosen in selected):
                selected.append(image)

    return selected[:limit]

# -------------------------
# Reference images cache (HDF5)
# -------------------------
def get_ref_images(
    fv,
    topk_ind: np.ndarray,
    layer_name: str,
    composite,
    n_ref: int = 12,
    ref_imgs_save_path: str = "../examples/output/ref_images_pidnet_BRK/",
) -> Dict[int, List[Image.Image]]:
    """
    Get and cache reference images. CPU/PIL based.
    """
    t0 = time.time()
    path_h5 = os.path.join(ref_imgs_save_path, f"{layer_name}.h5")
    os.makedirs(os.path.dirname(path_h5), exist_ok=True)

    print(f"[get_ref_images] start; layer={layer_name}, n_ref={n_ref}, cache={path_h5}")
    ref_imgs: Dict[int, List[Image.Image]] = {}
    keys_str = list(map(str, topk_ind))

    if os.path.exists(path_h5):
        print("[get_ref_images] cache exists -> reading known keys")
        with h5py.File(path_h5, "a") as f:
            existing = set(f.keys())
            missing = [k for k in keys_str if k not in existing]

            # read existing
            for k in keys_str:
                if k in f:
                    grp = f[k]
                    items = sorted(grp.keys(), key=lambda s: int(s))
                    imgs = [Image.fromarray(grp[i][:]) for i in items]
                    ref_imgs[int(k)] = imgs
                    print(f"[get_ref_images] loaded key={k} count={len(imgs)}")

            # compute missing
            if missing:
                print(f"[get_ref_images] computing missing={missing}")
                new_refs = fv.get_max_reference(
                    [int(k) for k in missing],
                    layer_name,
                    "relevance",
                    (0, n_ref),
                    composite=composite,
                    rf=False,
                    plot_fn=vis_opaque_img_border,
                )
                for key, images_list in new_refs.items():
                    grp = f.create_group(str(key))
                    take = images_list[:n_ref]
                    ref_imgs[key] = []
                    for idx, im in enumerate(take):
                        if isinstance(im, Image.Image):
                            arr = _as_uint8_array(im)
                            grp.create_dataset(str(idx), data=arr, compression="gzip", compression_opts=4)
                            ref_imgs[key].append(im)
                    print(f"[get_ref_images] saved key={key} count={len(ref_imgs[key])}")
    else:
        print("[get_ref_images] cache missing -> computing all")
        refs = fv.get_max_reference(
            topk_ind.tolist(), layer_name, "relevance", (0, n_ref),
            composite=composite, rf=False, plot_fn=vis_opaque_img_border
        )
        with h5py.File(path_h5, "w") as f:
            for key, images_list in refs.items():
                grp = f.create_group(str(key))
                take = images_list[:n_ref]
                ref_imgs[key] = []
                for idx, im in enumerate(take):
                    if isinstance(im, Image.Image):
                        arr = _as_uint8_array(im)
                        grp.create_dataset(str(idx), data=arr, compression="gzip", compression_opts=4)
                        ref_imgs[key].append(im)
                print(f"[get_ref_images] saved key={key} count={len(ref_imgs[key])}")

    print(f"[get_ref_images] done in {time.time() - t0:.2f}s; keys={list(ref_imgs.keys())}")
    return ref_imgs


# -------------------------
# Main plotting entry
# -------------------------
def plot_pcx_explanations(
    model_name: str,
    model: torch.nn.Module,
    dataset,
    sample_id: int,
    layer_name: str,
    n_concepts: int,
    n_refimgs: int,
    num_prototypes: int,
    ref_imgs_path: str,
    output_dir_crp: str,
    output_dir_pcx: str,
    concept_ids: Optional[List[int]] = None):
    print("[plot_pcx_explanations] starting")
    img, target = dataset[sample_id]
    print(f"[plot_pcx_explanations] sample_id={sample_id} img.shape={tuple(img.shape)}")

    fig = plot_one_image_pcx_explanation(
        model_name=model_name,
        model=model,
        img=img,
        dataset=dataset,
        sample_id=sample_id,
        layer=layer_name,
        n_concepts=n_concepts,
        num_prototypes=num_prototypes,
        n_refimgs=n_refimgs,
        ref_imgs_path=ref_imgs_path,
        output_dir_crp=output_dir_crp,
        output_dir_pcx=output_dir_pcx,
        concept_ids=concept_ids,
    )

    print("[plot_pcx_explanations] finished")
    return fig

def plot_one_image_pcx_explanation(
    model_name: str,
    model: torch.nn.Module,
    img: torch.Tensor,
    dataset,
    sample_id: int,
    layer: str,
    n_concepts: int,
    num_prototypes: int,
    n_refimgs: int,
    ref_imgs_path: str,
    output_dir_crp: str,
    output_dir_pcx: str,
    concept_ids: Optional[List[int]] = None,
) -> plt.Figure:
    t_start = time.time()
    device = "cpu"  # keep CPU for sklearn compatibility
    print(f"[plot_one_image_pcx_explanation] device={device}, layer={layer}")

    model = model.to(device).eval()

    print("[attr] building attribution/composite/visualization")
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": 1}]

    layer_map = get_layer_names(model, [torch.nn.Conv2d])
    print(f"[attr] layer_map size={len(layer_map)}")
    fv = VISUALIZATIONS[model_name](
        attribution, dataset, layer_map, preprocess_fn=lambda x: x,
        path=output_dir_crp, max_target="max", device=device
    )

    x = img[None, ...].to(device)
    print(f"[data] x.shape={tuple(x.shape)}")

    # Attribution on the chosen sample
    print("[attr] computing base attribution")
    x_copy = copy.deepcopy(x).requires_grad_()
    attr = attribution(x_copy, condition, composite, record_layer=_layer_candidates(layer), init_rel=1)
    resolved_layer = _resolve_attr_layer(attr, layer)  # key available in attr.{relevances,activations}
    if resolved_layer != layer:
        print(f"[attr] remapped layer '{layer}' -> '{resolved_layer}'")
    # model-valid key for future conditional attributions/hooks
    runtime_layer = next((c for c in _layer_candidates(resolved_layer) if c in layer_map), resolved_layer)
    if runtime_layer != resolved_layer:
        print(f"[attr] using runtime layer '{runtime_layer}' for conditional attribution hooks")

    # Heatmap
    heatmap_pil = zimage.imgify(attr.heatmap.detach().cpu(), cmap="bwr", symmetric=True, level=10)
    print("[attr] base heatmap computed")

    # Build binary mask from prediction argmax==1
    print("[mask] building predicted mask")
    pred = attr.prediction[0].argmax(dim=0)   # [H,W]
    mask_bool = (pred == 1).detach().cpu()    # [H,W]

    # Determine target size using dataset.reverse_augmentation if available
    try:
        sample_ = dataset.reverse_augmentation(x[0] + 0)  # [C,H,W]
        tgt_hw = (sample_.shape[1], sample_.shape[2])
        print(f"[mask] reverse_augmentation available, target size={tgt_hw}")
    except Exception:
        # fallback: use original PIL from fv
        pil_orig = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
        sample_ = TF.pil_to_tensor(pil_orig).float() / 255.0
        tgt_hw = (sample_.shape[1], sample_.shape[2])
        print(f"[mask] reverse_augmentation missing, fallback target size={tgt_hw}")

    # Resize mask to target size
    mask_f = mask_bool.float().unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    mask_resized = F.interpolate(mask_f, size=tgt_hw, mode="nearest").bool().squeeze(0).squeeze(0)  # [H,W]
    print(f"[mask] resized mask -> {tuple(mask_resized.shape)}")

    # Channel relevances
    print("[concepts] computing channel relevances")
    channel_rels = ChannelConcept().attribute(attr.relevances[resolved_layer], abs_norm=True)  # [1,C]
    channel_rels_vec = channel_rels[0]  # [C]
    print(f"[concepts] channel_rels shape={tuple(channel_rels.shape)}")

    # Load attributions matrix for GMM
    resolved_pcx_layer, attr_path = _resolve_pcx_attr_path(output_dir_pcx, resolved_layer)
    if not os.path.exists(attr_path):
        raise FileNotFoundError(
            f"Attributions file not found for layer '{resolved_layer}'. "
            f"Tried: {[os.path.join(output_dir_pcx, c, 'attributions.npy') for c in _layer_candidates(resolved_layer)]}"
        )
    if resolved_pcx_layer != resolved_layer:
        print(f"[gmm] remapped PCX layer '{resolved_layer}' -> '{resolved_pcx_layer}'")
    print(f"[gmm] loading attributions @ {attr_path}")
    attributions_np = np.load(attr_path)  # shape [N, C]
    print(f"[gmm] attributions shape={attributions_np.shape}")
    if attributions_np.size == 0:
        raise ValueError(f"[gmm] Empty attribution bank for layer '{layer}'. Re-run attribution export first.")

    finite_mask = np.isfinite(attributions_np).all(axis=1)
    if not finite_mask.all():
        bad = int((~finite_mask).sum())
        print(f"[gmm] WARNING: dropping {bad} rows with NaN/Inf values before fitting GMM.")
        attributions_np = attributions_np[finite_mask]

    attributions_np = np.nan_to_num(attributions_np, nan=0.0, posinf=0.0, neginf=0.0)

    nonzero_mask = ~(np.isclose(attributions_np.std(axis=1), 0.0) & np.isclose(attributions_np.mean(axis=1), 0.0))
    if not nonzero_mask.any():
        raise ValueError(
            f"[gmm] No valid (non-zero) attribution rows left for layer '{layer}'. "
            "Recompute the attribution bank with multiple samples."
        )
    if not nonzero_mask.all():
        dropped = int((~nonzero_mask).sum())
        print(f"[gmm] INFO: dropping {dropped} constant rows from attribution bank.")
        attributions_np = attributions_np[nonzero_mask]

    # Diverse top concepts
    num_ch = int(channel_rels_vec.numel())
    max_display_concepts = max(1, min(int(n_concepts), num_ch))
    if num_ch <= 0:
        raise ValueError(f"[concepts] No channels available for layer '{resolved_layer}'.")
    if concept_ids is not None:
        requested = np.asarray(concept_ids, dtype=int).reshape(-1)
        requested = requested[(requested >= 0) & (requested < num_ch)]
        if requested.size == 0:
            raise ValueError("[concepts] Provided concept_ids are invalid for this layer.")
        topk_ind = requested[:max_display_concepts]
        topk_rel = channel_rels_vec.detach().cpu().numpy()[topk_ind]
        print(f"[concepts] using explicit concept_ids={topk_ind.tolist()}")
    else:
        topk_ind, topk_rel = _select_diverse_concepts(
            channel_rels_vec=channel_rels_vec,
            attributions_np=attributions_np,
            max_concepts=min(max_display_concepts, num_ch),
            candidate_pool=max(10, max_display_concepts * 3),
        )
    n_display_concepts = int(len(topk_ind))
    print(f"[concepts] selected_ind={topk_ind.tolist()} selected_rel={np.round(topk_rel,3).tolist()}")

    # Conditional heatmaps per concept
    conditions = [{"y": 1, runtime_layer: int(c)} for c in topk_ind]
    print("[attr] computing conditional heatmaps")
    attribution.take_prediction = 0
    cond_heatmaps, _, _, _ = attribution(
        x.requires_grad_(), conditions, composite, record_layer=[runtime_layer], exclude_parallel=True
    )
    attribution.take_prediction = 0

    # Reference images
    print("[refs] computing/loading reference images")
    resolved_ref_layer, relmax_path = _resolve_crp_relmax_layer(output_dir_crp, resolved_layer)
    if resolved_ref_layer != runtime_layer:
        print(f"[refs] remapped CRP layer '{runtime_layer}' -> '{resolved_ref_layer}'")
        _ensure_crp_relmax_alias(output_dir_crp, resolved_ref_layer, runtime_layer)
        resolved_ref_layer = runtime_layer
        relmax_path = os.path.join(output_dir_crp, "RelMax_sum_normed", f"{runtime_layer}_data.npy")
    if not os.path.exists(relmax_path):
        raise FileNotFoundError(
            f"CRP RelMax file not found for layer '{runtime_layer}'. "
            f"Tried: {[os.path.join(output_dir_crp, 'RelMax_sum_normed', f'{c}_data.npy') for c in _layer_candidates(runtime_layer)]}"
        )
    ref_imgs = get_ref_images(
        fv=fv, topk_ind=topk_ind, layer_name=resolved_ref_layer,
        composite=composite, n_ref=n_refimgs, ref_imgs_save_path=ref_imgs_path
    )

    # Fit fresh GMMs each run so prototype selection adapts to current statistics
    print("[gmm] fitting GMM for current run")
    gmm: GaussianMixture = GaussianMixture(
        n_components=num_prototypes, reg_covar=1e-5, random_state=0
    ).fit(attributions_np)
    print("[gmm] building prototype component GMMs")
    prototype_gmms: List[GaussianMixture] = [
        GaussianMixture(n_components=1, covariance_type='full') for _ in range(num_prototypes)
    ]
    params = gmm._get_parameters()  # weights, means, covariances, precisions_cholesky (impl-dependent)
    for p, g_ in enumerate(prototype_gmms):
        # copy single-component parameters into 1-comp GMM
        g_._set_parameters([
            params[0][p:p+1] * 0 + 1,     # weights_ -> set to 1
            params[1][p:p+1],             # means_
            params[2][p:p+1],             # covariances_ (for full)
            *([params[3][p:p+1]] if len(params) > 3 else [])
        ])

    # Likelihoods and prototype selection
    print("[gmm] scoring full dataset and current sample")
    scores = gmm.score_samples(attributions_np)  # [N]
    sample_vec_t = channel_rels_vec.detach().cpu()
    if torch.isnan(sample_vec_t).any() or torch.isinf(sample_vec_t).any():
        count_bad = int((~torch.isfinite(sample_vec_t)).sum().item())
        print(f"[gmm] WARNING: sample channel relevances contain {count_bad} non-finite values; replacing with zeros.")
        sample_vec_t = torch.nan_to_num(sample_vec_t, nan=0.0, posinf=0.0, neginf=0.0)
    sample_vec = sample_vec_t.numpy().reshape(1, -1)  # [1,C]
    score_sample = float(gmm.score_samples(sample_vec)[0])
    proto_likes = [float(g_.score_samples(sample_vec)[0]) for g_ in prototype_gmms]
    best_proto = int(np.argmax(proto_likes))
    mean = torch.from_numpy(gmm.means_[best_proto])  # [C]
    print(f"[gmm] sample score={score_sample:.4f}, best_proto={best_proto}")

    # Pick the dataset sample closest to the chosen prototype
    print("[gmm] finding closest sample to prototype mean")
    attributions_t = torch.from_numpy(attributions_np)  # [N,C]
    dists = (attributions_t - mean[None, :]).pow(2).sum(dim=1)  # [N]
    closest_idx = int(torch.argmin(dists).item())
    print(f"[gmm] closest_idx={closest_idx}")

    # Prototype image and cond heatmaps
    data_p, _ = dataset[closest_idx]
    data_p = data_p.to(device)[None]
    print("[proto] computing prototype attributions")
    attr_p = attribution(data_p.requires_grad_(), [{"y": 1}], composite, record_layer=[runtime_layer])
    cond_heatmap_p, _, _, _ = attribution(data_p.requires_grad_(), conditions, composite)
    proto_heatmap_pil = zimage.imgify(attr_p.heatmap, cmap="bwr", symmetric=True, level=10)

    # Prototype mask
    print("[proto] building prototype mask")
    mask_p_bool = (attr_p.prediction[0].argmax(dim=0) == 1).detach().cpu()  # [h,w]
    mask_p_f = mask_p_bool.float().unsqueeze(0).unsqueeze(0)
    mask_p_resized = F.interpolate(mask_p_f, size=tgt_hw, mode="nearest").bool().squeeze(0).squeeze(0)

    # Prototype visualization
    pil_p = imgify(fv.get_data_sample(closest_idx, preprocessing=False)[0][0])
    img_p_t = TF.pil_to_tensor(pil_p)  # [C,H,W] uint8
    img_prototype_overlay = draw_segmentation_masks(img_p_t, masks=mask_p_resized, alpha=0.2, colors=["red"])

    # Input visualization
    pil_in = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
    img_in_t = TF.pil_to_tensor(pil_in)  # [C,H,W]
    img_in_overlay = draw_segmentation_masks(img_in_t, masks=mask_resized, alpha=0.2, colors=["red"])

    # -------------------------
    # Plotting
    # -------------------------
    print("[plot] assembling figure")
    n_rows = max(n_display_concepts, 4)
    concept_grid_cols = max(1, int(n_refimgs))
    concept_tile_size = 150
    concept_col_width = max(3.2, float(n_refimgs))
    width_ratios = [1.0, 1.0, concept_col_width, 1.05, 1.05, 1.0]

    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(14.5 + 1.2 * max(0, n_refimgs - 3), 2.15 * n_rows),
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.12, 'hspace': 0.34},
        facecolor="white",
        dpi=200
    )

    # Resize for plotting
    resize = torchvision.transforms.Resize((150, 150))
    resize_concept = torchvision.transforms.Resize((concept_tile_size, concept_tile_size))

    img_in_res = resize(img_in_overlay)
    base_heatmap_res = resize(heatmap_pil)
    proto_img_res = resize(img_prototype_overlay)
    proto_heatmap_res = resize(proto_heatmap_pil)

    # Percentile for outlier flag
    p1 = np.percentile(scores, 1)
    flood_segment_ratio = float(mask_resized.float().mean().item())
    flood_segment_too_small = flood_segment_ratio < 0.005  # <0.5% of pixels predicted as flood
    flood_segment_ratio_proto = float(mask_p_resized.float().mean().item())
    flood_segment_too_small_proto = flood_segment_ratio_proto < 0.005  # <0.5% of pixels predicted as flood
    relevance_too_small_thr = 1e-3  # 0.1%
    print(
        f"[plot] flood_segment_ratio={flood_segment_ratio:.6f}, "
        f"flood_segment_ratio_proto={flood_segment_ratio_proto:.6f}, "
        f"flood_segment_too_small={flood_segment_too_small}, "
        f"flood_segment_too_small_proto={flood_segment_too_small_proto}, "
        f"relevance_too_small_thr={relevance_too_small_thr:.6f}"
    )

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):
            ax.set_xticks([]); ax.set_yticks([])

            # If r exceeds actual concepts, hide the row
            if r >= n_display_concepts and c not in (0,):
                ax.axis("off")
                continue

            for spine in ax.spines.values():
                spine.set_visible(False)

            if c == 0:
                if r == 0:
                    ax.set_title("input", fontsize=12, pad=8)
                    ax.imshow(img_in_res.permute(1, 2, 0).cpu().numpy())
                elif r == 1:
                    ax.set_title("heatmap", fontsize=12, pad=8)
                    if flood_segment_too_small:
                        _show_message_box(ax, "Flood segment\nis too small")
                    else:
                        ax.imshow(base_heatmap_res)
                elif r == 2:
                    ax.set_title("class likelihood", fontsize=12, pad=8)
                    h = ax.hist(scores, bins=20, color='k')
                    ax.vlines(score_sample, 0, (h[0].max() if len(h[0]) else 1), linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_xticks([]); ax.set_yticks([])
                    lbl = "Outlier" if score_sample < p1 else "Ordinary"
                    color = "red" if lbl == "Outlier" else "green"
                    ax.text(
                        0.5, -0.35, lbl, transform=ax.transAxes, ha="center",
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor=color, alpha=0.25, linewidth=3)
                    )
                else:
                    ax.axis("off")

            elif c == 1:
                if r == 0:
                    ax.set_title("cond. heatmap", fontsize=12, pad=8)
                # cond heatmap for r-th concept
                concept_rel = float(topk_rel[r])
                if concept_rel <= relevance_too_small_thr:
                    _show_message_box(ax, "Relevance is\ntoo small")
                else:
                    hm_pil = zimage.imgify(cond_heatmaps[r], symmetric=True, cmap="bwr", level=10)
                    hm_res = resize(hm_pil)
                    ax.imshow(hm_res)
                ax.set_ylabel(f"concept {topk_ind[r]}\n{(topk_rel[r] * 100):.1f}%", fontsize=10, labelpad=8)

            elif c == 2:
                if r == 0:
                    ax.set_title("concept visualizations", fontsize=12, pad=8)
                # Show reference images grid for concept r
                if ref_imgs and int(topk_ind[r]) in ref_imgs and len(ref_imgs[int(topk_ind[r])]) > 0:
                    concept_images = _select_diverse_reference_images(ref_imgs[int(topk_ind[r])], n_refimgs)
                    resized_refs = []
                    for im in concept_images:
                        # Convert PIL -> Tensor [C,H,W]
                        t = TF.pil_to_tensor(resize_concept(im))
                        resized_refs.append(t)
                    if resized_refs:
                        grid = make_grid(
                            resized_refs,
                            nrow=concept_grid_cols,
                            padding=0,
                            pad_value=255,
                        )
                        grid_np = np.array(zimage.imgify(grid.detach().cpu()))
                        ax.imshow(grid_np)
                        ax.set_facecolor("white")
                else:
                    ax.axis("off")

            elif c == 3:
                if r == 0:
                    ax.set_title("Difference to prot.", fontsize=12, pad=8)
                ax.imshow(np.ones((150, 150, 3)), alpha=1.0)
                delta_R = (float(channel_rels_vec[topk_ind[r]].round(decimals=3)) - float(mean[topk_ind[r]].round(decimals=3))) * 100.0
                if delta_R > 2:
                    textstr = f"ΔR = {delta_R:+.1f}%\nover-used"
                    edge_color = "#c43d2f"
                    fill_color = "#fdecea"
                elif delta_R < -2:
                    textstr = f"ΔR = {delta_R:+.1f}%\nunder-used"
                    edge_color = "#c43d2f"
                    fill_color = "#fdecea"
                else:
                    textstr = f"ΔR = {delta_R:+.1f}%\nsimilar"
                    edge_color = "#1f9d55"
                    fill_color = "#e7f6ec"

                rect = patches.Rectangle((0, 0), 150, 150, linewidth=2.5, edgecolor=edge_color, facecolor='white')
                ax.add_patch(rect)
                lines = textstr.split('\n')
                ax.text(
                    75, 55, lines[1], fontproperties=FontProperties(weight='bold'),
                    va='center', ha='center', color=edge_color, fontsize=11
                )
                ax.text(
                    75, 88, lines[0], fontsize=10, va='center', ha='center',
                    bbox=dict(facecolor=fill_color, edgecolor='none', alpha=1.0, boxstyle="round,pad=0.25")
                )
                ax.set_xlim([0, 150]); ax.set_ylim([0, 150]); ax.axis("off")

            elif c == 4:
                if r == 0:
                    ax.set_title("Prot localization", fontsize=12, pad=8)
                proto_concept_rel = float(mean[topk_ind[r]])
                if proto_concept_rel <= relevance_too_small_thr:
                    _show_message_box(ax, "Relevance is\ntoo small")
                else:
                    hm_pil_p = zimage.imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", level=10)
                    ax.imshow(resize(hm_pil_p))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"concept {topk_ind[r]}\n{(float(mean[topk_ind[r]]) * 100):.1f}%", fontsize=10, labelpad=8)

            elif c == 5:
                if r == 0:
                    ax.set_title("prototype", fontsize=12, pad=8)
                    ax.imshow(proto_img_res.permute(1, 2, 0).cpu().numpy())
                elif r == 1:
                    ax.set_title("heatmap", fontsize=12, pad=8)
                    if flood_segment_too_small_proto:
                        _show_message_box(ax, "Flood segment\nis too small")
                    else:
                        ax.imshow(proto_heatmap_res)
                else:
                    ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    ####################################################################################################################

    # plot_dir = "../output/pcx/pcx_plots"

    # # --- make & save the 2D UMAP plot ---
    # fig2d, umap_path = plot_gmm_umap_2d(
    #     attributions=attributions_np,
    #     gmm=gmm,
    #     channel_rels=channel_rels,
    #     class_id=1,
    #     sample_id=sample_id,
    #     layer_name=layer,
    #     split="train",
    #     plot_dir=plot_dir,
    #     input_prediction_num=0
    # )
    # print("Saved 2D UMAP plot to:", umap_path)
    #
    # fig3d, umap3d_html_path = plot_gmm_umap_3d(
    #     attributions=attributions_np,
    #     gmm=gmm,
    #     channel_rels=channel_rels,
    #     class_id=1,
    #     sample_id=sample_id,
    #     layer_name=layer,
    #     split="train",
    #     plot_dir=plot_dir,
    #     input_prediction_num=0
    # )
    # print("Saved 3D UMAP HTML to:", umap3d_html_path)
    #
    # fig_neighbors, neighbors_path, neighbors_indices = plot_prototype_n_nearest(
    #     attributions_np=attributions_np,
    #     gmm=gmm,
    #     dataset=dataset,
    #     attribution=attribution,
    #     composite=composite,
    #     fv=fv,
    #     layer=layer,
    #     class_id=1,
    #     n_top=5,
    #     device=device,
    #     save_dir="../output/pcx/pcx_plots",
    #     split="train",
    #     input_prediction_num=0
    # )
    # print("Saved prototype-neighborhood grid to:", neighbors_path)

    # # --- save the main PCX grid ---
    # pcx_fname = f"pcx_layer-{layer.replace('.', '-')}_sid-{sample_id}_nprot-{num_prototypes}.png"
    # pcx_path = os.path.join(plot_dir, pcx_fname)
    # fig.savefig(pcx_path, dpi=200, bbox_inches="tight")
    # print("Saved PCX grid to:", pcx_path)

    ####################################################################################################################

    plt.tight_layout()
    return fig
