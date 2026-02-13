import os
import sys
import gc
import time
import h5py
import copy
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
def _as_uint8_array(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

# -------------------------
# Reference images cache (HDF5)
# -------------------------

def get_ref_images(fv, topk_ind, layer_name, composite, n_ref=12, ref_imgs_save_path="output/ref_imgs/"):
    ref_imgs_save_path = os.path.join(ref_imgs_save_path, f"{layer_name}.h5")
    os.makedirs(os.path.dirname(ref_imgs_save_path), exist_ok=True)

    ref_imgs = {}
    missing_keys = list(map(str, topk_ind))

    if os.path.exists(ref_imgs_save_path):
        with h5py.File(ref_imgs_save_path, "a") as f:
            existing_keys = set(f.keys())
            missing_keys = [str(k) for k in topk_ind if str(k) not in existing_keys]

            for k in topk_ind:
                str_k = str(k)
                if str_k in f:
                    group = f[str_k]
                    ref_imgs[int(str_k)] = [Image.fromarray(group[str(idx)][:]) for idx in sorted(group.keys(), key=int)]

            if missing_keys:
                print(f"Calculating and saving missing reference images for keys: {missing_keys}")
                new_refs = fv.get_max_reference([int(k) for k in missing_keys], layer_name, "relevance", (0, n_ref),
                                                composite=composite, rf=True, plot_fn=vis_opaque_img_border)
                for key, images_list in new_refs.items():
                    group = f.create_group(str(key))
                    assert len(images_list) >= n_ref
                    ref_imgs[key] = []
                    for idx, image in enumerate(images_list[:n_ref]):
                        if isinstance(image, Image.Image):
                            arr = np.array(image)
                            group.create_dataset(str(idx), data=arr)
                            ref_imgs[key].append(image)
                        else:
                            print(f"Warning: Item '{idx}' in key '{key}' is not a PIL image and will not be saved.")
    else:
        print("Reference image file does not exist, calculating all.")
        ref_imgs = fv.get_max_reference(topk_ind, layer_name, "relevance", (0, n_ref),
                                        composite=composite, rf=True, plot_fn=vis_opaque_img_border)
        with h5py.File(ref_imgs_save_path, "w") as f:
            for key, images_list in ref_imgs.items():
                group = f.create_group(str(key))
                assert len(images_list) >= n_ref
                for idx, image in enumerate(images_list[:n_ref]):
                    if isinstance(image, Image.Image):
                        arr = np.array(image)
                        group.create_dataset(str(idx), data=arr)
                    else:
                        print(f"Warning: Item '{idx}' in key '{key}' is not a PIL image and will not be saved.")

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
    outlier_percentile: float = 0.10):
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
        outlier_percentile=outlier_percentile,
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
    outlier_percentile: float = 1.0,
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
    attr = attribution(x_copy, condition, composite, record_layer=[layer], init_rel=1)

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
    channel_rels = ChannelConcept().attribute(attr.relevances[layer], abs_norm=True)  # [1,C]
    channel_rels_vec = channel_rels[0]  # [C]
    print(f"[concepts] channel_rels shape={tuple(channel_rels.shape)}")

    # Top-k concepts
    topk = torch.topk(channel_rels_vec, k=n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    topk_rel = topk.values.detach().cpu().numpy()
    print(f"[concepts] topk_ind={topk_ind.tolist()} topk_rel={np.round(topk_rel,3).tolist()}")

    # Conditional heatmaps per concept
    conditions = [{"y": 1, layer: int(c)} for c in topk_ind]
    print("[attr] computing conditional heatmaps")
    attribution.take_prediction = 0
    cond_heatmaps, _, _, _ = attribution(x.requires_grad_(), conditions, composite, exclude_parallel=True)
    attribution.take_prediction = 0

    # Reference images
    print("[refs] computing/loading reference images")
    ref_imgs = get_ref_images(
        fv=fv, topk_ind=topk_ind, layer_name=layer,
        composite=composite, n_ref=n_refimgs, ref_imgs_save_path=ref_imgs_path
    )

    # Load attributions matrix for GMM
    attr_path = os.path.join(output_dir_pcx, layer, "attributions.npy")
    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"Attributions file not found: {attr_path}")
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

    # Pick the dataset sample closest to the CURRENT SAMPLE
    print("[gmm] finding closest sample to current input sample")
    attributions_t = torch.from_numpy(attributions_np)  # [N,C]
    sample_vec_torch = torch.from_numpy(sample_vec).squeeze(0)  # [C]
    dists = (attributions_t - sample_vec_torch[None, :]).pow(2).sum(dim=1)  # [N]
    closest_idx = int(torch.argmin(dists).item())
    print(f"[gmm] closest_idx={closest_idx} (closest to input sample)")

    # Prototype image and cond heatmaps
    data_p, _ = dataset[closest_idx]
    data_p = data_p.to(device)[None]
    print("[proto] computing prototype attributions")
    attr_p = attribution(data_p.requires_grad_(), [{"y": 1}], composite, record_layer=[layer])
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
    n_rows = max(n_concepts, 4)
    width_ratios = [1, 1, n_refimgs / 4 if n_refimgs else 1, 1, 1, 1]

    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(4 * max(1, n_refimgs / 4), 1.8 * n_rows),
        gridspec_kw={'width_ratios': width_ratios},
        dpi=200
    )

    # Resize for plotting
    resize = torchvision.transforms.Resize((150, 150))

    img_in_res = resize(img_in_overlay)
    base_heatmap_res = resize(heatmap_pil)
    proto_img_res = resize(img_prototype_overlay)
    proto_heatmap_res = resize(proto_heatmap_pil)

    # Percentile for outlier flag (configurable via outlier_percentile parameter)
    p_threshold = np.percentile(scores, outlier_percentile)

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):
            ax.set_xticks([]); ax.set_yticks([])

            # If r exceeds actual concepts, hide the row
            if r >= n_concepts and c not in (0,):
                ax.axis("off")
                continue

            if c == 0:
                if r == 0:
                    ax.set_title("input")
                    ax.imshow(img_in_res.permute(1, 2, 0).cpu().numpy())
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(base_heatmap_res)
                elif r == 2:
                    ax.set_title("class likelihood")
                    h = ax.hist(scores, bins=20, color='k')
                    ax.vlines(score_sample, 0, (h[0].max() if len(h[0]) else 1), linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_xticks([]); ax.set_yticks([])
                    lbl = "Outlier" if score_sample < p_threshold else "Ordinary"
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
                    ax.set_title("cond. heatmap")
                # cond heatmap for r-th concept
                hm_pil = zimage.imgify(cond_heatmaps[r], symmetric=True, cmap="bwr", level=10)
                hm_res = resize(hm_pil)
                ax.imshow(hm_res)
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):.1f}%")

            elif c == 2:
                if r == 0:
                    ax.set_title("concept visualizations")
                # Show reference images grid for concept r
                if ref_imgs and int(topk_ind[r]) in ref_imgs and len(ref_imgs[int(topk_ind[r])]) > 0:
                    resized_refs = []
                    for im in ref_imgs[int(topk_ind[r])][:n_refimgs]:
                        # Convert PIL -> Tensor [C,H,W]
                        t = TF.pil_to_tensor(resize(im))
                        resized_refs.append(t)
                    if resized_refs:
                        grid = make_grid(resized_refs, nrow=max(1, int(n_refimgs / 2)), padding=0)
                        grid_np = np.array(zimage.imgify(grid.detach().cpu()))
                        ax.imshow(grid_np)
                else:
                    ax.axis("off")

            elif c == 3:
                if r == 0:
                    ax.set_title("Difference to prot.")
                ax.imshow(np.zeros((150, 150, 3)), alpha=0.2)
                delta_R = (float(channel_rels_vec[topk_ind[r]].round(decimals=3)) - float(mean[topk_ind[r]].round(decimals=3))) * 100.0
                if delta_R > 5:
                    textstr = f"ΔR = {delta_R:+.1f}%\n⚠ over-used"
                    edge_color = "#ff0000"
                elif delta_R < -5:
                    textstr = f"ΔR = {delta_R:+.1f}%\n⚠ under-used"
                    edge_color = "#ff0000"
                else:
                    textstr = f"ΔR = {delta_R:+.1f}%\n✓ similar"
                    edge_color = "#00cc00"

                rect = patches.Rectangle((0, 0), 150, 150, linewidth=3, edgecolor=edge_color, facecolor='white')
                ax.add_patch(rect)
                lines = textstr.split('\n')
                ax.text(75, 60, lines[0], fontsize=10, va='center', ha='center',
                        bbox=dict(facecolor=edge_color, edgecolor='none', alpha=0.2))
                ax.text(75, 90, lines[1], fontproperties=FontProperties(weight='bold'), va='center', ha='center', color=edge_color)
                ax.set_xlim([0, 150]); ax.set_ylim([0, 150]); ax.axis("off")

            elif c == 4:
                if r == 0:
                    ax.set_title("Prot localization")
                hm_pil_p = zimage.imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", level=10)
                ax.imshow(resize(hm_pil_p))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(float(mean[topk_ind[r]]) * 100):.1f}%")

            elif c == 5:
                if r == 0:
                    ax.set_title("prototype")
                    ax.imshow(proto_img_res.permute(1, 2, 0).cpu().numpy())
                elif r == 1:
                    ax.set_title("heatmap")
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
