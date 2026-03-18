import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import draw_segmentation_masks
from sklearn.mixture import GaussianMixture

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, load_maximization
from crp.image import imgify
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from LCRP.utils.render import vis_opaque_img_border

from src.plot_pcx_pidnet_new import get_ref_images


def _stack_images_vert(images: Sequence[Image.Image], size: Tuple[int, int]) -> np.ndarray:
    resized = [im.resize(size, Image.BICUBIC) for im in images]
    arrs = [np.asarray(im) for im in resized]
    return np.vstack(arrs) if arrs else np.zeros((size[1], size[0], 3), dtype=np.uint8)


def _stack_images_horiz(images: Sequence[Image.Image], size: Tuple[int, int]) -> np.ndarray:
    resized = [im.resize(size, Image.BICUBIC) for im in images]
    arrs = [np.asarray(im) for im in resized]
    return np.hstack(arrs) if arrs else np.zeros((size[1], size[0], 3), dtype=np.uint8)


def _fill_missing_samples(
    proto_samples: List[List[int]],
    pool: List[int],
    X: np.ndarray,
    samples_per_proto: int,
) -> None:
    used = set(s for lst in proto_samples for s in lst)

    def pick_farthest(chosen: set, candidates: List[int]) -> Optional[int]:
        if not candidates:
            return None
        if not chosen:
            return candidates.pop(0)
        dists = np.min(
            np.linalg.norm(X[candidates][:, None, :] - X[list(chosen)][None, :, :], axis=2),
            axis=1,
        )
        pick_idx = int(np.argmax(dists))
        return candidates.pop(pick_idx)

    for k in range(len(proto_samples)):
        while len(proto_samples[k]) < samples_per_proto and pool:
            pick = pick_farthest(used, pool)
            if pick is None:
                break
            proto_samples[k].append(pick)
            used.add(pick)


def _select_proto_samples(
    attributions: np.ndarray,
    num_prototypes: int,
    samples_per_proto: int,
    random_state: int = 0,
) -> Tuple[List[List[int]], GaussianMixture]:
    gmm = GaussianMixture(
        n_components=num_prototypes, reg_covar=1e-5, random_state=random_state
    ).fit(attributions)
    labels = gmm.predict(attributions)
    centers = gmm.means_

    proto_samples: List[List[int]] = [[] for _ in range(num_prototypes)]
    for k in range(num_prototypes):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(attributions[idx] - centers[k], axis=1)
        topk = idx[np.argsort(d)[:samples_per_proto]]
        proto_samples[k] = list(map(int, topk))

    remaining = [i for i in range(len(attributions)) if i not in {s for lst in proto_samples for s in lst}]
    _fill_missing_samples(proto_samples, remaining, attributions, samples_per_proto)
    return proto_samples, gmm


def plot_proto_concept_matrix(
    model_name: str,
    model: torch.nn.Module,
    dataset,
    layer_name: str,
    attributions: np.ndarray,
    num_prototypes: int,
    n_concepts: int = 5,
    n_refimgs: int = 3,
    samples_per_proto: int = 3,
    class_id: int = 1,
    concept_ids: Optional[Sequence[int]] = None,
    concept_seed_ids: Optional[Sequence[int]] = None,
    concept_fill: bool = False,
    skip_prototypes: Optional[Iterable[int]] = None,
    plot_top_k_prototypes: Optional[int] = None,
    ref_imgs_path: str = "output/ref_imgs_pidnet/",
    output_dir_crp: str = "output/crp/pidnet_flood/",
    proto_thumb_size: Tuple[int, int] = (140, 140),
    ref_thumb_size: Tuple[int, int] = (140, 140),
    concept_ref_scale: float = 1.0,
    concept_row_scale: float = 1.0,
    ref_col_scale: float = 1.0,
    value_scale: float = 1.0,
    overlay_masks: bool = True,
    mask_alpha: float = 0.25,
    concept_select_by: str = "relevance",
    concept_pool_size: int = 50,
    ref_select_by: str = "relevance",
    ref_pool_size: int = 50,
    ref_overlay_masks: bool = False,
    ref_mask_alpha: float = 0.25,
    ref_vis_th: float = 0.05,
    proto_random_state: int = 0,
    sort_prototypes_by: Optional[str] = "size",
    proto_samples_override: Optional[List[List[int]]] = None,
) -> Tuple[plt.Figure, Dict[str, np.ndarray]]:
    """
    Build a prototype-vs-concept matrix plot with prototype image stacks on top
    and concept reference strips on the left.
    """
    device = next(model.parameters()).device
    model.eval()

    attributions_np = np.asarray(attributions, dtype=np.float32)
    attributions_np = np.nan_to_num(attributions_np, nan=0.0, posinf=0.0, neginf=0.0)

    if proto_samples_override is not None:
        proto_samples = list(proto_samples_override)
    else:
        proto_samples, _ = _select_proto_samples(
            attributions_np, num_prototypes=num_prototypes, samples_per_proto=samples_per_proto,
            random_state=proto_random_state,
        )

    skip = set(skip_prototypes or [])
    proto_indices = [p for p in range(num_prototypes) if p not in skip and p < len(proto_samples) and len(proto_samples[p]) > 0]

    # Optionally sort prototype columns for more stable/meaningful ordering
    if sort_prototypes_by == "size":
        proto_indices = sorted(proto_indices, key=lambda p: len(proto_samples[p]), reverse=True)

    # If requested, plot only the most populated prototypes
    if plot_top_k_prototypes is not None and plot_top_k_prototypes > 0:
        proto_indices = sorted(
            proto_indices, key=lambda p: len(proto_samples[p]), reverse=True
        )[:plot_top_k_prototypes]

    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    fv = VISUALIZATIONS[model_name](
        attribution,
        dataset,
        layer_names,
        preprocess_fn=lambda x: x,
        path=output_dir_crp,
        max_target="max",
    )

    def sample_channel_rels(sample_id: int) -> torch.Tensor:
        x, _ = dataset[sample_id]
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim == 3 and x.shape[0] != 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        x = x.to(device).float().requires_grad_()
        attr = attribution(x.unsqueeze(0), [{"y": class_id}], composite, record_layer=[layer_name])
        rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)[0]
        return rels.detach().cpu()

    def sample_overlay_image(sample_id: int) -> Image.Image:
        # Prefer raw RGB via fv to avoid normalized/augmented appearance
        try:
            img_uint8 = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
        except Exception:
            img_tensor, _ = dataset[sample_id]
            img_uint8 = TF.to_pil_image(img_tensor[:3].detach().cpu())
        if not overlay_masks:
            return img_uint8
        img_tensor, _ = dataset[sample_id]
        x = img_tensor.to(device).requires_grad_()
        attr = attribution(x.unsqueeze(0), [{"y": class_id}], composite, record_layer=[layer_name])
        pred = attr.prediction[0].argmax(dim=0).detach().cpu().float()  # [h,w]
        mask = pred == class_id
        # resize mask to raw image size if needed
        if mask.shape != (img_uint8.height, img_uint8.width):
            mask_f = mask.float().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(
                mask_f, size=(img_uint8.height, img_uint8.width), mode="nearest"
            ).bool().squeeze(0).squeeze(0)
        img_uint8_t = TF.pil_to_tensor(img_uint8)
        over = draw_segmentation_masks(img_uint8_t, masks=mask, alpha=mask_alpha, colors=["red"])
        return TF.to_pil_image(over)

    def sample_overlay_image_with_alpha(sample_id: int, alpha: float) -> Image.Image:
        try:
            img_uint8 = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
        except Exception:
            img_tensor, _ = dataset[sample_id]
            img_uint8 = TF.to_pil_image(img_tensor[:3].detach().cpu())
        img_tensor, _ = dataset[sample_id]
        x = img_tensor.to(device).requires_grad_()
        attr = attribution(x.unsqueeze(0), [{"y": class_id}], composite, record_layer=[layer_name])
        pred = attr.prediction[0].argmax(dim=0).detach().cpu().float()
        mask = pred == class_id
        if mask.shape != (img_uint8.height, img_uint8.width):
            mask_f = mask.float().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(
                mask_f, size=(img_uint8.height, img_uint8.width), mode="nearest"
            ).bool().squeeze(0).squeeze(0)
        img_uint8_t = TF.pil_to_tensor(img_uint8)
        over = draw_segmentation_masks(img_uint8_t, masks=mask, alpha=alpha, colors=["red"])
        return TF.to_pil_image(over)

    def sample_raw_image(sample_id: int) -> Image.Image:
        try:
            img_uint8 = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
            return img_uint8
        except Exception:
            img_tensor, _ = dataset[sample_id]
            return TF.to_pil_image(img_tensor[:3].detach().cpu())

    def _get_input_tensor(sample_id: int) -> torch.Tensor:
        x, _ = dataset[sample_id]
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim == 3 and x.shape[0] != 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        return x.to(device).float()

    def _get_raw_tensor(sample_id: int) -> torch.Tensor:
        try:
            raw = fv.get_data_sample(sample_id, preprocessing=False)[0]
            if isinstance(raw, torch.Tensor) and raw.ndim == 4:
                raw = raw[0]
            if isinstance(raw, torch.Tensor):
                t = raw[:3].detach().cpu().float()
                if t.max() > 1.5:
                    t = t / 255.0
                return t
        except Exception:
            pass
        img_tensor, _ = dataset[sample_id]
        t = img_tensor[:3].detach().cpu().float()
        if t.max() > 1.5:
            t = t / 255.0
        return t

    def _concept_overlay_images(concept_id: int, sample_ids: List[int]) -> List[Image.Image]:
        if not sample_ids:
            return []
        data_batch = torch.stack([_get_input_tensor(sid) for sid in sample_ids], dim=0)
        heatmaps = fv._attribution_on_reference(
            data_batch, concept_id, layer_name, composite, rf=False, targets=None
        ).detach().cpu()
        raw_batch = torch.stack([_get_raw_tensor(sid) for sid in sample_ids], dim=0)
        return vis_opaque_img_border(
            raw_batch, heatmaps, rf=False, alpha=ref_mask_alpha, vis_th=ref_vis_th
        )

    def _pred_area(sample_id: int) -> float:
        x, _ = dataset[sample_id]
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim == 3 and x.shape[0] != 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        x = x.to(device).float()
        with torch.no_grad():
            out = model(x.unsqueeze(0))
            if isinstance(out, (list, tuple)):
                out = out[0]
            pred = out.argmax(dim=1)[0]
        return float((pred == class_id).float().mean().item())

    proto_rels: Dict[int, torch.Tensor] = {}
    for p in proto_indices:
        rels = []
        for sid in proto_samples[p]:
            rels.append(sample_channel_rels(sid))
        proto_rels[p] = torch.stack(rels, dim=0).mean(dim=0) if rels else None

    seed_ids: List[int] = []
    if concept_seed_ids:
        seen = set()
        for cid in concept_seed_ids:
            if cid not in seen:
                seed_ids.append(int(cid))
                seen.add(int(cid))

    if concept_ids is None:
        agg = None
        for p in proto_indices:
            if proto_rels[p] is None:
                continue
            agg = proto_rels[p] if agg is None else agg + proto_rels[p]
        if agg is None:
            raise ValueError("No prototype relevances computed; check attribution inputs.")
        pool_k = min(max(n_concepts, concept_pool_size), int(agg.numel()))
        topk = torch.topk(agg, k=pool_k)
        candidate_ids = [int(i) for i in topk.indices.tolist()]
        if concept_select_by == "seg_area":
            try:
                d_c_sorted, _, _ = load_maximization(fv.RelMax.PATH, layer_name)
            except Exception:
                d_c_sorted = None
            if d_c_sorted is not None:
                scores = []
                for cid in candidate_ids:
                    idxs = d_c_sorted[0:n_refimgs, cid]
                    areas = [_pred_area(int(sid)) for sid in idxs]
                    score = float(np.mean(areas)) if areas else 0.0
                    scores.append((cid, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                concept_ids = [cid for cid, _ in scores[:n_concepts]]
            else:
                concept_ids = candidate_ids[:n_concepts]
        else:
            concept_ids = candidate_ids[:n_concepts]

    base_ids = list(concept_ids or [])
    if seed_ids:
        base_ids = seed_ids + [cid for cid in base_ids if cid not in seed_ids]

    if (concept_ids is None) or concept_fill:
        needed = max(0, n_concepts - len(base_ids))
        if needed > 0:
            agg = None
            for p in proto_indices:
                if proto_rels[p] is None:
                    continue
                agg = proto_rels[p] if agg is None else agg + proto_rels[p]
            if agg is not None:
                pool_k = min(max(n_concepts, concept_pool_size), int(agg.numel()))
                topk = torch.topk(agg, k=pool_k)
                candidate_ids = [int(i) for i in topk.indices.tolist()]
            else:
                candidate_ids = []
            candidate_ids = [cid for cid in candidate_ids if cid not in base_ids]
            if concept_select_by == "seg_area" and candidate_ids:
                try:
                    d_c_sorted, _, _ = load_maximization(fv.RelMax.PATH, layer_name)
                except Exception:
                    d_c_sorted = None
                if d_c_sorted is not None:
                    scores = []
                    for cid in candidate_ids:
                        idxs = d_c_sorted[0:n_refimgs, cid]
                        areas = [_pred_area(int(sid)) for sid in idxs]
                        score = float(np.mean(areas)) if areas else 0.0
                        scores.append((cid, score))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    candidate_ids = [cid for cid, _ in scores]
            base_ids.extend(candidate_ids[:needed])

    concept_ids = list(base_ids)[:n_concepts]

    # reference images for concepts
    ref_imgs: Dict[int, List[Image.Image]] = {}
    if ref_select_by == "seg_area":
        try:
            d_c_sorted, _, _ = load_maximization(fv.RelMax.PATH, layer_name)
        except Exception:
            d_c_sorted = None
        if d_c_sorted is not None:
            pool_k = max(n_refimgs, ref_pool_size)
            for cid in concept_ids:
                cand = list(map(int, d_c_sorted[0:pool_k, int(cid)]))
                scored = [(sid, _pred_area(int(sid))) for sid in cand]
                scored.sort(key=lambda x: x[1], reverse=True)
                pick = [sid for sid, _ in scored[:n_refimgs]]
                if ref_overlay_masks:
                    ref_imgs[int(cid)] = _concept_overlay_images(int(cid), [int(s) for s in pick])
                else:
                    ref_imgs[int(cid)] = [sample_raw_image(int(sid)) for sid in pick]
        else:
            ref_imgs = get_ref_images(
                fv=fv,
                topk_ind=np.array(concept_ids, dtype=int),
                layer_name=layer_name,
                composite=composite,
                n_ref=n_refimgs,
                ref_imgs_save_path=ref_imgs_path,
            )
    else:
        ref_imgs = get_ref_images(
            fv=fv,
            topk_ind=np.array(concept_ids, dtype=int),
            layer_name=layer_name,
            composite=composite,
            n_ref=n_refimgs,
            ref_imgs_save_path=ref_imgs_path,
        )

    # matrix values
    values = np.zeros((len(concept_ids), len(proto_indices)), dtype=np.float32)
    for c_idx, cid in enumerate(concept_ids):
        for p_idx, p in enumerate(proto_indices):
            rel_vec = proto_rels.get(p)
            if rel_vec is None:
                values[c_idx, p_idx] = 0.0
            else:
                values[c_idx, p_idx] = float(rel_vec[cid].item()) * float(value_scale)

    vmax = float(values.max()) if values.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    n_rows = 1 + len(concept_ids)
    n_cols = 1 + len(proto_indices)
    size_scale_w = max(ref_thumb_size[0], proto_thumb_size[0]) / 140.0
    size_scale_h = max(ref_thumb_size[1], proto_thumb_size[1]) / 140.0
    fig_w = (1.6 * (len(proto_indices) + ref_col_scale) + 1.0) * size_scale_w
    fig_h = (1.2 * (1 + len(concept_ids) * concept_row_scale) + 1.0) * size_scale_h
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        dpi=220,
        facecolor="white",
        gridspec_kw={
            "height_ratios": [samples_per_proto] + [1 * concept_row_scale] * len(concept_ids),
            "width_ratios": [ref_col_scale] + [1] * len(proto_indices),
        },
    )
    axs = np.atleast_2d(axs)

    # top-left label
    axs[0, 0].axis("off")
    axs[0, 0].text(0.0, 1.0, "Concept refs (RAW RGB) →", ha="left", va="top", fontsize=10)

    # prototype image stacks
    for col, p in enumerate(proto_indices, start=1):
        ax = axs[0, col]
        imgs = [sample_overlay_image(int(sid)) for sid in proto_samples[p]]
        stack = _stack_images_vert(imgs, proto_thumb_size)
        ax.imshow(stack, aspect="auto")
        ax.set_title(f"Prototype {p}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # concept refs + matrix
    cmap = plt.get_cmap("Reds")
    ref_size = (
        int(ref_thumb_size[0] * concept_ref_scale),
        int(ref_thumb_size[1] * concept_ref_scale),
    )
    for r, cid in enumerate(concept_ids, start=1):
        ax_refs = axs[r, 0]
        imgs = ref_imgs.get(int(cid), [])
        if imgs:
            strip = _stack_images_horiz(imgs[:n_refimgs], ref_size)
            ax_refs.imshow(strip, aspect="auto")
        else:
            ax_refs.text(0.5, 0.5, "no refs", ha="center", va="center", fontsize=8)
        ax_refs.set_title(f"concept {cid}", fontsize=9, loc="left")
        ax_refs.set_xticks([]); ax_refs.set_yticks([])

        for c, p in enumerate(proto_indices, start=1):
            ax = axs[r, c]
            val = float(values[r - 1, c - 1])
            if abs(val) < 5e-7:
                val = 0.0
            norm = np.clip(val / vmax, 0, 1)
            ax.set_facecolor(cmap(norm))
            if abs(val) < 1e-3:
                label = f"{val:.2e}"
            else:
                label = f"{val:.02f}"
            ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)

    plt.tight_layout()

    meta = {
        "values": values,
        "concept_ids": np.array(concept_ids),
        "proto_indices": np.array(proto_indices),
        "proto_samples": proto_samples,
        "proto_sample_counts": np.array([len(proto_samples[p]) for p in proto_indices]),
        "proto_random_state": proto_random_state,
        "sort_prototypes_by": sort_prototypes_by,
    }
    return fig, meta


def plot_prototypes_only(
    model_name: str,
    model: torch.nn.Module,
    dataset,
    layer_name: str,
    attributions: np.ndarray,
    num_prototypes: int,
    samples_per_proto: int = 3,
    class_id: int = 1,
    skip_prototypes: Optional[Iterable[int]] = None,
    plot_top_k_prototypes: Optional[int] = None,
    proto_thumb_size: Tuple[int, int] = (140, 140),
    overlay_masks: bool = True,
    mask_alpha: float = 0.25,
    proto_random_state: int = 0,
    sort_prototypes_by: Optional[str] = "size",
    proto_samples_override: Optional[List[List[int]]] = None,
) -> Tuple[plt.Figure, Dict[str, np.ndarray]]:
    """
    Plot only prototype image stacks.

    Prototypes are always computed for all `num_prototypes`. `skip_prototypes`
    only removes columns from the rendered figure, not from the clustering.
    """
    device = next(model.parameters()).device
    model.eval()

    attributions_np = np.asarray(attributions, dtype=np.float32)
    attributions_np = np.nan_to_num(attributions_np, nan=0.0, posinf=0.0, neginf=0.0)

    if proto_samples_override is not None:
        proto_samples = list(proto_samples_override)
    else:
        proto_samples, _ = _select_proto_samples(
            attributions_np,
            num_prototypes=num_prototypes,
            samples_per_proto=samples_per_proto,
            random_state=proto_random_state,
        )

    skip = set(skip_prototypes or [])
    proto_indices = [
        p for p in range(num_prototypes)
        if p not in skip and p < len(proto_samples) and len(proto_samples[p]) > 0
    ]

    if sort_prototypes_by == "size":
        proto_indices = sorted(proto_indices, key=lambda p: len(proto_samples[p]), reverse=True)

    if plot_top_k_prototypes is not None and plot_top_k_prototypes > 0:
        proto_indices = sorted(
            proto_indices, key=lambda p: len(proto_samples[p]), reverse=True
        )[:plot_top_k_prototypes]

    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    fv = VISUALIZATIONS[model_name](
        attribution,
        dataset,
        layer_names,
        preprocess_fn=lambda x: x,
        path="unused",
        max_target="max",
    )

    def sample_overlay_image(sample_id: int) -> Image.Image:
        try:
            img_uint8 = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
        except Exception:
            img_tensor, _ = dataset[sample_id]
            img_uint8 = TF.to_pil_image(img_tensor[:3].detach().cpu())
        if not overlay_masks:
            return img_uint8
        img_tensor, _ = dataset[sample_id]
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.from_numpy(np.asarray(img_tensor))
        if img_tensor.ndim == 3 and img_tensor.shape[0] != 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        x = img_tensor.to(device).float().requires_grad_()
        attr = attribution(x.unsqueeze(0), [{"y": class_id}], composite, record_layer=[layer_name])
        pred = attr.prediction[0].argmax(dim=0).detach().cpu().float()
        mask = pred == class_id
        if mask.shape != (img_uint8.height, img_uint8.width):
            mask_f = mask.float().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(
                mask_f, size=(img_uint8.height, img_uint8.width), mode="nearest"
            ).bool().squeeze(0).squeeze(0)
        img_uint8_t = TF.pil_to_tensor(img_uint8)
        over = draw_segmentation_masks(img_uint8_t, masks=mask, alpha=mask_alpha, colors=["red"])
        return TF.to_pil_image(over)

    n_cols = max(1, len(proto_indices))
    fig_w = 1.8 * n_cols * max(proto_thumb_size[0] / 140.0, 1.0)
    fig_h = 1.8 * samples_per_proto * max(proto_thumb_size[1] / 140.0, 1.0)
    fig, axs = plt.subplots(
        samples_per_proto,
        n_cols,
        figsize=(fig_w, fig_h),
        dpi=220,
        facecolor="white",
    )
    axs = np.atleast_2d(axs)

    for col, p in enumerate(proto_indices):
        imgs = [sample_overlay_image(int(sid)) for sid in proto_samples[p]]
        for row in range(samples_per_proto):
            ax = axs[row, col]
            if row < len(imgs):
                ax.imshow(np.asarray(imgs[row].resize(proto_thumb_size, Image.BICUBIC)))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
            if row == 0:
                ax.set_title(f"Prototype {p}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"top {row + 1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    meta = {
        "proto_indices": np.array(proto_indices),
        "proto_samples": proto_samples,
        "proto_sample_counts": np.array([len(proto_samples[p]) for p in proto_indices]),
        "proto_random_state": proto_random_state,
        "sort_prototypes_by": sort_prototypes_by,
    }
    return fig, meta
