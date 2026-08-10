import os
import sys
import gc
import time
import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify

logger = logging.getLogger(__name__)

from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from LCRP.utils.render import vis_opaque_img_border

DISPLAY_SIZE = 150

def plot_explanations(model_name, model, dataset, sample_id, class_id, layer, prediction_num, mode, n_concepts,
                      n_refimgs, output_dir):
    # Taken from L-CRP/experiments/plot_crp_explanation.py (visualization only)
    # Datasets may return either (image, label) or a longer tuple
    # (image, label, edge, size, name) depending on implementation.
    out = dataset[sample_id]
    try:
        img, t = out
    except Exception:
        # Fall back to taking the first two elements when more are returned
        img, t = out[0], out[1]
    print(getattr(img, "shape", str(type(img))))

    fig = plot_one_image_explanation(
        model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir
    )

    plt.show()
    print("Done plotting.")


def log_memory(message="", reset_max=False):
    """Log memory usage in a readable format."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if reset_max:
            torch.cuda.reset_max_memory_allocated()
        print(f"MEMORY {message}: {allocated:.2f}GB (max: {max_allocated:.2f}GB)")
    else:
        print(f"MEMORY {message}: CUDA not available")


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _with_model_on_cpu(model, fn):
    """Move the SAME model object to CPU for canonizer-safe work, then restore device."""
    original_device = _model_device(model)
    model.to("cpu").eval()
    try:
        return fn()
    finally:
        model.to(original_device).eval()


def _to_2d(array):
    """Convert activation tensors of arbitrary shape to a 2D spatial map."""
    arr = np.array(array)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[1] > 10 and arr.shape[2] > 10:
            return arr.max(axis=0)
        if arr.shape[2] <= 4 and arr.shape[0] > 10 and arr.shape[1] > 10:
            return arr.mean(axis=2)
        if arr.shape[0] == 1:
            return arr[0]
        return arr.mean(axis=0)
    shape = arr.shape
    idx = list(np.argsort(shape)[-2:])
    perm = [i for i in range(arr.ndim) if i not in idx] + idx
    arr = np.transpose(arr, perm)
    while arr.ndim > 2:
        arr = arr.mean(axis=0)
    return arr


def _colorize_signed_map(raw_map, cmap_name="seismic", clip_percentile=99.5, power=1.15, saturation=0.85):
    """Normalize activations and return a high-contrast RGB visualization."""
    map2d = _to_2d(raw_map).astype("float32")
    if map2d.size == 0:
        return np.zeros((1, 1, 3), dtype=np.float32)

    abs_values = np.abs(map2d)
    scale = np.percentile(abs_values, clip_percentile)
    if scale <= 0:
        scale = abs_values.max()
    if scale <= 0:
        return np.zeros(map2d.shape + (3,), dtype=np.float32)

    normalized = np.clip(map2d / (scale + 1e-12), -1.0, 1.0)
    normalized = np.sign(normalized) * (np.abs(normalized) ** power)

    cmap = plt.cm.get_cmap(cmap_name)
    colored = cmap((normalized + 1.0) / 2.0)[..., :3]
    colored = colored * saturation + (1.0 - saturation)
    return colored


def _resize_for_display(array, size=DISPLAY_SIZE, is_mask=False):
    """Resize numpy-like arrays to a fixed square output for consistent plotting."""
    arr = np.array(array)
    if arr.size == 0:
        return arr

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    tensor = torch.from_numpy(arr).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = tensor.reshape(1, 1, *tensor.shape[-2:])

    if not is_mask:
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.min() < 0.0:
            tensor = tensor - tensor.min()
            if tensor.max() > 0:
                tensor = tensor / tensor.max()

    mode = "nearest" if is_mask else "bilinear"
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode=mode,
        align_corners=False if mode == "bilinear" else None
    ).squeeze(0)

    if resized.shape[0] == 1:
        out = resized[0].cpu().numpy()
        if is_mask:
            return out > 0.5
        return np.clip(out, 0.0, 1.0)

    out = resized.permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def _recover_visual_tensor(dataset, tensor: torch.Tensor) -> torch.Tensor:
    """
    Try reverse_augmentation / reverse_normalization on the dataset; fall back to
    min-max scaling so plotting keeps working even when helpers are missing.
    """
    candidate = tensor.detach().cpu()

    def _ensure_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        return torch.from_numpy(np.array(obj)).detach().cpu()

    recovered = None
    for attr_name in ("reverse_augmentation", "reverse_normalization"):
        fn = getattr(dataset, attr_name, None)
        if callable(fn):
            try:
                recovered = _ensure_tensor(fn(candidate))
                break
            except Exception as exc:
                logger.debug(f"{attr_name} failed for visualization fallback: {exc}")
        else:
            recovered = None

    if recovered is None:
        x = candidate.float()
        if torch.isfinite(x).all():
            mn = x.min()
            mx = x.max()
            if mx > mn:
                x = (x - mn) / (mx - mn)
            else:
                x = x * 0.0
        else:
            x = torch.zeros_like(x)
        x = (x * 255.0).clamp(0, 255)
    else:
        x = recovered.float()
        if x.max() <= 1.5 and x.min() >= 0:
            x = x * 255.0

    if x.ndim == 3:
        if x.shape[0] > 3:
            x = x[:3]
        elif x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
    elif x.ndim == 2:
        x = x.unsqueeze(0).repeat(3, 1, 1)

    return x.clamp(0, 255).to(torch.float32)


def _run_attr_cpu(model, attribution_cls, composite, img_or_batch, *,
                  condition=None, conditions=None, record_layer=None,
                  init_rel=1, exclude_parallel=False, take_prediction=None):
    """
    Run CRP attribution on CPU (avoids BN-folding device mismatches in canonizers),
    then restore the SAME model object back to its original device.
    """
    def _do():
        attribution_cpu = attribution_cls(model)
        if take_prediction is not None:
            setattr(attribution_cpu, "take_prediction", take_prediction)

        x_cpu = img_or_batch.detach().cpu().requires_grad_(True)
        rec = [] if record_layer is None else list(record_layer)  # never pass None

        if conditions is not None:
            return attribution_cpu(
                x_cpu, conditions, composite,
                record_layer=rec, init_rel=init_rel, exclude_parallel=exclude_parallel
            )
        else:
            return attribution_cpu(
                x_cpu, condition, composite,
                record_layer=rec, init_rel=init_rel
            )

    return _with_model_on_cpu(model, _do)


def plot_one_image_explanation_optimized(model_name, model, img, dataset, class_id, layer, prediction_num, mode,
                                         n_concepts, n_refimgs, output_dir):
    # --- device & model state ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    log_memory("STARTING")

    # Build composite after model is on the right device
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    # Ensure input is a tensor on the right device and has a batch dim
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    elif hasattr(img, "to") and isinstance(img, torch.Tensor):
        pass
    elif hasattr(img, "mode"):  # PIL Image
        img = F.to_tensor(img)
    else:
        img = torch.as_tensor(img)

    if img.ndim == 2:
        img = img.unsqueeze(0)

    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    # Create the figure with axes
    concept_space = max(n_refimgs * ratio * 0.8 + 3, 6)
    total_rows = n_concepts + 1
    fig_width = 4.0 + concept_space
    fig_height = 2.1 * total_rows
    fig, axs = plt.subplots(
        total_rows, 3,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [5, 5, concept_space]},
        dpi=200
    )
    # Normalize axs shape when n_concepts == 1 (matplotlib returns 1D array)
    if total_rows == 1:
        axs = np.array([axs])

    log_memory("AFTER FIGURE CREATION")

    heatmap_display = None
    raw_display = None
    segmentation_display = None
    segmentation_mask = None
    raw_display = None
    segmentation_display = None

    # -------------------------
    # Branch: segmentation models
    # -------------------------
    if "deeplab" in model_name or "unet" in model_name or "pidnet" in model_name:
        log_memory("BEFORE SEGMENTATION ATTR")

        # Run attribution on CPU to avoid canonizer device mismatch
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1
        )

        log_memory("AFTER SEGMENTATION ATTR")

        # Heatmap (use matplotlib 'Reds' colormap for stronger red visualization)
        raw_heat = attr.heatmap.detach().cpu().numpy()

        heatmap = _colorize_signed_map(raw_heat)
        heatmap_display = _resize_for_display(heatmap, DISPLAY_SIZE)

        # Mask
        if "DLR" in output_dir:
            mask = attr.prediction[0][0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask > 0.5
        else:
            mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()

        # Recover a display-ready tensor even if dataset lacks helpers
        sample_ = _recover_visual_tensor(dataset, img[0])
        raw_display = _resize_for_display(sample_.permute(1, 2, 0).numpy(), DISPLAY_SIZE)

        # pidnet: resize mask to sample_ spatial size (CPU ops)
        if "pidnet" in model_name:
            mask = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            resized_mask = torch.nn.functional.interpolate(
                mask, size=(sample_.shape[1], sample_.shape[2]), mode='nearest'
            )
            mask = resized_mask.bool().squeeze(0).squeeze(0)  # (H,W)

        # Apply mask to image (draw_segmentation_masks expects CPU tensors)
        if sample_.shape[0] == 3:
            vis = draw_segmentation_masks(sample_.to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])
        else:
            vis = draw_segmentation_masks(sample_[:3, :, :].to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])

        img_ = F.to_pil_image(vis.cpu())
        img_np = np.asarray(img_)
        mask_np = mask.cpu().numpy().astype(float)
        segmentation_display = _resize_for_display(img_np, DISPLAY_SIZE)
        segmentation_mask = _resize_for_display(mask_np, DISPLAY_SIZE, is_mask=True)

        # Clear variables
        del mask, sample_, vis, img_np, mask_np
        gc.collect()

    # -------------------------
    # Branch: detection models
    # -------------------------
    elif "yolo" in model_name or "ssd" in model_name:
        log_memory("BEFORE DETECTION ATTR")

        # Attribution on CPU (canonizers happy)
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1, take_prediction=prediction_num
        )

        log_memory("AFTER DETECTION ATTR")

        # Heatmap (use matplotlib 'Reds' colormap for stronger red visualization)
        raw_heat = attr.heatmap.detach().cpu().numpy()

        heatmap = _colorize_signed_map(raw_heat)
        heatmap_display = _resize_for_display(heatmap, DISPLAY_SIZE)

        # Boxes on the model's (restored) device
        img_dev = img.to(_model_device(model))
        predicted_boxes = model.predict_with_boxes(img_dev)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted_idx = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted_idx]
        predicted_boxes = predicted_boxes[sorted_idx]
        predicted_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]

        boxes = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]

        # Visualize boxes
        base_tensor = _recover_visual_tensor(dataset, img[0])
        raw_display = _resize_for_display(base_tensor.permute(1, 2, 0).numpy(), DISPLAY_SIZE)
        base_img = base_tensor.to(torch.uint8)
        result = draw_bounding_boxes(base_img, boxes.cpu(), colors=colors, width=8)

        img_ = F.to_pil_image(result.cpu())
        img_np = np.asarray(img_)
        segmentation_display = _resize_for_display(img_np, DISPLAY_SIZE)

        # Clear variables
        del predicted_boxes, predicted_classes, sorted_idx, boxes, result, base_img, img_np
        gc.collect()

    else:
        raise NameError(f"Unknown model family for visualization: {model_name}")

    log_memory("AFTER DETECTION/SEGMENTATION")

    if heatmap_display is None:
        heatmap_display = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.float32)
    if raw_display is None:
        base_tensor = _recover_visual_tensor(dataset, img[0])
        raw_display = _resize_for_display(base_tensor.permute(1, 2, 0).numpy(), DISPLAY_SIZE)
    if segmentation_display is None:
        segmentation_display = raw_display.copy()

    # === Channel relevance ===
    if mode == "relevance":
        channel_rels = ChannelConcept().attribute(attr.relevances[layer], abs_norm=True)
    else:
        channel_rels = attr.activations[layer].detach().cpu().flatten(start_dim=2).max(2)[0]
        channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]

    # === Top concepts ===
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    topk_rel = topk.values.detach().cpu().numpy()

    print("Concepts:", topk)

    attribution_start_ts = time.time()

    # === Conditional heatmaps ===
    conditions = [{"y": class_id, layer: c} for c in topk_ind]
    if mode == "relevance":
        log_memory("BEFORE CONDITIONAL HEATMAPS")

        # Second attribution also on CPU; pass record_layer=[layer]
        cond_heatmap, _, _, _ = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, conditions=conditions, record_layer=[layer], init_rel=1,
            exclude_parallel=True, take_prediction=prediction_num
        )

        log_memory("AFTER CONDITIONAL HEATMAPS")
    else:
        cond_heatmap = torch.stack([attr.activations[layer][0][t] for t in topk_ind]).detach().cpu()

    logger.debug(f"Time to compute conditional heatmaps: {time.time() - attribution_start_ts:.2f}s")

    # === Reference images (run on CPU to avoid canonizer device mismatch) ===
    print("Computing reference images...")
    log_memory("BEFORE REF IMAGES")
    ref_images_start_ts = time.time()

    def _get_refs_cpu():
        # Build a CPU attribution and FV object bound to CPU
        attribution_cpu = ATTRIBUTORS[model_name](model)
        fv_cpu = VISUALIZATIONS[model_name](
            attribution_cpu, dataset, get_layer_names(model, [torch.nn.Conv2d]),
            preprocess_fn=lambda x: x, path=output_dir, max_target="max", device="cpu"
        )
        # Call get_max_reference on CPU
        return fv_cpu.get_max_reference(
            topk_ind, layer, mode, (0, n_refimgs), composite=composite, rf=True,
            plot_fn=vis_opaque_img_border, batch_size=2
        )

    ref_imgs = _with_model_on_cpu(model, _get_refs_cpu)

    logger.debug(f"Time to compute reference images: {time.time() - ref_images_start_ts:.2f}s")
    log_memory("AFTER REF IMAGES")

    # === Plotting ===
    plotting_start_ts = time.time()
    print("Plotting...")
    resize = torchvision.transforms.Resize((150, 150))

    for r in range(total_rows):
        row_axs = axs[r]
        log_memory(f"PLOTTING CONCEPT {r}")

        if r == 0:
            for c, ax in enumerate(row_axs):
                if c == 0:
                    ax.set_title("input (raw)")
                    ax.imshow(raw_display)
                    ax.set_aspect("equal")
                else:
                    ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
            gc.collect()
            continue

        concept_idx = r - 1

        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 1:
                    ax.set_title("input")
                    ax.imshow(segmentation_display)
                    if segmentation_mask is not None:
                        ax.contour(segmentation_mask.astype(float), colors="black", linewidths=[2])
                    ax.set_aspect("equal")
                elif r == 2:
                    ax.set_title("heatmap")
                    ax.imshow(heatmap_display)
                    ax.set_aspect("equal")
                else:
                    ax.axis('off')

            if c == 1:
                if r == 1:
                    ax.set_title("cond. heatmap")
                cond_map = cond_heatmap[concept_idx]
                if torch.is_tensor(cond_map):
                    cond_map = cond_map.detach().cpu().numpy()
                cond_vis = _colorize_signed_map(cond_map)
                ax.imshow(_resize_for_display(cond_vis, DISPLAY_SIZE))
                ax.set_aspect("equal")
                ax.set_ylabel(f"concept {topk_ind[concept_idx]}\n relevance: {(topk_rel[concept_idx] * 100):2.1f}%")

            elif c >= 2:
                if r == 1 and c == 2:
                    ax.set_title("concept visualizations")

                if ref_imgs and topk_ind[concept_idx] in ref_imgs:
                    resized_refs = []
                    for i in ref_imgs[topk_ind[concept_idx]]:
                        img_tensor = resize(torch.from_numpy(np.array(i)).permute((2, 0, 1)))
                        resized_refs.append(img_tensor)

                    grid = make_grid(resized_refs, nrow=int(max(1, n_refimgs // 2)), padding=0)
                    grid = np.array(zimage.imgify(grid.detach().cpu()))
                    ax.imshow(grid)
                    ax.yaxis.set_label_position("right")

                    del resized_refs, grid
                    gc.collect()

            ax.set_xticks([])
            ax.set_yticks([])

        gc.collect()

    fig.subplots_adjust(wspace=0.02, hspace=0.5)

    logger.debug(f"Time to plot: {time.time() - plotting_start_ts:.2f}s")
    log_memory("AFTER PLOTTING")

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return fig


def plot_one_image_explanation(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts,
                               n_refimgs, output_dir):
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

        fig = plot_one_image_explanation_optimized(
            model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir
        )

        gc.collect()
        torch.cuda.empty_cache()
        return fig

    except Exception as e:
        print(f"Error during explanation: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        raise


def plot_one_image_explanation_old(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts,
                                   n_refimgs, output_dir):
    """Known-good but VRAM-heavy baseline. Kept for reference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    composite  = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition  = [{"y": class_id}]

    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    fig, axs = plt.subplots(
        n_concepts, 3,
        figsize=((1.6 + 1 / ratio) * n_refimgs / 4, 1.6 * n_concepts),
        gridspec_kw={'width_ratios': [4, 4, n_refimgs * ratio]},
        dpi=200
    )
    if n_concepts == 1:
        axs = np.array([axs])

    if "deeplab" in model_name or "unet" in model_name:
        # Attribution on CPU
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1
        )
        heatmap = zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True)

        if "DLR" in output_dir:
            mask = attr.prediction[0][0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask > 0.5
        else:
            mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()

        sample_ = dataset.reverse_augmentation(img[0].detach().cpu())

        if sample_.shape[0] == 3:
            vis = draw_segmentation_masks(sample_.to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])
        else:
            vis = draw_segmentation_masks(sample_[:3, :, :].to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])

        img_ = F.to_pil_image(vis.cpu())
        axs[0][0].imshow(np.asarray(img_))
        axs[0][0].contour(mask.cpu().numpy(), colors="black", linewidths=[2])

    elif "yolo" in model_name or "ssd" in model_name:
        # Attribution on CPU
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1, take_prediction=prediction_num
        )
        heatmap = np.array(zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True))
        heatmap = zimage.imgify(heatmap, symmetric=True)

        img_dev = img.to(_model_device(model))
        predicted_boxes  = model.predict_with_boxes(img_dev)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted_idx = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted_idx]
        predicted_boxes   = predicted_boxes[sorted_idx]
        predicted_boxes   = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]
        boxes  = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]

        base_img = dataset.reverse_normalization(img[0].detach().cpu()).to(torch.uint8)
        result   = draw_bounding_boxes(base_img, boxes.cpu(), colors=colors, width=8)

        img_ = F.to_pil_image(result.cpu())
        axs[0][0].imshow(np.asarray(img_))
    else:
        raise NameError

    # === Conditional heatmaps ===
    topk_dummy = torch.topk(torch.tensor([1.0, 0.0]).unsqueeze(0), 1)  # placeholder if needed elsewhere
    # Reference images on CPU (same approach as optimized path) if you still need them here:
    # ...

    plt.tight_layout()
    return fig


def fig_to_array(fig):
    """Convert a matplotlib figure to a numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
