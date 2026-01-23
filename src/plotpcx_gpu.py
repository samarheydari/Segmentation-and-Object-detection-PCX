import os
import gc
import sys
import copy
import warnings
import joblib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from PIL import Image
#import plotly
#import plotly.graph_objs
#import plotly.graph_objects as go


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, make_grid

import zennit.image as zimage
from sklearn.mixture import GaussianMixture

from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
from LCRP.utils.render import vis_opaque_img_border as _vis_opaque_img_border_orig  # ← alias original
from crp.image import imgify

# Add the parent directory to the Python path - bad practice, but it's just for the example
#sys.path.append("/Users/heydari/Desktop/test/FHHI-XAI-PIDNET/")

from src.glocal_analysis import run_analysis
from src.datasets.flood_dataset import FloodDataset
from src.datasets.DLR_dataset import DatasetDLR
from src.plot_crp_explanations import plot_explanations, plot_one_image_explanation
from src.minio_client import MinIOClient
from LCRP.models import get_model
from src.device_utils import resolve_device, device_to_str

from contextlib import nullcontext

# Device handling: use torch.device everywhere
DEFAULT_DEVICE = resolve_device()

PANEL_SIZE = (180, 180)  # (height, width) of each subplot content
HEATMAP_CMAP_NAME = "pcx_focus_red"
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    HEATMAP_CMAP_NAME,
    [
        (0.0, "#ffffff"),
        (0.15, "#ffe5e5"),
        (1.0, "#b00000"),
    ],
)
if HEATMAP_CMAP_NAME not in plt.colormaps():
    plt.register_cmap(HEATMAP_CMAP_NAME, _HEATMAP_CMAP)


def _tensor_to_uint8_image(t: torch.Tensor) -> torch.Tensor:
    """Convert arbitrary tensor image to uint8 (C,H,W)."""
    t = t.detach().cpu()
    if t.dtype == torch.uint8:
        return t
    t = t.float()
    if t.max().item() <= 1.0:
        t = t * 255.0
    t = t.clamp(0, 255)
    return t.to(torch.uint8)


def _resize_array_to_panel(arr: np.ndarray) -> np.ndarray:
    """Resize an HxWx3 array to the standard panel size."""
    if arr is None:
        return None
    try:
        arr_np = np.array(arr)
        if arr_np.ndim == 2:
            arr_np = np.stack([arr_np] * 3, axis=-1)
        if arr_np.dtype != np.uint8:
            arr_min, arr_max = arr_np.min(), arr_np.max()
            if arr_max > arr_min:
                arr_np = (255 * (arr_np - arr_min) / (arr_max - arr_min)).clip(0, 255)
            arr_np = arr_np.astype(np.uint8)
        pil = Image.fromarray(arr_np)
        pil_resized = pil.resize((PANEL_SIZE[1], PANEL_SIZE[0]), Image.BILINEAR)
        return np.asarray(pil_resized)
    except Exception:
        return arr

def _coerce_device(device_like=None) -> torch.device:
    """Return a concrete torch.device for the given specification."""
    return resolve_device(device_like) if device_like is not None else DEFAULT_DEVICE


# ---- minimal addition: safe wrapper to align heatmaps/masks to image (H,W) ----
def _align_heatmaps_to_imgHW(data_batch, heatmaps):
    """
    Ensure heatmaps' last two dims match image H,W. If they are W,H, transpose.
    Works with torch.Tensor or numpy arrays, and with list/tuple containers.
    """
    # derive image H,W from data_batch
    if isinstance(data_batch, torch.Tensor):
        # expect (B, C, H, W)
        H, W = int(data_batch.shape[-2]), int(data_batch.shape[-1])
    else:
        # list/tuple of PIL images
        sample = data_batch[0]
        H, W = sample.size[1], sample.size[0]  # PIL: (W,H)

    def _align_one(hm):
        is_numpy = isinstance(hm, np.ndarray)
        hm_t = torch.from_numpy(hm) if is_numpy else hm
        if not torch.is_tensor(hm_t):
            return hm  # leave unchanged if unsupported
        if hm_t.ndim >= 2:
            h2, w2 = int(hm_t.shape[-2]), int(hm_t.shape[-1])
            if (h2, w2) != (H, W) and (h2, w2) == (W, H):
                hm_t = hm_t.transpose(-2, -1)  # swap (W,H) -> (H,W)
        return hm_t.numpy() if is_numpy else hm_t

    if isinstance(heatmaps, (list, tuple)):
        return type(heatmaps)(_align_one(hm) for hm in heatmaps)
    else:
        return _align_one(heatmaps)

def _to_cpu_like(obj):
    """Recursively move tensors to CPU for compatibility with CPU-only render utilities."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_like(o) for o in obj)
    return obj


def vis_opaque_img_border_safe(data_batch, heatmaps, rf, **kwargs):
    """
    Wrapper around original compositor that normalizes heatmap/mask shape
    to image (H,W) to avoid broadcasting errors.
    """
    heatmaps_aligned = _align_heatmaps_to_imgHW(data_batch, heatmaps)
    data_cpu = _to_cpu_like(data_batch)
    heatmaps_cpu = _to_cpu_like(heatmaps_aligned)
    return _vis_opaque_img_border_orig(data_cpu, heatmaps_cpu, rf, **kwargs)
# -------------------------------------------------------------------------------


def _maybe_reset_cuda_max_memory(device_like=None):
    dev = _coerce_device(device_like)
    if dev.type == "cuda":
        try:
            with torch.cuda.device(dev):
                torch.cuda.reset_max_memory_allocated()
        except Exception:
            # older pytorch may not support or permission issues
            pass


def _maybe_empty_cuda_cache(device_like=None):
    dev = _coerce_device(device_like)
    if dev.type == "cuda":
        try:
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
        except Exception:
            pass


def _resolve_layer_concept(fv, layer_name):
    """Robustly locate the concept object associated with ``layer_name``."""
    if not hasattr(fv, "layer_map"):
        return None

    layer_map = fv.layer_map
    if isinstance(layer_map, dict):
        return layer_map.get(layer_name)

    if isinstance(layer_map, (list, tuple)):
        for entry in layer_map:
            if isinstance(entry, tuple) and len(entry) == 2:
                key, value = entry
                if key == layer_name:
                    return value
            elif hasattr(entry, "name") and entry.name == layer_name:
                return entry

    return None


def _call_get_max_reference(fv, concept_ids, layer_name, composite, n_ref, plot_fn, batch_size: int):
    """
    Wrapper that prefers RF-aware references but gracefully falls back
    to vanilla references if channel bookkeeping is incomplete.
    """
    concept_ids = list(concept_ids)

    concept = _resolve_layer_concept(fv, layer_name)
    rf_candidates, fallback_candidates = [], list(concept_ids)
    if concept is not None and hasattr(concept, "c_n_map"):
        rf_candidates = []
        fallback_candidates = []
        for cid in concept_ids:
            try:
                _ = concept.c_n_map[cid]
                rf_candidates.append(cid)
            except (IndexError, KeyError, TypeError):
                fallback_candidates.append(cid)

    def _fetch(ids, rf_flag, allow_retry=True):
        if not ids:
            return {}
        try:
            return fv.get_max_reference(
                ids,
                layer_name,
                "relevance",
                (0, n_ref),
                composite=composite,
                rf=rf_flag,
                plot_fn=plot_fn,
                batch_size=batch_size,
            )
        except IndexError as exc:
            if rf_flag and allow_retry:
                warnings.warn(
                    f"Falling back to rf=False for concepts {ids} at layer '{layer_name}' "
                    f"because receptive-field indices are inconsistent ({exc}).",
                    RuntimeWarning,
                )
                return _fetch(ids, False, allow_retry=False)
            raise

    refs = {}
    refs.update(_fetch(rf_candidates, True))
    refs.update(_fetch(fallback_candidates, False))

    if not refs:
        raise IndexError(f"Unable to derive reference images for layer '{layer_name}' and concepts {concept_ids}.")

    return refs


def get_ref_images(fv, topk_ind, layer_name, composite, n_ref=12, ref_imgs_save_path="examples/output/ref_imgs_pidnet/"):
    """
    Get and cache reference images. CPU/PIL based.
    """
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
                    # read stored PIL images from dataset arrays
                    ref_imgs[int(str_k)] = [Image.fromarray(group[str(idx)][:]) for idx in sorted(group.keys(), key=int)]

            if missing_keys:
                print(f"Calculating and saving missing reference images for keys: {missing_keys}")
                new_refs = _call_get_max_reference(
                    fv,
                    [int(k) for k in missing_keys],
                    layer_name,
                    composite,
                    n_ref,
                    vis_opaque_img_border_safe,
                    batch_size=1,
                )
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
        ref_imgs = _call_get_max_reference(
            fv,
            list(map(int, topk_ind)),
            layer_name,
            composite,
            n_ref,
            vis_opaque_img_border_safe,
            batch_size=1,
        )
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


def plot_pcx_explanations(model_name, model, dataset, sample_id, n_concepts, n_refimgs, num_prototypes,
                          layer_name, ref_imgs_path, output_dir_pcx, output_dir_crp, device=None,
                          precision: str = "fp32"):
    """
    Wrapper that loads the sample, resets memory trackers, calls pidnet version,
    and ensures cleanup.
    """
    try:
        active_device = _coerce_device(device)
        _maybe_reset_cuda_max_memory(active_device)

        image_tensor, t = dataset[sample_id]
        image_tensor = image_tensor.to(active_device)
        if precision == "autocast_fp16" and active_device.type == "cuda":
            image_tensor = image_tensor.half()
        fig = plot_pcx_explanations_pidnet(
            model_name, model, dataset, image_tensor,
            n_concepts=n_concepts, n_refimgs=n_refimgs, num_prototypes=num_prototypes,
            layer_name=layer_name, ref_imgs_path=ref_imgs_path,
            output_dir_pcx=output_dir_pcx, output_dir_crp=output_dir_crp,
            device=active_device,
            precision=precision
        )

        gc.collect()
        _maybe_empty_cuda_cache(active_device)

        return fig

    except Exception as e:
        print(f"Error during explanation: {e}")
        gc.collect()
        _maybe_empty_cuda_cache(active_device if 'active_device' in locals() else None)
        raise


def plot_pcx_explanations_pidnet(model_name, model, dataset, image_tensor,
                                 n_concepts=5, n_refimgs=12, num_prototypes=2,
                                 layer_name="decoder.center.0.0",
                                 ref_imgs_path="examples/output/ref_imgs_pidnet/",
                                 output_dir_pcx="examples/output/pcx/pidnet_flood/",
                                 output_dir_crp="examples/output/crp/pidnet_flood/",
                                 device=None,
                                 precision: str = "fp32"):
    """
    Main function that computes PCX/CRP visualizations.
    This version keeps tensors on GPU when possible and only moves to CPU for
    scikit-learn, plotting and PIL operations.
    """
    active_device = _coerce_device(device)
    non_blocking = active_device.type == "cuda"
    amp_enabled = precision == "autocast_fp16" and active_device.type == "cuda"

    # ensure model in eval and on correct device
    model = model.to(active_device)
    model.eval()
    print(f"[plot_pcx_explanations_pidnet] using device={device_to_str(active_device)}")

    # get layer names (unchanged)
    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])

    # Setting up CRP / attribution / visualizer objects
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    fv = VISUALIZATIONS[model_name](
        attribution,
        dataset,
        layer_names,
        preprocess_fn=lambda x: x,
        path=output_dir_crp,
        max_target="max"
    )
    cc = ChannelConcept()

    def _amp_ctx():
        return torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False) if amp_enabled else nullcontext()

    # Prepare the image tensor on the computation device
    img = image_tensor[None, ...].to(active_device, non_blocking=non_blocking)

    # Decide how many reference images to use given current memory headroom
    effective_n_refimgs = int(n_refimgs)
    if active_device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(active_device).total_memory
        reserved_mem = torch.cuda.memory_reserved(active_device)
        if total_mem > 0 and reserved_mem / total_mem > 0.8:
            effective_n_refimgs = max(4, min(effective_n_refimgs, 6))

    # Load attributions file (stored on disk as numpy)
    folder = f"{output_dir_pcx}/{layer_name}/"
    if not os.path.exists(folder + "attributions.npy"):
        raise FileNotFoundError(f"Attributions file not found: {folder + 'attributions.npy'}")
    # load to CPU then to device as needed; keep a CPU copy for sklearn
    attributions_np = np.load(folder + "attributions.npy")  # numpy on CPU
    # torch tensor on device for any tensor ops
    attributions = torch.from_numpy(attributions_np).to(active_device, non_blocking=non_blocking)

    data = img
    class_id = 1

    # Always refit the GMM so prototype selection reflects the current run
    attributions_for_gmm = attributions_np  # stay on CPU for sklearn
    gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions_for_gmm)

    prototype_gmms = [GaussianMixture(n_components=1, covariance_type='full',) for _ in range(num_prototypes)]
    for p, g_ in enumerate(prototype_gmms):
        g_._set_parameters([
            param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
            for j, param in enumerate(gmm._get_parameters())
        ])

    # Calculate scores for the dataset (CPU)
    scores = gmm.score_samples(attributions_np)
    # raise ValueError(f"device: {device} data device {data.device} model device {model.device}")
    # Run attribution on the input image (this will happen on device)
    with _amp_ctx():
        attr = attribution(
            data.requires_grad_(),
            [{"y": class_id}],
            composite,
            record_layer=[layer_name],
            init_rel=1,
        )

    # Channel (neuron) relevance on the given layer for this image
    rel_tensor = attr.relevances[layer_name].detach()
    channel_rels = cc.attribute(rel_tensor, abs_norm=True).detach()
    attr_heatmap = attr.heatmap.detach().cpu() if hasattr(attr, "heatmap") else None
    pred_tensor = attr.prediction[0].detach() if hasattr(attr, "prediction") else None

    # Move tensors off GPU as soon as possible to free memory
    channel_rels_cpu = channel_rels.cpu()
    pred_tensor_cpu = pred_tensor.cpu() if pred_tensor is not None else None

    del rel_tensor
    del channel_rels
    del attr
    _maybe_empty_cuda_cache(active_device)

    channel_rels = channel_rels_cpu

    # Score the sample against the GMM (sklearn CPU) - convert to CPU numpy
    channel_rels_np = channel_rels.numpy()
    score_sample = gmm.score_samples(channel_rels_np)
    likelihoods = [g_.score_samples(channel_rels_np) for g_ in prototype_gmms]

    # Compute mean of the most likely prototype, then keep it as tensor on device for comparisons
    mean_np = gmm.means_[np.argmax(likelihoods)]
    mean = torch.from_numpy(mean_np).to(active_device, non_blocking=non_blocking)
    if precision == "autocast_fp16" and active_device.type == "cuda":
        mean = mean.half()

    # Find the closest sample to mean using device tensors (move mean to device)
    # attributions is on device; ensure shapes align
    # attributions shape expected (N, D)
    closest_sample_to_mean = ((attributions - mean[None]).pow(2).sum(dim=1)).argmin().item()
    mean_cpu = mean.detach().cpu()
    del mean
    _maybe_empty_cuda_cache(active_device)

    # Save CPU-safe GMM data for inspection
    try:
        joblib.dump((attributions.detach().cpu().numpy(), gmm, channel_rels.detach().cpu().numpy(), mean_cpu.numpy()), "examples/output/pcx/gmm_data.pkl")
    except Exception:
        # fallback: save only plain things
        try:
            joblib.dump((attributions.detach().cpu(), channel_rels.detach().cpu(), mean_cpu), "examples/output/pcx/gmm_data_fallback.pkl")
        except Exception:
            pass
    del attributions
    _maybe_empty_cuda_cache(active_device)

    # Closest prototype sample from dataset (data_p, target_p usually CPU tensors)
    data_p, target_p = dataset[closest_sample_to_mean]
    # keep a CPU copy of target_p for mask computations; move input data to device for model ops
    target_p_cpu = target_p.detach().cpu() if isinstance(target_p, torch.Tensor) else target_p
    data_p_device = data_p[None].to(active_device, non_blocking=non_blocking)

    # Getting top concepts/neurons for the given image in the given layer
    channel_rels = channel_rels.float()
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    effective_n_concepts = min(len(topk_ind), n_concepts)
    if effective_n_concepts == 0:
        effective_n_concepts = 1
    if active_device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(active_device).total_memory
        reserved_mem = torch.cuda.memory_reserved(active_device)
        if total_mem > 0:
            usage = reserved_mem / total_mem
            if usage > 0.85:
                effective_n_concepts = min(effective_n_concepts, 1)
            elif usage > 0.75:
                effective_n_concepts = min(effective_n_concepts, 2)
    topk_ind = topk_ind[:effective_n_concepts]
    channel_rels_plot = channel_rels
    del channel_rels
    _maybe_empty_cuda_cache(active_device)

    # Get reference images (CPU / PIL)
    _maybe_empty_cuda_cache(active_device)
    ref_imgs = get_ref_images(fv, topk_ind, layer_name, composite=composite, n_ref=effective_n_refimgs, ref_imgs_save_path=ref_imgs_path)

    # Calculate conditional heatmaps and prototype heatmaps (calls to attribution may return CPU or device tensors)
    conditions = [{"y": class_id, layer_name: int(c)} for c in topk_ind]

    def _to_cpu_container(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu_container(o) for o in obj)
        return obj

    def _extract_heatmap(res):
        if hasattr(res, "heatmap"):
            return res.heatmap
        if isinstance(res, (list, tuple)):
            return res[0]
        return res

    def _collect_conditional_heatmaps(input_tensor, conds):
        heatmaps = []
        for cond in conds:
            # Clone to avoid autograd graph accumulation and keep memory bounded
            _maybe_empty_cuda_cache(active_device)
            inp = input_tensor.detach().clone().requires_grad_()
            with _amp_ctx():
                result = attribution(inp, [cond], composite)
            heatmaps.append(_to_cpu_container(_extract_heatmap(result)))
            del result
            del inp
            _maybe_empty_cuda_cache(active_device)
            gc.collect()
        return heatmaps

    try:
        cond_heatmap_p = _collect_conditional_heatmaps(data_p_device, conditions)
        data_p_cpu = data_p_device.detach().cpu()
        del data_p_device
        _maybe_empty_cuda_cache(active_device)

        cond_heatmap = _collect_conditional_heatmaps(data, conditions)
        data = None
        img_cpu = img.detach().cpu()
        del img
        _maybe_empty_cuda_cache(active_device)
    except torch.cuda.OutOfMemoryError:
        if active_device.type == "cuda":
            print("[plot_pcx_explanations_pidnet] CUDA OOM during conditional attribution; retrying on CPU.")
            _maybe_empty_cuda_cache(active_device)
            image_tensor_cpu = image_tensor.detach().cpu()
            channel_rels_plot = channel_rels_plot.detach().cpu()
            model_cpu = model.to(torch.device("cpu"))
            model_cpu.eval()
            try:
                fig_cpu = plot_pcx_explanations_pidnet(
                    model_name,
                    model_cpu,
                    dataset,
                    image_tensor=image_tensor_cpu,
                    n_concepts=n_concepts,
                    n_refimgs=n_refimgs,
                    num_prototypes=num_prototypes,
                    layer_name=layer_name,
                    ref_imgs_path=ref_imgs_path,
                    output_dir_pcx=output_dir_pcx,
                    output_dir_crp=output_dir_crp,
                    device=torch.device("cpu"),
                    precision="fp32",
                )
            finally:
                model.to(active_device)
                model.eval()
            _maybe_empty_cuda_cache(active_device)
            return fig_cpu
        raise

    # Segmentation mask for plotting (CPU)
    # Use stored prediction to build mask
    if pred_tensor_cpu is not None:
        mask = (pred_tensor_cpu.argmax(dim=0) == class_id).detach().cpu()
    else:
        # In case prediction missing, make empty mask (safe fallback)
        # shape fallback to small size if needed
        mask = torch.zeros((1, 1), dtype=torch.bool)

    # Reverse augmentation expects CPU tensors; move the original input to CPU for reverse_augmentation
    sample_cpu_for_plot = dataset.reverse_augmentation(img_cpu.float())
    # Resize mask if pidnet-style needs change
    if "pidnet" in model_name:
        # Convert mask to float and add batch + channel dims
        mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
        # Target spatial size based on reversed sample
        try:
            target_h = sample_cpu_for_plot[:3, :, :][0].shape[1]
            target_w = sample_cpu_for_plot[:3, :, :][0].shape[2]
            resized_mask = torch.nn.functional.interpolate(mask_f, size=(target_h, target_w), mode='nearest')
            mask = resized_mask.bool().squeeze().squeeze()
        except Exception:
            # fallback: keep original mask
            mask = mask.squeeze() if mask.dim() > 0 else mask

    # Draw segmentation overlay using torchvision (CPU tensors)
    try:
        base_sample_tensor = _tensor_to_uint8_image(sample_cpu_for_plot[:3, :, :][0])
        img_with_mask = F.to_pil_image(
            draw_segmentation_masks(base_sample_tensor, masks=mask, alpha=0.3, colors=["red"])
        )
    except Exception:
        # fallback: convert sample to PIL without masks
        try:
            fallback_tensor = _tensor_to_uint8_image(sample_cpu_for_plot[:3, :, :][0])
            img_with_mask = F.to_pil_image(fallback_tensor)
        except Exception:
            img_with_mask = Image.new("RGB", (PANEL_SIZE[1], PANEL_SIZE[0]), color=(128, 128, 128))

    # Prototype mask: ensure CPU tensor
    try:
        if isinstance(target_p_cpu, torch.Tensor):
            mask_prototype = (((target_p_cpu - target_p_cpu.min()) / (target_p_cpu.max() - target_p_cpu.min())) > 0.5)[0]
        else:
            # fallback if target not tensor
            mask_prototype = torch.zeros((1, 1), dtype=torch.bool)
    except Exception:
        mask_prototype = torch.zeros((1, 1), dtype=torch.bool)

    # Sample prototype image: reverse augmentation expects CPU input; use data_p_device.cpu()
    sample_prototype_cpu = None
    prototype_base_panel = None
    try:
        sample_prototype_cpu = dataset.reverse_augmentation(data_p_cpu.float())
        base_proto_tensor = _tensor_to_uint8_image(sample_prototype_cpu[:3, :, :][0])
        base_prototype_img = F.to_pil_image(base_proto_tensor)
        prototype_base_panel = base_prototype_img.resize((PANEL_SIZE[1], PANEL_SIZE[0]), Image.BILINEAR)
        img_prototype = base_prototype_img
    except Exception:
        # fallback
        try:
            if sample_prototype_cpu is None:
                sample_prototype_cpu = dataset.reverse_augmentation(data_p_cpu.float())
            base_proto_tensor = _tensor_to_uint8_image(sample_prototype_cpu[:3, :, :][0])
            base_prototype_img = F.to_pil_image(base_proto_tensor)
            prototype_base_panel = base_prototype_img.resize((PANEL_SIZE[1], PANEL_SIZE[0]), Image.BILINEAR)
            img_prototype = base_prototype_img
        except Exception:
            img_prototype = Image.new("RGB", (PANEL_SIZE[1], PANEL_SIZE[0]), color=(128, 128, 128))
            prototype_base_panel = img_prototype
    finally:
        _maybe_empty_cuda_cache(active_device)
        data_p_cpu = None

    # Try to get prototype RGB image via dataset sample for true colors
    prototype_rgb_panel = None
    try:
        fv_dataset_orig = getattr(fv, "dataset", None)
        fv.dataset = dataset
        proto_sample_pil = imgify(fv.get_data_sample(closest_sample_to_mean, preprocessing=False)[0][0])
        if fv_dataset_orig is not None:
            fv.dataset = fv_dataset_orig
        prototype_rgb_panel = _resize_array_to_panel(np.asarray(proto_sample_pil))
    except Exception:
        if prototype_base_panel is not None:
            prototype_rgb_panel = _resize_array_to_panel(np.asarray(prototype_base_panel))
        else:
            prototype_rgb_panel = np.ones((PANEL_SIZE[0], PANEL_SIZE[1], 3), dtype=np.uint8) * 128

    # --- PLOTTING ---
    # set up figure size depending on n_concepts actually used
    n_rows = max(3, effective_n_concepts)
    panel_cols = 6
    concept_col_ratio = np.clip(effective_n_refimgs * 0.6, 3.6, 6.2)
    width_ratios = [1.2, 1.15, concept_col_ratio, 1.15, 1.05, 1.05]
    fig_width = max(14.2, 1.35 * sum(width_ratios))
    fig_height = max(6.6, 2.05 * n_rows)
    fig, axs = plt.subplots(
        n_rows,
        panel_cols,
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.08, 'hspace': 0.36},
        figsize=(fig_width, fig_height),
        dpi=200,
    )
    resize = torchvision.transforms.Resize(PANEL_SIZE, antialias=True)
    panel_h, panel_w = PANEL_SIZE

    # Ensure axs is 2D array for consistent indexing
    if axs.ndim == 1:
        axs = axs[None, :]

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):
            try:
                if c == 2:
                    ax.set_aspect('auto')
                else:
                    ax.set_box_aspect(1)
            except AttributeError:
                if c == 2:
                    ax.set_aspect('auto', adjustable='box')
                else:
                    ax.set_aspect('equal', adjustable='box')
            # Default off for empty cells; we'll enable as necessary
            try:
                if c == 0:
                    if r == 0:
                        ax.set_title("input", fontsize=12, pad=8)
                        overlay_img = img_with_mask.resize((PANEL_SIZE[1], PANEL_SIZE[0]), Image.BILINEAR)
                        ax.imshow(np.asarray(overlay_img))
                        try:
                            mask_np = mask.detach().cpu().numpy()
                            if mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            mask_np = (Image.fromarray((mask_np > 0).astype(np.uint8) * 255)
                                       .resize((PANEL_SIZE[1], PANEL_SIZE[0]), Image.NEAREST))
                            ax.contour(np.array(mask_np, dtype=float), colors="black", linewidths=[1])
                        except Exception:
                            pass
                    elif r == 1:
                        ax.set_title("heatmap", fontsize=12, pad=8)
                        heatmap_img = _heatmap_to_array(attr_heatmap)
                        if heatmap_img is not None:
                            ax.imshow(_resize_array_to_panel(heatmap_img))
                        else:
                            ax.axis("off")
                    elif r == 2:
                        ax.set_title("class likelihood")
                        a = ax.hist(scores, bins=20, color='k')
                        # score_sample may be array; pick scalar if so
                        s_sample_val = float(score_sample) if np.ndim(score_sample) == 0 else float(score_sample[0])
                        ax.vlines(s_sample_val, 0, a[0].max(), linestyle='--', linewidth=3, label="sample")
                        ax.legend()
                        ax.set_ylabel("density")
                        ax.set_xlabel("log-likelihood")
                        ax.set_yticks([])
                        ax.set_xticks([])

                        # outlier thresholds
                        lower_threshold = np.percentile(scores, 1)
                        upper_threshold = np.percentile(scores, 99)

                        outlier_text = "Outlier" if (s_sample_val < lower_threshold or s_sample_val > upper_threshold) else "Ordinary"
                        bbox_props = dict(boxstyle="round,pad=0.3",
                                          edgecolor="red" if outlier_text == "Outlier" else "green",
                                          facecolor="red" if outlier_text == "Outlier" else "green",
                                          alpha=0.3, linewidth=4)
                        ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10,
                                fontweight='bold', color="red" if outlier_text == "Outlier" else "green", bbox=bbox_props)
                    else:
                        ax.axis("off")

                # Non-first column visualizations
                if c == 1:
                    if r == 0:
                        ax.set_title("cond. heatmap", fontsize=12, pad=8)
                    ch = cond_heatmap[r] if r < len(cond_heatmap) else None
                    heatmap_img = _heatmap_to_array(ch)
                    if heatmap_img is not None:
                        ax.imshow(_resize_array_to_panel(heatmap_img))
                    else:
                        ax.axis("off")
                    ax.set_ylabel(
                        f"concept {topk_ind[r]}\n relevance: {(channel_rels_plot[0][topk_ind[r]] * 100):2.1f}%",
                        fontsize=10,
                    )
                    ax.yaxis.labelpad = 10

                elif c == 2:
                    if r == 0:
                        ax.set_title("concept visualizations", fontsize=12, pad=8)
                    # build grid from ref images (PIL)
                    try:
                        concept_refs = ref_imgs[topk_ind[r]][:effective_n_refimgs]
                        grid = make_grid([resize(torch.from_numpy(np.asarray(i).copy()).permute((2, 0, 1))) for i in concept_refs],
                                         nrow=max(1, effective_n_refimgs // 2), padding=0)
                        grid = np.array(zimage.imgify(grid.detach().cpu()))
                        ax.imshow(grid, interpolation="nearest")
                        ax.yaxis.set_label_position("right")
                    except Exception:
                        ax.axis("off")

                elif c == 3:
                    plt.rc('text', usetex=False)
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    bold_font = FontProperties(weight='bold')

                    if r == 0:
                        ax.set_title("Difference to prot.", fontsize=12, pad=8)
                    blank_panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 255
                    ax.imshow(blank_panel, interpolation="nearest")
                    try:
                        delta_R = (channel_rels_plot[0][topk_ind[r]].round(decimals=3) - mean_cpu[topk_ind[r]].round(decimals=3)) * 100
                        delta_R = float(delta_R)
                    except Exception:
                        delta_R = 0.0

                    if delta_R > 2:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n⚠ over-used"
                        edge_color = "#ff0000"
                    elif delta_R < -2:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n⚠ under-used"
                        edge_color = "#ff0000"
                    else:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n✓ similar"
                        edge_color = "#00cc00"

                    rect = patches.Rectangle((0, 0), panel_w, panel_h, linewidth=3, edgecolor=edge_color, facecolor='white')
                    ax.add_patch(rect)
                    lines = textstr.split('\n')
                    symbol_line = lines[1] if len(lines) > 1 else ""
                    text_line = lines[0] if lines else ""

                    ax.text(panel_w / 2, panel_h * 0.35, text_line, fontsize=10,
                            verticalalignment='center', horizontalalignment='center',
                            bbox=dict(facecolor=edge_color, edgecolor='none'))
                    ax.text(panel_w / 2, panel_h * 0.6, symbol_line, fontproperties=bold_font,
                            verticalalignment='center', horizontalalignment='center', color=edge_color)

                    ax.set_xlim([0, panel_w])
                    ax.set_ylim([0, panel_h])
                    ax.axis("off")

                elif c == 4:
                    if r == 0:
                        ax.set_title("Prot localization", fontsize=12, pad=8)
                    proto_heatmap = cond_heatmap_p[r] if r < len(cond_heatmap_p) else None
                    proto_img = _heatmap_to_array(proto_heatmap)
                    if proto_img is not None:
                        ax.imshow(_resize_array_to_panel(proto_img))
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel(
                            f"concept {topk_ind[r]}\n relevance: {(mean_cpu[topk_ind[r]] * 100):2.1f}%",
                            fontsize=10,
                        )
                        ax.yaxis.labelpad = 10
                    else:
                        ax.axis("off")

                elif c == 5:
                    if r == 0:
                        ax.set_title("prototype", fontsize=12, pad=8)
                    try:
                        ax.imshow(prototype_rgb_panel, zorder=1)
                    except Exception:
                        ax.axis("off")

            except IndexError:
                axs[r][c].axis("off")
            except Exception:
                # if any plotting step fails, do not crash whole plot
                try:
                    ax.axis("off")
                except Exception:
                    pass

            # remove ticks
            try:
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception:
                pass
            try:
                ax.margins(0)
            except Exception:
                pass
            try:
                ax.set_facecolor("white")
            except Exception:
                pass

    fig.subplots_adjust(left=0.08, right=0.975, wspace=0.12, hspace=0.38)
    setattr(fig, "_n_refimgs_used", effective_n_refimgs)
    setattr(fig, "_n_concepts_used", effective_n_concepts)
    channel_rels_plot = None
    img_cpu = None
    sample_cpu_for_plot = None
    sample_prototype_cpu = None
    gc.collect()
    _maybe_empty_cuda_cache(active_device)
    return fig


def compute_outlier_scores(model_name, model, dataset, layer_name="decoder.center.0.0", num_prototypes=2,
                           output_dir_pcx="examples/output/pcx/pidnet_flood/", device=None):
    """
    Compute outlier scores using GMM on stored attributions.
    Uses GPU where relevant but keeps sklearn parts on CPU.
    """
    active_device = _coerce_device(device)
    model = model.to(active_device)
    model.eval()

    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_names,
                                     preprocess_fn=lambda x: x,
                                     path=output_dir_pcx,
                                     max_target="max")

    folder = f"{output_dir_pcx}/{layer_name}/"
    if not os.path.exists(folder + "attributions.npy"):
        raise FileNotFoundError(f"Attributions file not found: {folder + 'attributions.npy'}")

    attributions_np = np.load(folder + "attributions.npy")
    attributions = torch.from_numpy(attributions_np).to(active_device)

    # Refit GMM so outlier scores reflect latest statistics
    gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions_np)

    # compute log-likelihood scores on CPU
    scores = gmm.score_samples(attributions_np)

    lower_threshold = np.percentile(scores, 1)
    upper_threshold = np.percentile(scores, 99)

    outliers = [i for i, score in enumerate(scores) if score < lower_threshold or score > upper_threshold]

    return outliers, scores, lower_threshold, upper_threshold
def _heatmap_to_array(hm):
    """Convert a heatmap tensor/list into an RGB numpy array using the project colormap."""
    if hm is None:
        return None
    if isinstance(hm, (list, tuple)):
        for candidate in hm:
            arr = _heatmap_to_array(candidate)
            if arr is not None:
                return arr
        return None
    if torch.is_tensor(hm):
        hm = hm.detach().cpu()
        if hm.ndim == 4 and hm.shape[0] == 1:
            hm = hm[0]
        if hm.ndim == 3 and hm.shape[0] == 1:
            hm = hm[0]
    elif isinstance(hm, np.ndarray):
        hm = np.array(hm)
    elif isinstance(hm, Image.Image):
        return np.asarray(hm)
    try:
        hm_np = np.array(hm, dtype=float)
    except Exception:
        return None

    hm_np = np.nan_to_num(hm_np)
    while hm_np.ndim > 2 and hm_np.shape[0] == 1:
        hm_np = hm_np[0]
    hm_np = np.squeeze(hm_np)
    if hm_np.ndim != 2:
        return None

    hm_np = np.abs(hm_np)
    vmax = np.percentile(hm_np, 99)
    if vmax <= 1e-12:
        return None
    normed = np.clip(hm_np / vmax, 0, 1)
    normed = np.power(normed, 0.7)
    normed[normed < 0.08] = 0
    cmap = plt.get_cmap(HEATMAP_CMAP_NAME)
    rgba = cmap(normed)
    rgba[..., :3] *= 255
    return rgba[..., :3].astype(np.uint8)
