import os
import sys
import json
import h5py
import joblib
import numpy as np
import torch
from typing import Union, Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.mixture import GaussianMixture

# If you need CRP/PCX utilities, ensure your sys.path includes their roots
sys.path.append("/home/heydari/paper-camera-ready/paper/paper/")

from LCRP.utils.render import vis_opaque_img_border, vis_opaque_img_border_v2

from crp.image import imgify


# -----------------------------
# Helpers for safe CPU handling
# -----------------------------
def _to_cpu_array(obj):
    """Recursively move tensors to CPU numpy arrays for safe plotting."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_array(o) for o in obj)
    return obj


def vis_opaque_img_border_safe(data_batch, heatmaps, rf, **kwargs):
    """Ensure vis_opaque_img_border always receives CPU tensors/arrays."""
    data_cpu = _to_cpu_array(data_batch)
    heatmaps_cpu = _to_cpu_array(heatmaps)
    return vis_opaque_img_border(data_cpu, heatmaps_cpu, rf, **kwargs)


def get_ref_images(fv, topk_ind, layer_name, composite, class_id, n_ref=12, ref_imgs_save_path="output/ref_imgs/"):
    ref_imgs_save_path = os.path.join(ref_imgs_save_path, f"{layer_name}_class_{class_id}.h5")
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
                    ref_imgs[int(str_k)] = [Image.fromarray(group[str(idx)][:]) for idx in
                                            sorted(group.keys(), key=int)]

            if missing_keys:
                print(f"Calculating and saving missing reference images for keys: {missing_keys}")
                new_refs = fv.get_max_reference([int(k) for k in missing_keys], layer_name, "relevance", (0, n_ref),
                                                composite=composite, rf=True, plot_fn=vis_opaque_img_border, batch_size=2)
                for key, images_list in new_refs.items():
                    group = f.create_group(str(key))
                    if len(images_list) < n_ref:
                        print(f"  WARNING: Concept {key} only has {len(images_list)} images, need {n_ref}")
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
                if len(images_list) < n_ref:
                    print(f"  WARNING: Concept {key} only has {len(images_list)} images, need {n_ref}")
                for idx, image in enumerate(images_list[:n_ref]):
                    if isinstance(image, Image.Image):
                        arr = np.array(image)
                        group.create_dataset(str(idx), data=arr)
                    else:
                        print(f"Warning: Item '{idx}' in key '{key}' is not a PIL image and will not be saved.")

    return ref_imgs


# ----------------------
# Image/crop preparation
# ----------------------
def _to_pil_image_any(x: Union[Image.Image, np.ndarray, torch.Tensor], bgr_input: bool = True) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.copy()
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW
            from torchvision.transforms.functional import to_pil_image
            return to_pil_image(t)
        raise ValueError("Tensor image must be CHW.")
    if isinstance(x, np.ndarray):
        arr = x
        # Convert BGR to RGB if needed
        if bgr_input and arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.0 else np.clip(arr / 255.0, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        if arr.ndim != 3:
            raise ValueError("NumPy image must be HxWxC.")
        return Image.fromarray(arr)
    raise TypeError("Unsupported image type.")


def get_detection_crop_input(
        orig_img: Union[Image.Image, np.ndarray, torch.Tensor],
        box: Union[torch.Tensor, np.ndarray, list, tuple],
        input_size: Union[int, Tuple[int, int]] = (640, 640),
        context: float = 0.4,
        draw_box: bool = True,
        letterbox_mode: bool = True,  # NEW parameter
) -> Image.Image:
    """Returns a PIL crop around the provided box."""
    pil = _to_pil_image_any(orig_img)
    W, H = pil.size  # Original image size

    if isinstance(input_size, int):
        inW = inH = int(input_size)
    else:
        inW, inH = int(input_size[0]), int(input_size[1])

    try:
        b = torch.as_tensor(box, dtype=torch.float32).flatten()[:4].cpu()
    except Exception:
        s = min(W, H) // 2
        return pil.crop((W // 2 - s // 2, H // 2 - s // 2, W // 2 + s // 2, H // 2 + s // 2))

    if torch.isnan(b).any() or torch.isinf(b).any():
        s = min(W, H) // 2
        return pil.crop((W // 2 - s // 2, H // 2 - s // 2, W // 2 + s // 2, H // 2 + s // 2))

    if letterbox_mode:
        # Letterbox-aware rescaling (accounts for padding)
        ratio = min(inH / H, inW / W)
        pad_x = (inW - W * ratio) / 2
        pad_y = (inH - H * ratio) / 2

        x1 = (b[0].item() - pad_x) / ratio
        y1 = (b[1].item() - pad_y) / ratio
        x2 = (b[2].item() - pad_x) / ratio
        y2 = (b[3].item() - pad_y) / ratio
    else:
        # Simple resize scaling (original behavior)
        sx, sy = W / float(inW), H / float(inH)
        x1, y1, x2, y2 = (b * torch.tensor([sx, sy, sx, sy])).tolist()
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))

    if x2 <= x1 or y2 <= y1:
        s = min(W, H) // 2
        return pil.crop((W // 2 - s // 2, H // 2 - s // 2, W // 2 + s // 2, H // 2 + s // 2))

    bw, bh = x2 - x1, y2 - y1
    mx, my = context * bw, context * bh

    cx1 = max(0, int(round(x1 - mx)))
    cy1 = max(0, int(round(y1 - my)))
    cx2 = min(W, int(round(x2 + mx)))
    cy2 = min(H, int(round(y2 + my)))

    if cx2 <= cx1:
        cx1, cx2 = max(0, (cx1 + cx2) // 2 - 50), min(W, (cx1 + cx2) // 2 + 50)
    if cy2 <= cy1:
        cy1, cy2 = max(0, (cy1 + cy2) // 2 - 50), min(H, (cy1 + cy2) // 2 + 50)

    if cx2 <= cx1 or cy2 <= cy1:
        s = min(W, H) // 2
        cx1, cy1, cx2, cy2 = W // 2 - s // 2, H // 2 - s // 2, W // 2 + s // 2, H // 2 + s // 2

    crop = pil.crop((cx1, cy1, cx2, cy2))

    if draw_box and crop.size[0] > 0 and crop.size[1] > 0:
        d = ImageDraw.Draw(crop)
        box_rel = [max(0, x1 - cx1), max(0, y1 - cy1), min(crop.size[0], x2 - cx1), min(crop.size[1], y2 - cy1)]
        if box_rel[2] > box_rel[0] and box_rel[3] > box_rel[1]:
            d.rectangle(box_rel, outline="yellow", width=3)

    return crop


def _normalize_hw(input_size):
    """Return (H, W) ints for torchvision.Resize."""
    if isinstance(input_size, int):
        return (int(input_size), int(input_size))
    w, h = map(int, input_size)
    return (h, w)


def _normalize_wh(input_size):
    """Return (W, H) ints for coordinate math."""
    if isinstance(input_size, int):
        return (int(input_size), int(input_size))
    w, h = map(int, input_size)
    return (w, h)


def get_detection_crop(
        model,
        orig_dataset,
        idx,
        device,
        class_id,
        input_size=640,  # int or (W,H)
        score_thresh=0.25,
        class_names=None,
        input_prediction_num=0,
):
    # --- normalize sizes
    size_hw = _normalize_hw(input_size)  # Resize expects (H, W)
    inW, inH = _normalize_wh(input_size)  # for coord mapping (W, H)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size_hw, interpolation=InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.to(torch.float32)),
    ])

    # 1) load & prep
    img_data, _ = orig_dataset[idx]
    img_orig = T.ToPILImage()(img_data) if isinstance(img_data, torch.Tensor) else img_data.copy()
    orig_w, orig_h = img_orig.size

    img_t = transform(img_orig).unsqueeze(0).to(device)

    # 2) predict
    scores_all, bbox_all = model.predict_with_boxes(img_t)
    scores = scores_all[0]  # [M, C]
    bbox = bbox_all[0]  # [M, 4]

    # 3) filter by class + threshold
    confs, labels = scores.max(dim=1)
    keep = (confs > score_thresh) & (labels == class_id)
    boxes, confs, labels = bbox[keep], confs[keep], labels[keep]

    # --- Fallback: return full image when no usable detection ---
    if boxes.numel() == 0 or input_prediction_num >= boxes.shape[0]:
        return img_orig.copy()

    # 4) select the requested box
    box = boxes[input_prediction_num]
    lb = labels[input_prediction_num].item()

    # 5) map model-input → original coords
    sx, sy = orig_w / float(inW), orig_h / float(inH)
    x1o, y1o, x2o, y2o = (box[[0, 1, 2, 3]] * torch.tensor([sx, sy, sx, sy], device=box.device)).tolist()

    # 6) margin (zoom)
    w_box, h_box = x2o - x1o, y2o - y1o
    zf = 2.0 if lb == 0 else 0.4
    mx, my = w_box * zf, h_box * zf

    # 7) crop & draw
    cx1 = max(0, x1o - mx);
    cy1 = max(0, y1o - my)
    cx2 = min(orig_w, x2o + mx);
    cy2 = min(orig_h, y2o + my)
    cropped = img_orig.crop((cx1, cy1, cx2, cy2))
    ImageDraw.Draw(cropped).rectangle([x1o - cx1, y1o - cy1, x2o - cx1, y2o - cy1], outline="yellow", width=3)
    return cropped


# ------------------------------------------
# Prototype grid with concept references PNG
# ------------------------------------------
def prot_with_concepts(attributions, gmm, model, orig_dataset, device, class_id, layer_name, dataset, composite, fv):
    NUM_SAMPLES_PER_PROTO = 6
    K_CONCEPTS_PER_PROTO = 3
    N_REF_PER_CONCEPT = 2

    A_np = attributions.detach().cpu().numpy()
    means = gmm.means_.astype(np.float32)
    K, C = means.shape

    # nearest samples to each prototype mean
    dists_prots = np.linalg.norm(A_np[:, None, :] - means[None, :, :], axis=2)  # [N, K]
    ranked = np.argsort(dists_prots, axis=0)  # [N, K]
    num_take = min(NUM_SAMPLES_PER_PROTO, ranked.shape[0])

    all_crops = [[] for _ in range(K)]
    for pj in range(K):
        taken = 0
        k = 0
        max_try = min(ranked.shape[0], NUM_SAMPLES_PER_PROTO * 50)
        while taken < NUM_SAMPLES_PER_PROTO and k < max_try:
            idx = int(ranked[k, pj]);
            k += 1
            try:
                crop = get_detection_crop(
                    model=model, orig_dataset=orig_dataset, idx=idx, device=device,
                    class_id=class_id, input_size=640, score_thresh=0.25,
                    class_names=dataset.class_names, input_prediction_num=0
                )
                t = torchvision.transforms.ToTensor()(crop)
                all_crops[pj].append(t);
                taken += 1
            except Exception:
                fallback_tensor, _ = dataset[idx]
                fallback_uint8 = dataset.reverse_normalization(fallback_tensor).clamp(0, 255).byte()
                fallback_pil = F.to_pil_image(fallback_uint8)
                all_crops[pj].append(torchvision.transforms.ToTensor()(fallback_pil));
                taken += 1

        # pad if still short
        while len(all_crops[pj]) < NUM_SAMPLES_PER_PROTO:
            idx0 = int(ranked[0, pj])
            fallback_tensor, _ = dataset[idx0]
            fallback_uint8 = dataset.reverse_normalization(fallback_tensor).clamp(0, 255).byte()
            fallback_pil = F.to_pil_image(fallback_uint8)
            all_crops[pj].append(torchvision.transforms.ToTensor()(fallback_pil))

    M = torch.from_numpy(means).float()
    k_pick = int(min(K_CONCEPTS_PER_PROTO, C))

    per_proto_lists = []
    for pj in range(K):
        row = M[pj]
        pos_mask = row > 0
        if torch.any(pos_mask):
            num_pos = int(pos_mask.sum().item())
            take = min(k_pick, num_pos)
            masked = row.masked_fill(~pos_mask, float('-inf'))
            _, idx = torch.topk(masked, k=take, largest=True)
        else:
            _, idx = torch.topk(row, k=k_pick, largest=False)
        per_proto_lists.append(idx.tolist())

    flat = [int(i) for lst in per_proto_lists for i in lst]
    seen = set();
    top_concepts = []
    for i in flat:
        if i not in seen:
            seen.add(i);
            top_concepts.append(i)

    top_concepts = [int(i) for i in top_concepts]
    N_CONCEPTS = len(top_concepts)

    ref_imgs_concepts = get_ref_images(
        fv, top_concepts, layer_name,
        composite=composite, class_id=class_id,
        n_ref=N_REF_PER_CONCEPT,
        ref_imgs_save_path="../output_synthetic/ref_imgs_6/"
    )

    labels = gmm.predict(A_np)
    counts = np.bincount(labels, minlength=K).astype(float)
    coverage_pct = (counts / max(1, A_np.shape[0])) * 100.0

    mu = A_np.mean(axis=0).astype(np.float32)
    mu_n = mu / (np.linalg.norm(mu) + 1e-12)
    Pn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    sim_mean = (Pn @ mu_n)

    concept_matrix = torch.from_numpy(means[:, top_concepts]).T  # [N_CONCEPTS, K]

    THUMB = 120
    resize_square = T.Compose([T.Resize(THUMB), T.CenterCrop((THUMB, THUMB))])
    all_resized = [[resize_square(t.clamp(0, 1)) for t in col] for col in all_crops]

    top_row_ratio = max(8, NUM_SAMPLES_PER_PROTO + 2)
    fig_pc, axs = plt.subplots(
        nrows=N_CONCEPTS + 1, ncols=K + 1,
        figsize=(K + 6, N_CONCEPTS + 6), dpi=170,
        gridspec_kw={'width_ratios': [6] + [1] * K, 'height_ratios': [top_row_ratio] + [1] * N_CONCEPTS}
    )

    # top row: prototype strips (vertical)
    for pj in range(K):
        grid = torchvision.utils.make_grid(all_resized[pj], nrow=1, padding=1)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)
        axs[0, pj + 1].imshow(grid_np, aspect='auto')
        axs[0, pj + 1].set_title(f"Prototype {pj}\nCovers {coverage_pct[pj]:.0f}%\nSim. {sim_mean[pj]:.2f}", fontsize=9)
        axs[0, pj + 1].axis("off")
    axs[0, 0].axis("off")

    for i, cidx in enumerate(top_concepts):
        imgs = ref_imgs_concepts.get(int(cidx), [])
        tiles = []
        for im in imgs[:N_REF_PER_CONCEPT]:
            if isinstance(im, Image.Image):
                t = F.to_tensor(im).clamp(0, 1)
            else:
                arr = np.asarray(im)
                if arr.ndim == 3:
                    t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).clamp(0, 1)
                else:
                    continue
            tiles.append(resize_square(t))
        if len(tiles) == 0:
            tiles = [torch.zeros(3, 150, 150)]
        nrow = max(1, min(N_REF_PER_CONCEPT, len(tiles)))
        grid = make_grid(tiles, nrow=nrow, padding=0)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)

        axs[i + 1, 0].imshow(grid_np)
        axs[i + 1, 0].set_ylabel(f"concept {int(cidx)}", rotation=90, labelpad=8)
        axs[i + 1, 0].set_yticks([]);
        axs[i + 1, 0].set_xticks([])

    vmax = float(concept_matrix.abs().max().item()) if hasattr(concept_matrix, "abs") else float(
        np.abs(concept_matrix).max())
    for i in range(N_CONCEPTS):
        for j in range(K):
            val = concept_matrix[i, j].item()
            axs[i + 1, j + 1].imshow([[abs(val)]], vmin=0, vmax=vmax,
                                     cmap=("Reds" if val >= 0 else "Blues"))
            color = "white" if abs(val) > 0.5 * vmax else "black"
            axs[i + 1, j + 1].text(0, 0, f"{val * 100:.1f}", ha="center", va="center", color=color, fontsize=10)
            axs[i + 1, j + 1].axis("off")

    plt.tight_layout()

    plot_dir = "../output_synthetic/pcx/pcx_plots"
    out_png = os.path.join(plot_dir, f"{layer_name}_class{class_id}_K{K}_proto_concepts.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig_pc.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig_pc)
    print(f"➡️ prototype+concept figure: {out_png}")


def get_detection_crop_exact(model, orig_dataset, ds_idx, box_idx, device,
                             input_size=640, context=0.4, draw_box=True):
    size_hw = _normalize_hw(input_size)
    inW, inH = _normalize_wh(input_size)

    img_data, _ = orig_dataset[ds_idx]
    img_orig = T.ToPILImage()(img_data) if torch.is_tensor(img_data) else img_data.copy()

    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size_hw, interpolation=InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.to(torch.float32)),
    ])
    x = tfm(img_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        scores_all, boxes_all = model.predict_with_boxes(x)
    boxes = boxes_all[0] if boxes_all.ndim == 3 else boxes_all

    if not (0 <= int(box_idx) < boxes.shape[0]):
        raise IndexError(f"box_idx {box_idx} out of range for ds_idx {ds_idx} (have {boxes.shape[0]})")

    box = boxes[int(box_idx)]
    return get_detection_crop_input(img_orig, box, input_size=(inW, inH), context=context, draw_box=draw_box)


# ------------------------------------------
# Interactive 3D HTML (UMAP / PCA selectable)
# ------------------------------------------
def export_gmm_view_html(
        *,
        attributions, gmm, dataset, orig_dataset, model, class_id,
        get_detection_crop_fn, get_detection_crop_input_fn, meta,
        test_channel_rels=None, test_orig_img=None, test_predicted_box=None, test_context=None,
        device="cuda", input_size=640, score_thresh=0.25, input_prediction_num=0,
        export_dir="../output_synthetic/pcx/gmm_html",
        max_points=20000,  # downsample to keep HTML light
        crop_size=256,  # exported crop size
        title=None,
        reducer="umap",  # "pca" or "umap"
        reducer_n_components=2,  # keep 3 for 3D plot
        reducer_kwargs=None  # e.g. dict(n_neighbors=30, min_dist=0.05, metric="cosine")
):
    import plotly.graph_objects as go
    import plotly.io as pio

    def _save_crop_for_row(row_idx: int, fname: str):
        dsi = int(meta[row_idx]["dataset_idx"])
        # load original image at dataset index
        img_data, _ = orig_dataset[dsi]
        orig_img = T.ToPILImage()(img_data) if torch.is_tensor(img_data) else img_data.copy()

        # prefer the saved box from meta
        if "box" in meta[row_idx]:
            box = np.asarray(meta[row_idx]["box"], dtype=np.float32)

            # FIXED: meta["box"] is in ORIGINAL coordinates
            # Use original_shape as input_size so no scaling occurs
            if "original_shape" in meta[row_idx]:
                orig_shape = meta[row_idx]["original_shape"]  # [H, W]
                box_input_size = (orig_shape[1], orig_shape[0])  # (W, H)
            else:
                box_input_size = orig_img.size  # (W, H)

            crop = get_detection_crop_input_fn(
                orig_img=orig_img,
                box=box,
                input_size=box_input_size,  # FIXED
                context=(2.0 if int(class_id) == 0 else 0.4),
                draw_box=True
            )
        elif "box_letterbox" in meta[row_idx]:
            box = np.asarray(meta[row_idx]["box_letterbox"], dtype=np.float32)
            if "letterbox_shape" in meta[row_idx]:
                lb_shape = meta[row_idx]["letterbox_shape"]
                box_input_size = (lb_shape[1], lb_shape[0])
            else:
                inW, inH = (input_size if isinstance(input_size, (tuple, list)) else (int(input_size), int(input_size)))
                box_input_size = (inW, inH)

            crop = get_detection_crop_input_fn(
                orig_img=orig_img,
                box=box,
                input_size=box_input_size,
                context=(2.0 if int(class_id) == 0 else 0.4),
                draw_box=True
            )
        else:
            bxi = int(meta[row_idx]["box_idx"])
            crop = get_detection_crop_exact(
                model=model, orig_dataset=orig_dataset,
                ds_idx=dsi, box_idx=bxi, device=device,
                input_size=input_size,
                context=(2.0 if int(class_id) == 0 else 0.4),
                draw_box=True
            )

        Image.fromarray(np.asarray(crop)).convert("RGB").resize((crop_size, crop_size)) \
            .save(os.path.join(crops_dir, fname))

    # ---- helper for covariance from GMM (for Mahalanobis fallback) ----
    def _cov_from_gmm(gmm_obj, k):
        ct = getattr(gmm_obj, "covariance_type", "full")
        if ct == "full":
            return gmm_obj.covariances_[k]
        elif ct == "diag":
            return np.diag(gmm_obj.covariances_[k])
        elif ct == "tied":
            return gmm_obj.covariances_
        elif ct == "spherical":
            d = gmm_obj.means_.shape[1]
            return np.eye(d) * gmm_obj.covariances_[k]
        else:
            raise ValueError(f"Unsupported covariance_type: {ct}")

    os.makedirs(export_dir, exist_ok=True)
    crops_dir = os.path.join(export_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # ---- data to numpy ----
    A_full = attributions.detach().cpu().numpy() if hasattr(attributions, "detach") else np.asarray(attributions)
    N, C = A_full.shape
    idx_all = np.arange(N)

    # ---- optional class-component labels (compute once) ----
    comps_full = gmm.predict(A_full)

    # ---- optional downsampling (stratified by component) ----
    if N > max_points:
        take = []
        rng = np.random.RandomState(0)
        for k in range(gmm.means_.shape[0]):
            ids = idx_all[comps_full == k]
            if len(ids) > 0:
                n_k = max(1, int(max_points * len(ids) / N))
                take.append(rng.choice(ids, size=min(n_k, len(ids)), replace=False))
        idx = np.concatenate(take)
    else:
        idx = idx_all

    # Subset arrays for plotting
    A = A_full[idx]
    comps = comps_full[idx]

    # ---- dimensionality reduction (PCA or UMAP) ----
    if reducer_kwargs is None:
        reducer_kwargs = {}

    use_umap = (str(reducer).lower() == "umap")
    reducer_name = "PCA"

    if use_umap:
        try:
            import umap
            default_kwargs = dict(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=0)
            default_kwargs.update(reducer_kwargs)
            um = umap.UMAP(n_components=reducer_n_components, **default_kwargs)
            Xd = um.fit_transform(A)  # (n, d)
            try:
                M_d = um.transform(gmm.means_)  # (K, d)
            except Exception:
                # fallback: nearest-neighbor barycentric mapping of means
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(10, len(A))).fit(A)
                idxs = nn.kneighbors(gmm.means_, return_distance=False)
                M_d = np.vstack([Xd[inds].mean(axis=0) for inds in idxs])

            def embed_new(x1d):
                try:
                    return um.transform(x1d.reshape(1, -1))[0]
                except Exception:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=min(10, len(A))).fit(A)
                    inds = nn.kneighbors(x1d.reshape(1, -1), return_distance=False)[0]
                    return Xd[inds].mean(axis=0)

            reducer_name = "UMAP"
        except Exception as e:
            print(f"[WARN] UMAP not available ({e}); falling back to PCA.")
            use_umap = False

    if not use_umap:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=reducer_n_components, random_state=0).fit(A)
        Xd = pca.transform(A)  # (n, d)
        M_d = pca.transform(gmm.means_)  # (K, d)

        def embed_new(x1d): return pca.transform(x1d.reshape(1, -1))[0]

    if reducer_n_components != 3:
        raise ValueError(f"reducer_n_components must be 3 for 3D plotting, got {reducer_n_components}")

    X3 = Xd[:, :3]
    M3 = M_d[:, :3]

    # ---- nearest member per prototype (from FULL A for stability) ----
    def _nearest_index_to_component_mean_full(k: int) -> int:
        if hasattr(gmm, "precisions_cholesky_"):
            L = gmm.precisions_cholesky_[k].astype(np.float32)
            mu = gmm.means_[k].astype(np.float32)
            diff = (A_full.astype(np.float32) - mu[None, :])
            m2 = np.sum((diff @ L.T) * (diff @ L.T), axis=1)
            return int(np.argmin(m2))
        # fallback: pseudo-inverse covariance
        from numpy.linalg import pinv
        S = _cov_from_gmm(gmm, k).astype(np.float32)
        iS = pinv(S)
        diff = A_full - gmm.means_[k][None, :]
        m2 = np.einsum("ni,ij,nj->n", diff, iS, diff)
        return int(np.argmin(m2))

    K = gmm.means_.shape[0]
    proto_nearest = [_nearest_index_to_component_mean_full(k) for k in range(K)]

    # Use the row-aware saver everywhere:
    for i in idx:
        _save_crop_for_row(int(i), f"ds_{int(i)}.jpg")

    for k, i_near in enumerate(proto_nearest):
        _save_crop_for_row(int(i_near), f"proto_{k}.jpg")

    # ---- test crop (for the clicked detection) ----
    test_name = None
    if test_orig_img is not None and test_predicted_box is not None:
        if isinstance(input_size, int):
            inW = inH = int(input_size)
        else:
            inW, inH = int(input_size[0]), int(input_size[1])
        ctx = 0.4 if test_context is None else float(test_context)
        disp_img = get_detection_crop_input_fn(
            orig_img=test_orig_img,
            box=(test_predicted_box.detach().cpu().numpy() if hasattr(test_predicted_box, "detach") else np.asarray(
                test_predicted_box)),
            input_size=(inW, inH), context=ctx, draw_box=True
        )
        Image.fromarray(np.asarray(disp_img)).convert("RGB").resize((crop_size, crop_size)) \
            .save(os.path.join(crops_dir, "test.jpg"))
        test_name = "test.jpg"

    # ---- build Plotly figure ----
    hover_txt = [f"id={int(i)} | comp={int(c)}" for i, c in zip(idx, comps)]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X3[:, 0], y=X3[:, 1], z=X3[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.85),
        name="dataset",
        customdata=np.stack([idx, comps], axis=1),
        hovertemplate="train_idx=%{customdata[0]}<br>comp=%{customdata[1]}<br>"
                      "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter3d(
        x=M3[:, 0], y=M3[:, 1], z=M3[:, 2],
        mode="markers",
        marker=dict(size=9, symbol="diamond-open", line=dict(width=2, color="red")),
        name="prototypes",
        customdata=np.arange(K),
        hovertemplate="prototype k=%{customdata}<extra></extra>",
    ))
    if test_channel_rels is not None:
        xstar3 = embed_new((test_channel_rels.detach().cpu().numpy()
                            if hasattr(test_channel_rels, "detach") else np.asarray(test_channel_rels)))[:3]
        fig.add_trace(go.Scatter3d(
            x=[xstar3[0]], y=[xstar3[1]], z=[xstar3[2]],
            mode="markers",
            marker=dict(size=12, symbol="x", line=dict(width=2), color="gold"),
            name="test",
            hovertemplate="test sample<extra></extra>",
        ))

    fig.update_layout(
        title=(title or f"GMM 3D — class={class_id}") + f" — K={gmm.n_components} prototypes",
        scene=dict(
            xaxis_title=("UMAP-1" if reducer_name == "UMAP" else "PC1"),
            yaxis_title=("UMAP-2" if reducer_name == "UMAP" else "PC2"),
            zaxis_title=("UMAP-3" if reducer_name == "UMAP" else "PC3"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=720
    )

    html_fig = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    template = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>{title or "GMM 3D"}</title>
<style>
body {{ margin:0; font-family: sans-serif; }}
#wrap {{ display:flex; height:100vh; }}
#plot {{ flex: 1 1 auto; min-width: 0; }}
#inspector {{ width: 360px; padding: 10px; border-left: 1px solid #ddd; box-sizing:border-box; }}
#inspector img {{ width: 100%; height: auto; border: 1px solid #ccc; }}
#meta {{ font-size: 12px; margin-top: 8px; white-space: pre-line; }}
</style></head>
<body>
<div id="wrap">
  <div id="plot">{html_fig}</div>
  <div id="inspector">
    <h3>Inspector</h3>
    <img id="crop" src="{('crops/' + test_name) if test_name else ''}" alt="Click a point">
    <div id="meta">{'test sample' if test_name else 'Click a point or prototype'}</div>
    <p style="font-size:12px;color:#555">Dataset points are saved as <code>crops/ds_{{id}}.jpg</code>,
    prototypes as <code>crops/proto_{{k}}.jpg</code>.</p>
  </div>
</div>
<script>
(function() {{
  const plotDiv = document.querySelector('#plot').querySelector('div.js-plotly-plot');
  const cropImg = document.getElementById('crop');
  const metaDiv = document.getElementById('meta');

  plotDiv.on('plotly_click', function(e) {{
    const tr = e.points[0].curveNumber;
    const pt = e.points[0];
    const name = plotDiv.data[tr].name;

    if (name === 'dataset') {{
      const id = pt.customdata[0];
      const comp = pt.customdata[1];
      cropImg.src = 'crops/ds_' + id + '.jpg';
      metaDiv.textContent = 'dataset id=' + id + ' | comp=' + comp;
    }} else if (name === 'prototypes') {{
      const k = pt.customdata;
      cropImg.src = 'crops/proto_' + k + '.jpg';
      metaDiv.textContent = 'prototype k=' + k + ' (nearest member)';
    }} else if (name === 'test') {{
      cropImg.src = 'crops/test.jpg';
      metaDiv.textContent = 'test sample';
    }}
  }});
}})();
</script>
</body></html>"""

    out_path = os.path.join(export_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(template)

    with open(os.path.join(export_dir, "manifest.json"), "w") as f:
        json.dump({
            "class_id": int(class_id),
            "n_points_plotted": int(len(idx)),
            "prototypes": [int(p) for p in proto_nearest],
            "has_test": bool(test_name),
            "reducer": reducer_name
        }, f, indent=2)

    print(f"[OK] Exported interactive HTML to: {out_path}")
    print("Open it directly, or serve the folder with:  python -m http.server 8000  (then visit http://localhost:8000/"
          + os.path.basename(export_dir) + "/index.html)")


def export_gmm_view_2d(
        *,
        attributions,
        gmm,
        dataset,
        orig_dataset,
        model,
        class_id,
        get_detection_crop_fn,
        get_detection_crop_input_fn,
        meta,
        test_channel_rels=None,
        test_orig_img=None,
        test_predicted_box=None,
        test_context=None,
        device="cuda",
        input_size=640,
        score_thresh=0.25,
        input_prediction_num=0,
        export_dir="../output_synthetic/pcx/gmm_2d",
        max_points=20000,
        crop_size=256,
        title=None,
        reducer="umap",  # "pca" or "umap"
        reducer_kwargs=None,
        figsize=(14, 10),
        dpi=150,
        show_kde=True,
        show_prototype_labels=True,
        show_legend=True,
        cmap="tab10",
):
    """
    Export a 2D scatter plot of GMM attributions with optional test point overlay.

    Parameters
    ----------
    attributions : torch.Tensor or np.ndarray
        Attribution vectors, shape (N, C).
    gmm : GaussianMixture
        Fitted GMM model.
    dataset : Dataset
        Dataset with transforms for model input.
    orig_dataset : Dataset
        Original dataset without transforms.
    model : nn.Module
        Detection model.
    class_id : int
        Class ID being visualized.
    get_detection_crop_fn : callable
        Function to get detection crops by index.
    get_detection_crop_input_fn : callable
        Function to get detection crops from box coordinates.
    meta : list[dict]
        Metadata for each attribution row.
    test_channel_rels : torch.Tensor, optional
        Test sample's channel relevances (ignored - test sample not shown).
    test_orig_img : PIL.Image, optional
        Test sample's original image (ignored - test sample not shown).
    test_predicted_box : np.ndarray, optional
        Test sample's predicted box (ignored - test sample not shown).
    test_context : float, optional
        Context margin for test crop (ignored - test sample not shown).
    device : str
        Device for model inference.
    input_size : int or tuple
        Input size for model.
    score_thresh : float
        Score threshold for detections.
    input_prediction_num : int
        Prediction index for test sample.
    export_dir : str
        Directory to save output.
    max_points : int
        Maximum points to plot (downsampled if exceeded).
    crop_size : int
        Size of saved crop images.
    title : str, optional
        Plot title.
    reducer : str
        Dimensionality reduction method ("pca" or "umap").
    reducer_kwargs : dict, optional
        Additional kwargs for reducer.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.
    show_kde : bool
        Whether to show KDE contours.
    show_prototype_labels : bool
        Whether to label prototype markers.
    show_legend : bool
        Whether to show legend.
    cmap : str
        Colormap for component coloring (ignored - all points are blue).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from scipy import stats
    from PIL import Image
    import torchvision.transforms as T
    import torch

    # ---- helper for covariance from GMM ----
    def _cov_from_gmm(gmm_obj, k):
        ct = getattr(gmm_obj, "covariance_type", "full")
        if ct == "full":
            return gmm_obj.covariances_[k]
        elif ct == "diag":
            return np.diag(gmm_obj.covariances_[k])
        elif ct == "tied":
            return gmm_obj.covariances_
        elif ct == "spherical":
            d = gmm_obj.means_.shape[1]
            return np.eye(d) * gmm_obj.covariances_[k]
        else:
            raise ValueError(f"Unsupported covariance_type: {ct}")

    # ---- data to numpy ----
    A_full = attributions.detach().cpu().numpy() if hasattr(attributions, "detach") else np.asarray(attributions)
    N, C = A_full.shape
    idx_all = np.arange(N)

    # ---- component labels ----
    comps_full = gmm.predict(A_full)
    K = gmm.n_components

    # ---- optional downsampling (stratified by component) ----
    if N > max_points:
        take = []
        rng = np.random.RandomState(0)
        for k in range(K):
            ids = idx_all[comps_full == k]
            if len(ids) > 0:
                n_k = max(1, int(max_points * len(ids) / N))
                take.append(rng.choice(ids, size=min(n_k, len(ids)), replace=False))
        idx = np.concatenate(take)
    else:
        idx = idx_all

    A = A_full[idx]
    comps = comps_full[idx]

    # ---- dimensionality reduction ----
    if reducer_kwargs is None:
        reducer_kwargs = {}

    use_umap = (str(reducer).lower() == "umap")
    reducer_name = "PCA"

    if use_umap:
        try:
            from umap import UMAP
            default_kwargs = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0, n_jobs=1)
            default_kwargs.update(reducer_kwargs)
            um = UMAP(n_components=2, **default_kwargs)
            X2 = um.fit_transform(A)
            try:
                M2 = um.transform(gmm.means_)
            except Exception:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(10, len(A))).fit(A)
                idxs = nn.kneighbors(gmm.means_, return_distance=False)
                M2 = np.vstack([X2[inds].mean(axis=0) for inds in idxs])

            def embed_new(x1d):
                try:
                    return um.transform(x1d.reshape(1, -1))[0]
                except Exception:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=min(10, len(A))).fit(A)
                    inds = nn.kneighbors(x1d.reshape(1, -1), return_distance=False)[0]
                    return X2[inds].mean(axis=0)

            reducer_name = "UMAP"
        except Exception as e:
            print(f"[WARN] UMAP not available ({e}); falling back to PCA.")
            use_umap = False

    if not use_umap:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0).fit(A)
        X2 = pca.transform(A)
        M2 = pca.transform(gmm.means_)

        def embed_new(x1d):
            return pca.transform(x1d.reshape(1, -1))[0]

    # ---- compute coverage per prototype ----
    labels_full = gmm.predict(A_full)
    counts = np.bincount(labels_full, minlength=K).astype(float)
    coverage_pct = (counts / max(1, len(A_full))) * 100.0

    # ---- nearest member per prototype (Mahalanobis) ----
    def _nearest_index_to_component_mean_full(k: int) -> int:
        if hasattr(gmm, "precisions_cholesky_"):
            L = gmm.precisions_cholesky_[k].astype(np.float32)
            mu = gmm.means_[k].astype(np.float32)
            diff = (A_full.astype(np.float32) - mu[None, :])
            m2 = np.sum((diff @ L.T) ** 2, axis=1)
            return int(np.argmin(m2))
        from numpy.linalg import pinv
        S = _cov_from_gmm(gmm, k).astype(np.float32)
        iS = pinv(S)
        diff = A_full - gmm.means_[k][None, :]
        m2 = np.einsum("ni,ij,nj->n", diff, iS, diff)
        return int(np.argmin(m2))

    proto_nearest = [_nearest_index_to_component_mean_full(k) for k in range(K)]

    # ---- create figure ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")

    # ---- KDE background contours ----
    if show_kde and len(X2) > 10:
        try:
            x, y = X2[:, 0], X2[:, 1]
            xmin, xmax = x.min() - 2, x.max() + 2
            ymin, ymax = y.min() - 2, y.max() + 2
            Xg, Yg = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            pos = np.vstack([Xg.ravel(), Yg.ravel()])
            ker = stats.gaussian_kde(np.vstack([x, y]), bw_method=0.3)
            Z = np.reshape(ker(pos).T, Xg.shape).T
            ax.contour(Xg, Yg, Z.T, levels=6, cmap='Greys', alpha=0.3, zorder=0)
        except Exception as e:
            print(f"[WARN] KDE failed: {e}")

    # ---- scatter plot (all blue) ----
    ax.scatter(
        X2[:, 0], X2[:, 1],
        c='#1f77b4',
        s=15,
        alpha=0.7,
        label='Samples',
        zorder=1
    )

    # ---- prototype centers ----
    ax.scatter(
        M2[:, 0], M2[:, 1],
        s=200,
        facecolor='red',
        edgecolor='black',
        linewidth=2,
        marker='D',
        zorder=3,
        label='Prototype centers'
    )

    # ---- prototype labels ----
    if show_prototype_labels:
        for k, (px, py) in enumerate(M2):
            ax.text(
                px, py, str(k),
                fontsize=11,
                fontweight='bold',
                color='yellow',
                ha='center',
                va='center',
                zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')]
            )

    # ---- styling ----
    ax.set_xlabel(f"{reducer_name}-1", fontsize=12)
    ax.set_ylabel(f"{reducer_name}-2", fontsize=12)
    ax.set_title(title or f"GMM Attribution Space — Class {class_id} — K={K} prototypes", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_legend:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    return fig