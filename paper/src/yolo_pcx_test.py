import os
import sys
import json
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import joblib
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.font_manager import FontProperties
from sklearn.mixture import GaussianMixture
from crp.helper import get_layer_names
from crp.concepts import ChannelConcept
from crp.image import imgify
from torchvision.utils import draw_bounding_boxes, make_grid
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from src.pcx_helper import (get_detection_crop_input, get_ref_images, get_detection_crop,
                             prot_with_concepts, export_gmm_view_html, get_detection_crop_exact,
                             export_gmm_view_2d)

sys.path.append("..")


def find_matching_box_idx(model, data_tensor, target_box, device, iou_thresh=0.3):
    """Find detection index matching target_box by IoU."""
    with torch.no_grad():
        scores_all, boxes_all = model.predict_with_boxes(data_tensor.to(device))
    boxes = boxes_all[0] if boxes_all.ndim == 3 else boxes_all
    if boxes.shape[0] == 0:
        return 0, 0.0

    if not torch.is_tensor(target_box):
        target_box = torch.tensor(target_box, dtype=torch.float32)
    target_box = target_box.to(boxes.device).flatten()[:4]

    x1 = torch.max(target_box[0], boxes[:, 0])
    y1 = torch.max(target_box[1], boxes[:, 1])
    x2 = torch.min(target_box[2], boxes[:, 2])
    y2 = torch.min(target_box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ious = inter / (area1 + area2 - inter + 1e-6)

    return int(ious.argmax()), float(ious.max())


def plot_pcx_explanations(class_id, model_name, model, img, orig_img, dataset, orig_dataset,
                          n_concepts=5, n_refimgs=12, num_prototypes=None, prediction_num=0,
                          layer_name="decoder.center.0.0",
                          ref_imgs_path="output_brk/ref_imgs/", output_dir_pcx="output_BRK/pcx/yolo_person_car",
                          output_dir_crp="output_BRK/crp/yolo_person_car/", plot_prot_crops=True,
                          letterbox_shape=None, original_shape=None, rescale_boxes_fn=None, dataset_type=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])
    num_prototypes = num_prototypes[class_id]

    # CRP setup
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_names,
                                    preprocess_fn=lambda x: x,
                                    path=output_dir_crp,
                                    max_target="max")
    cc = ChannelConcept()

    # get input
    data = img[None, ...].to(device)

    # load attributions and metadata
    folder = f"{output_dir_pcx}/{layer_name}/"
    attributions = torch.from_numpy(np.load(os.path.join(folder, f"attributions_{class_id}.npy")))
    with open(os.path.join(folder, f"meta_class_{class_id}.json"), "r") as f:
        meta = json.load(f)

    assert attributions.shape[0] == len(meta), \
        f"per-det rows mismatch: A={attributions.shape[0]} vs meta={len(meta)}"

    # GMM fitting/loading
    cache_path = f'{output_dir_pcx}/gmms/gmm_cache_{layer_name}_class_{class_id}_prot_{num_prototypes}.pkl'
    prototype_cache_path = f'{output_dir_pcx}/gmm_prototypes/prototype_gmms_cache_{layer_name}_class_{class_id}_prot_{num_prototypes}.pkl'

    if os.path.exists(cache_path):
        gmm = joblib.load(cache_path)
    else:
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump(gmm, cache_path)

    # Reorder GMM components by coverage (descending)
    A_np = attributions.detach().cpu().numpy().astype(np.float32)
    labels_raw = gmm.predict(A_np)
    K = gmm.n_components
    counts = np.bincount(labels_raw, minlength=K).astype(float)
    order = np.argsort(-counts)

    def _reorder_first_axis(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim >= 1 and arr.shape[0] == K:
            return arr[order, ...]
        return arr

    gmm.means_ = _reorder_first_axis(gmm.means_)
    if hasattr(gmm, 'weights_'):              gmm.weights_ = _reorder_first_axis(gmm.weights_)
    if hasattr(gmm, 'covariances_'):          gmm.covariances_ = _reorder_first_axis(gmm.covariances_)
    if hasattr(gmm, 'precisions_'):           gmm.precisions_ = _reorder_first_axis(gmm.precisions_)
    if hasattr(gmm, 'precisions_cholesky_'):  gmm.precisions_cholesky_ = _reorder_first_axis(gmm.precisions_cholesky_)

    order_cache_dir = os.path.join(output_dir_pcx, "gmms", "orders")
    os.makedirs(order_cache_dir, exist_ok=True)
    np.save(os.path.join(order_cache_dir, f"order_by_coverage_{layer_name}_class_{class_id}_K{K}.npy"), order)

    # Rebuild per-component prototype GMMs in new order
    prototype_gmms = []
    base_params = gmm._get_parameters()
    for p in range(K):
        g1 = GaussianMixture(n_components=1, covariance_type=gmm.covariance_type)
        g1._set_parameters([
            (base_params[j][p:p + 1] if j > 0 else base_params[j][p:p + 1] * 0 + 1.0)
            for j in range(len(base_params))
        ])
        prototype_gmms.append(g1)

    os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True)
    joblib.dump(prototype_gmms, prototype_cache_path)

    # Dataset scores & attribution on input
    scores = gmm.score_samples(attributions)
    data = data.to(device).requires_grad_(True)

    attribution.take_prediction = prediction_num
    attr = attribution(data, condition, composite, record_layer=[layer_name], init_rel=1)

    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True).detach().cpu().float()
    score_sample = gmm.score_samples(channel_rels.detach().cpu())

    # Choose prototype (γ)
    x_star = channel_rels.detach().cpu().numpy()
    A = attributions.detach().cpu().numpy().astype(np.float32)

    post = gmm.predict_proba(x_star)
    chosen_proto = int(post.argmax(axis=1)[0])

    # Class-level percentile
    scores = gmm.score_samples(A)
    score_star = float(score_sample[0])
    p_mix = ((scores < score_star).sum() + 0.5) / (len(scores) + 1)

    # Component-local percentile
    lbl = gmm.predict(A)
    idx_k = np.where(lbl == chosen_proto)[0]
    A_k = A[idx_k]
    g_k = prototype_gmms[chosen_proto]
    scores_k = g_k.score_samples(A_k)
    s_star_k = float(g_k.score_samples(x_star)[0])
    p_local = ((scores_k < s_star_k).sum() + 0.5) / (len(scores_k) + 1)
    coverage = len(idx_k) / max(1, len(A))

    # Mean & Mahalanobis nearest sample
    mean = torch.from_numpy(gmm.means_[chosen_proto])
    mu = gmm.means_[chosen_proto].astype(np.float32)
    L = gmm.precisions_cholesky_[chosen_proto].astype(np.float32)
    diff = A - mu[None, :]
    y = diff @ L.T
    m2 = np.sum(y * y, axis=1)
    closest_row = int(np.argmin(m2))
    ds_idx = int(meta[closest_row]["dataset_idx"])
    box_idx_proto = int(meta[closest_row]["box_idx"])

    # Prepare prototype image
    orig_img_p_raw, _ = orig_dataset[ds_idx]
    if torch.is_tensor(orig_img_p_raw):
        orig_img_p_pil = F.to_pil_image(orig_img_p_raw.byte() if orig_img_p_raw.dtype == torch.uint8
                                        else (orig_img_p_raw * 255).byte())
    else:
        orig_img_p_pil = Image.fromarray(np.array(orig_img_p_raw).astype(np.uint8))

    orig_np_p = np.array(orig_img_p_pil)

    from yolov6.data.data_augment import letterbox
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 64
    img_lb_p = letterbox(orig_np_p, new_shape=640, stride=stride)[0]
    img_tensor_p = torch.from_numpy(img_lb_p.transpose((2, 0, 1)).copy()).float() / 255.0
    data_p = img_tensor_p[None, ...].to(device).requires_grad_(True)

    # Re-verify box_idx_proto by IoU (model output order can vary)
    with torch.no_grad():
        _, boxes_verify = model.predict_with_boxes(data_p)
        boxes_v = boxes_verify[0] if boxes_verify.ndim == 3 else boxes_verify

    target_box_lb = torch.tensor(meta[closest_row]["box_letterbox"], dtype=torch.float32, device=boxes_v.device)
    best_iou_v, best_idx_v = 0, 0
    for i in range(boxes_v.shape[0]):
        b = boxes_v[i]
        x1, y1 = max(target_box_lb[0], b[0]), max(target_box_lb[1], b[1])
        x2, y2 = min(target_box_lb[2], b[2]), min(target_box_lb[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (target_box_lb[2] - target_box_lb[0]) * (target_box_lb[3] - target_box_lb[1])
        area2 = (b[2] - b[0]) * (b[3] - b[1])
        iou_v = float(inter / (area1 + area2 - inter + 1e-6))
        if iou_v > best_iou_v:
            best_iou_v, best_idx_v = iou_v, i
    box_idx_proto = best_idx_v

    # Model inference on prototype
    with torch.no_grad():
        scores_p_all, boxes_p_all = model.predict_with_boxes(data_p)
    boxes_p = boxes_p_all[0] if boxes_p_all.ndim == 3 else boxes_p_all
    n_det = boxes_p.shape[0]

    if "box_letterbox" in meta[closest_row] and n_det > 0:
        target_box_proto = np.array(meta[closest_row]["box_letterbox"], dtype=np.float32)
        box_idx_proto, _ = find_matching_box_idx(model, data_p, target_box_proto, device)
    elif box_idx_proto >= n_det:
        box_idx_proto = 0

    # Draw prototype with stored box
    box_to_draw = torch.tensor(meta[closest_row]["box_letterbox"], dtype=torch.float32).unsqueeze(0)
    img_uint8_p = (data_p[0] * 255).clamp(0, 255).byte().cpu()
    result = draw_bounding_boxes(img_uint8_p, box_to_draw, colors=["#ffcc00"], width=8)
    img_prototype = F.to_pil_image(result)

    # Detection crop for prototype
    ctx = 2.0 if class_id == 0 else 0.4
    predicted_box_p = np.array(meta[closest_row]["box"], dtype=np.float32)
    stored_orig_shape = meta[closest_row]["original_shape"]
    input_size_p = (stored_orig_shape[1], stored_orig_shape[0])

    detection_crop_p = get_detection_crop_input(
        orig_img=orig_img_p_pil, box=predicted_box_p,
        input_size=input_size_p, context=ctx, draw_box=True,
    )

    # Extra diagnostics
    gamma = post[0]
    pi_k = float(gmm.weights_[chosen_proto])

    diff_k = A_k - mu[None, :]
    y_k = diff_k @ L.T
    m2_k = np.sum(y_k * y_k, axis=1)
    y_star = (x_star - mu[None, :]) @ L.T
    m2_star = float(np.sum(y_star * y_star))
    p_m2 = ((m2_k >= m2_star).sum() + 0.5) / (len(m2_k) + 1)

    # Top-K concepts
    rel = channel_rels[0].detach().cpu()
    proto = mean.detach().cpu()
    init_idxs = torch.topk(rel, n_concepts).indices.tolist()
    topk_ind = init_idxs

    # Reference images
    ref_imgs = get_ref_images(fv, topk_ind, layer_name,
                              composite=composite, class_id=class_id,
                              n_ref=n_refimgs,
                              ref_imgs_save_path=f"{ref_imgs_path}/ref_imgs_12/")

    # Conditional heatmaps
    conditions = [{"y": class_id, layer_name: c} for c in topk_ind]
    attribution.take_prediction = prediction_num
    cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions,
                                        composite, exclude_parallel=True)

    # Prototype heatmaps
    attribution.take_prediction = box_idx_proto
    cond_heatmap_p, _, _, _ = attribution(
        data_p.requires_grad_(), conditions, composite, exclude_parallel=True
    )

    attribution.take_prediction = box_idx_proto
    attr_p = attribution(
        data_p.requires_grad_(), condition, composite,
        record_layer=[layer_name], init_rel=1
    )

    attr_p_heatmap = attr_p.heatmap.detach().cpu()
    cond_heatmap_p = cond_heatmap_p.detach().cpu()

    # ===================== PROTOTYPE + CONCEPT FIGURE =====================
    if plot_prot_crops:
        NUM_SAMPLES_PER_PROTO = 6
        K_CONCEPTS_PER_PROTO = 3
        N_REF_PER_CONCEPT = 6

        A_np = attributions.detach().cpu().numpy()
        means = gmm.means_.astype(np.float32)
        K, C = means.shape

        # Nearest samples to each prototype (Mahalanobis)
        diff = A_np[:, None, :] - gmm.means_[None, :, :]
        L_prec = getattr(gmm, "precisions_cholesky_", None)
        if L_prec is not None:
            y = np.einsum("nkc,kdc->nkd", diff, L_prec)
            m2 = np.einsum("nkd,nkd->nk", y, y)
        else:
            m2 = np.einsum("nkc,nkc->nk", diff, diff)
        ranked = np.argsort(m2, axis=0)

        # Build crop strips per prototype
        all_crops = [[] for _ in range(K)]
        for pj in range(K):
            taken, k = 0, 0
            max_try = min(ranked.shape[0], NUM_SAMPLES_PER_PROTO * 50)
            while taken < NUM_SAMPLES_PER_PROTO and k < max_try:
                row_idx = int(ranked[k, pj])
                k += 1
                ds_idx_crop = int(meta[row_idx]["dataset_idx"])
                img_data, _ = orig_dataset[ds_idx_crop]
                orig_img_p_crop = T.ToPILImage()(img_data) if torch.is_tensor(img_data) else img_data.copy()

                try:
                    if "box" in meta[row_idx]:
                        if "original_shape" in meta[row_idx]:
                            orig_shape_crop = meta[row_idx]["original_shape"]
                            input_size_crop = (orig_shape_crop[1], orig_shape_crop[0])
                        elif original_shape is not None:
                            input_size_crop = (original_shape[1], original_shape[0])
                        else:
                            input_size_crop = (640, 640)
                        crop = get_detection_crop_input(
                            orig_img=orig_img_p_crop,
                            box=np.asarray(meta[row_idx]["box"], dtype=np.float32),
                            input_size=input_size_crop,
                            context=(2.0 if class_id == 0 else 0.4),
                            draw_box=True
                        )
                    else:
                        crop = get_detection_crop_exact(
                            model=model, orig_dataset=orig_dataset,
                            ds_idx=ds_idx_crop, box_idx=int(meta[row_idx]["box_idx"]),
                            device=device, input_size=640,
                            context=(2.0 if class_id == 0 else 0.4), draw_box=True
                        )
                    all_crops[pj].append(torchvision.transforms.ToTensor()(crop))
                    taken += 1
                except Exception:
                    fallback_tensor, _ = dataset[ds_idx_crop]
                    fallback_uint8 = dataset.reverse_normalization(fallback_tensor).clamp(0, 255).byte()
                    all_crops[pj].append(torchvision.transforms.ToTensor()(F.to_pil_image(fallback_uint8)))
                    taken += 1

            while len(all_crops[pj]) < NUM_SAMPLES_PER_PROTO:
                row0 = int(ranked[0, pj])
                ds0 = int(meta[row0]["dataset_idx"])
                fb_tensor, _ = dataset[ds0]
                fb_uint8 = dataset.reverse_normalization(fb_tensor).clamp(0, 255).byte()
                all_crops[pj].append(torchvision.transforms.ToTensor()(F.to_pil_image(fb_uint8)))

        # Select top concepts across prototypes
        M = torch.from_numpy(means).float()
        k_pick = int(min(K_CONCEPTS_PER_PROTO, C))

        per_proto_lists = []
        for pj in range(K):
            row = M[pj]
            pos_mask = row > 0
            if torch.any(pos_mask):
                take = min(k_pick, int(pos_mask.sum().item()))
                masked = row.masked_fill(~pos_mask, float('-inf'))
                _, idx = torch.topk(masked, k=take, largest=True)
            else:
                _, idx = torch.topk(row, k=k_pick, largest=False)
            per_proto_lists.append(idx.tolist())

        # Order-preserving unique merge
        seen = set()
        top_concepts = []
        for i in [int(i) for lst in per_proto_lists for i in lst]:
            if i not in seen:
                seen.add(i)
                top_concepts.append(i)
        N_CONCEPTS = len(top_concepts)

        ref_imgs_concepts = get_ref_images(
            fv, top_concepts, layer_name,
            composite=composite, class_id=class_id,
            n_ref=N_REF_PER_CONCEPT,
            ref_imgs_save_path=f"{ref_imgs_path}/ref_imgs_6/"
        )

        # Coverage and similarity
        labels = gmm.predict(A_np)
        counts = np.bincount(labels, minlength=K).astype(float)
        coverage_pct = (counts / max(1, A_np.shape[0])) * 100.0

        mu_class = A_np.mean(axis=0).astype(np.float32)
        mu_n = mu_class / (np.linalg.norm(mu_class) + 1e-12)
        Pn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
        sim_mean = Pn @ mu_n

        concept_matrix = torch.from_numpy(means[:, top_concepts]).T

        THUMB = 120
        resize_square = T.Compose([T.Resize(THUMB), T.CenterCrop((THUMB, THUMB))])
        all_resized = [[resize_square(t.clamp(0, 1)) for t in col] for col in all_crops]

        top_row_ratio = max(8, NUM_SAMPLES_PER_PROTO + 2)
        fig_pc, axs_pc = plt.subplots(
            nrows=N_CONCEPTS + 1, ncols=K + 1,
            figsize=(K + 6, N_CONCEPTS + 6), dpi=170,
            gridspec_kw={'width_ratios': [6] + [1] * K, 'height_ratios': [top_row_ratio] + [1] * N_CONCEPTS}
        )

        for pj in range(K):
            grid = torchvision.utils.make_grid(all_resized[pj], nrow=1, padding=1)
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            axs_pc[0, pj + 1].imshow(grid_np, aspect='auto')
            axs_pc[0, pj + 1].set_title(
                f"Prototype {pj}\nCovers {coverage_pct[pj]:.0f}%\nSim. {sim_mean[pj]:.2f}", fontsize=9)
            axs_pc[0, pj + 1].axis("off")
        axs_pc[0, 0].axis("off")

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
                row0 = int(ranked[0, 0])
                ds0 = int(meta[row0]["dataset_idx"])
                full_tensor, _ = dataset[ds0]
                full_uint8 = dataset.reverse_normalization(full_tensor).clamp(0, 255).byte()
                tiles = [resize_square(F.to_tensor(F.to_pil_image(full_uint8)).clamp(0, 1))]

            grid = make_grid(tiles, nrow=max(1, min(len(tiles), N_REF_PER_CONCEPT)), padding=0)
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            axs_pc[i + 1, 0].imshow(grid_np)
            axs_pc[i + 1, 0].set_ylabel(f"concept {int(cidx)}", rotation=90, labelpad=8)
            axs_pc[i + 1, 0].set_yticks([])
            axs_pc[i + 1, 0].set_xticks([])

        vmax = float(concept_matrix.abs().max().item())
        for i in range(N_CONCEPTS):
            for j in range(K):
                val = concept_matrix[i, j].item()
                axs_pc[i + 1, j + 1].imshow([[abs(val)]], vmin=0, vmax=vmax,
                                             cmap=("Reds" if val >= 0 else "Blues"))
                color = "white" if abs(val) > 0.5 * vmax else "black"
                axs_pc[i + 1, j + 1].text(0, 0, f"{val * 100:.1f}", ha="center", va="center",
                                           color=color, fontsize=10)
                axs_pc[i + 1, j + 1].axis("off")

        plt.tight_layout()
        plt.show()

    # ===================== 2D GMM PLOT =====================
    scores_det, boxes_det = model.predict_with_boxes(data)
    scores_det = scores_det[0]
    boxes_det = boxes_det[0]

    if letterbox_shape is not None and original_shape is not None and rescale_boxes_fn is not None:
        boxes_det = torch.from_numpy(rescale_boxes_fn(
            boxes_det, letterbox_shape=letterbox_shape, original_shape=original_shape
        ))

    if prediction_num >= boxes_det.shape[0]:
        raise IndexError(f"Only {boxes_det.shape[0]} detections, asked for #{prediction_num}")

    predicted_box = boxes_det[int(prediction_num)]
    pred_confidence = float(scores_det[prediction_num, class_id].item())

    # Outlier thresholds
    global_outlier = (p_mix < 0.01)
    locally_typical = (p_local >= 0.01) or (p_m2 >= 0.05) or (gamma[chosen_proto] >= 0.85)
    small_component = (coverage <= 0.10) or (pi_k <= 0.10)
    is_correct = (pred_confidence >= 0.50)
    show_extra_diag = global_outlier and locally_typical and small_component and is_correct

    pred_label = dataset.class_names[class_id]

    # Input image with bounding box
    with torch.no_grad():
        _, boxes_raw = model.predict_with_boxes(data)
    box_letterbox = boxes_raw[0][prediction_num].clone().detach().float()[None]

    result = draw_bounding_boxes(
        (dataset.reverse_normalization(data[0])).type(torch.uint8),
        box_letterbox, colors=["#ffcc00"], width=8
    )
    img_ = F.to_pil_image(result)

    inW, inH = int(data.shape[-1]), int(data.shape[-2])
    detection_crop = get_detection_crop_input(
        orig_img=orig_img,
        box=predicted_box.detach().cpu().numpy() if torch.is_tensor(predicted_box) else predicted_box,
        input_size=(original_shape[1], original_shape[0]) if original_shape is not None else (inW, inH),
        context=ctx, draw_box=True,
    )

    fig_2d = export_gmm_view_2d(
        attributions=attributions, gmm=gmm, dataset=dataset, orig_dataset=orig_dataset,
        model=model, class_id=class_id,
        get_detection_crop_fn=get_detection_crop_exact,
        get_detection_crop_input_fn=get_detection_crop_input,
        meta=meta, test_channel_rels=channel_rels[0],
        test_orig_img=orig_img, test_predicted_box=predicted_box, test_context=ctx,
        device=device,
        export_dir=f"output_{dataset_type}/pcx/gmm_2d/{layer_name}",
        title=f"GMM 2D — {layer_name} | class {class_id}",
        reducer="umap", show_kde=True,
    )
    plt.show()

    # ===================== MAIN EXPLANATION PLOT =====================
    width_ratios = [1, 1, n_refimgs / 4, 1, 1, 1]
    n_rows = max(n_concepts, 5 if show_extra_diag else 4)
    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(4 * n_refimgs / 4, 1.8 * n_rows),
        gridspec_kw={'width_ratios': width_ratios},
        dpi=200
    )
    resize = torchvision.transforms.Resize((150, 150))

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):
            if c in (1, 2, 3, 4) and r >= n_concepts:
                ax.axis("off")
                continue

            if c == 0:
                if r == 0:
                    ax.set_title("input")
                    ax.imshow(img_.resize((150, 150), Image.BILINEAR))
                elif r == 1:
                    ax.set_title("heatmap")
                    hm = imgify(attr.heatmap.detach().cpu(), cmap="bwr", symmetric=True, level=5)
                    ax.imshow(hm.resize((150, 150), Image.BILINEAR))
                elif r == 2:
                    ax.set_title("Detection", fontsize=10)
                    ax.text(0.02, 0.98, f"{pred_label} {pred_confidence * 100:.1f}%",
                            transform=ax.transAxes, fontsize=7, fontweight="bold", color="yellow",
                            va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.3, edgecolor="none"))
                    ax.imshow(detection_crop)
                elif r == 3:
                    a = ax.hist(scores, bins=30, density=True, color='k', alpha=0.85)
                    ax.vlines(score_star, 0, (a[0].max() if len(a[0]) else 1),
                              linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_yticks([])
                    ax.set_xticks([])

                    lower_threshold = np.percentile(scores, 5)
                    outlier_text = "Outlier" if score_sample < lower_threshold else "Ordinary"
                    oc = "red" if outlier_text == "Outlier" else "green"
                    ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10,
                            fontweight='bold', color=oc,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor=oc, facecolor=oc, alpha=0.3, linewidth=4))
                else:
                    ax.axis("off")

            if c == 1:
                if r == 0:
                    ax.set_title("Input localization")
                cond_h = imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True, level=5)
                ax.imshow(cond_h.resize((150, 150), Image.BILINEAR))
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}")

            elif c == 2:
                if r == 0:
                    ax.set_title("concept visualization")
                grid = make_grid(
                    [resize(torch.from_numpy(np.asarray(i).copy()).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs / 2), padding=0)
                ax.imshow(grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                ax.yaxis.set_label_position("right")

            elif c == 3:
                plt.rc('text', usetex=False)
                plt.rcParams['font.family'] = 'DejaVu Sans'
                bold_font = FontProperties(weight='bold')

                if r == 0:
                    ax.set_title("Difference to prot")
                ax.imshow(np.zeros((150, 150, 3)), alpha=0.2, cmap=None)
                delta_R = (channel_rels[0][topk_ind[r]].round(decimals=3) - mean[topk_ind[r]].round(decimals=3)) * 100
                if delta_R > 2.5:
                    textstr = f"ΔR = {delta_R:+2.1f}\n⚠ over-used"
                    edge_color = "#ff0000"
                elif delta_R < -2.5:
                    textstr = f"ΔR = {delta_R:+2.1f}\n⚠ under-used"
                    edge_color = "#ff0000"
                else:
                    textstr = f"ΔR = {delta_R:+2.1f}\n✓ similar"
                    edge_color = "#00cc00"

                ax.add_patch(patches.Rectangle((0, 0), 150, 150, linewidth=3,
                                               edgecolor=edge_color, facecolor='white'))
                text_line, symbol_line = textstr.split('\n')
                ax.text(75, 60, text_line, fontsize=10, va='center', ha='center',
                        bbox=dict(facecolor=edge_color, edgecolor='none'))
                ax.text(75, 90, symbol_line, fontproperties=bold_font, va='center',
                        ha='center', color=edge_color)
                ax.set_xlim([0, 150])
                ax.set_ylim([0, 150])
                ax.axis("off")

            elif c == 5:
                if r == 0:
                    ax.set_title("prototype")
                    ax.imshow(img_prototype.resize((150, 150), Image.BILINEAR))
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(imgify(attr_p_heatmap, cmap="bwr", symmetric=True, level=5).resize((150, 150), Image.BILINEAR))
                elif r == 2:
                    ax.set_title("detection")
                    ax.imshow(detection_crop_p)
                else:
                    ax.axis("off")

            elif c == 4:
                if r == 0:
                    ax.set_title("Prot localization")
                cond_h_p = imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True, level=5)
                ax.imshow(cond_h_p.resize((150, 150), Image.BILINEAR))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(mean[topk_ind[r]] * 100):2.1f}")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig