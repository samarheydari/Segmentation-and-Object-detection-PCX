import os
import torch
import torchvision
# Add the parent directory to the Python path - bad practice, but it's just for the example
import sys
import torchvision.transforms.functional as F
import matplotlib.patches as patches
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, ImageDraw
import torchvision.transforms.functional as TF

import torchvision.transforms as T
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
import json
sys.path.append("..")
#from src.minio_client import MinIOClient
from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
# from sklearn.mixture import GaussianMixture
from crp.image import imgify
from torchvision.utils import draw_bounding_boxes, make_grid
from matplotlib.font_manager import FontProperties
from src.pcx_helper import get_detection_crop_input, get_ref_images, get_detection_crop, prot_with_concepts, export_gmm_view_html, get_detection_crop_exact


# Add this helper function
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
                          ref_imgs_path="../output/ref_imgs/", output_dir_pcx="../output_synthetic/pcx/yolo_person_car",
                          output_dir_crp="../output/crp/yolo_person_car/", plot_prot_crops=True,
                          letterbox_shape=None, original_shape=None, rescale_boxes_fn=None, dataset_type=None):

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)
    model.eval()

    # layers and prototypes
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
    data = img
    data = data[None, ...].to(device)

    # load attributions
    folder = f"{output_dir_pcx}/{layer_name}/"
    attr_path = os.path.join(folder, f"attributions_{class_id}.npy")
    meta_path = os.path.join(folder, f"meta_class_{class_id}.json")

    attributions = torch.from_numpy(np.load(attr_path))
    with open(meta_path, "r") as f:
        meta = json.load(f)  # list of dicts: {"dataset_idx", "box_idx", "cls", "conf", "box"}

    assert attributions.shape[0] == len(meta), \
        f"per-det rows mismatch: A={attributions.shape[0]} vs meta={len(meta)}"

    # GMM fitting/loading
    cache_path = f'{output_dir_pcx}/gmms/gmm_cache_{layer_name}_class_{class_id}_prot_{num_prototypes}.pkl'  # CHANGED
    prototype_cache_path = f'{output_dir_pcx}/gmm_prototypes/prototype_gmms_cache_{layer_name}_class_{class_id}_prot_{num_prototypes}.pkl'  # CHANGED

    # --- GMM: load or fit (unchanged) ---
    if os.path.exists(cache_path):
        gmm = joblib.load(cache_path)
    else:
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump(gmm, cache_path)

    # --- REORDER BY COVERAGE (must happen BEFORE using gmm anywhere else) ---
    A_np = attributions.detach().cpu().numpy().astype(np.float32)  # (N, C)
    labels_raw = gmm.predict(A_np)  # component index per sample (current order)
    K = gmm.n_components
    counts = np.bincount(labels_raw, minlength=K).astype(float)  # coverage counts per component
    order = np.argsort(-counts)  # DESC: most coverage first
    perm_inv = np.empty_like(order);
    perm_inv[order] = np.arange(K)  # map old -> new indices

    def _reorder_first_axis(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        # reorder only if the first dimension matches K
        if arr.ndim >= 1 and arr.shape[0] == K:
            return arr[order, ...]
        return arr

    # apply permutation to gmm internals
    gmm.means_ = _reorder_first_axis(gmm.means_)
    if hasattr(gmm, 'weights_'):              gmm.weights_ = _reorder_first_axis(gmm.weights_)
    if hasattr(gmm, 'covariances_'):          gmm.covariances_ = _reorder_first_axis(gmm.covariances_)
    if hasattr(gmm, 'precisions_'):           gmm.precisions_ = _reorder_first_axis(gmm.precisions_)
    if hasattr(gmm, 'precisions_cholesky_'):  gmm.precisions_cholesky_ = _reorder_first_axis(gmm.precisions_cholesky_)

    # (optional) persist the order so all figures/runs keep the same numbering
    order_cache_dir = os.path.join(output_dir_pcx, "gmms", "orders")
    os.makedirs(order_cache_dir, exist_ok=True)
    np.save(os.path.join(order_cache_dir, f"order_by_coverage_{layer_name}_class_{class_id}_K{K}.npy"), order)

    # --- REBUILD prototype_gmms IN THE NEW ORDER ---
    # (Avoid relying on an old cache that used the old numbering)
    prototype_gmms = []
    base_params = gmm._get_parameters()  # private API but OK since you already use it elsewhere
    for p in range(K):
        g1 = GaussianMixture(n_components=1, covariance_type=gmm.covariance_type)
        # weights -> 1 for the single component; means/covs taken from component p
        g1._set_parameters([
            (base_params[j][p:p + 1] if j > 0 else base_params[j][p:p + 1] * 0 + 1.0)
            for j in range(len(base_params))
        ])
        prototype_gmms.append(g1)

    # (optional) cache these if you want
    os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True)
    joblib.dump(prototype_gmms, prototype_cache_path)

    # dataset scores
    scores = gmm.score_samples(attributions)
    data = data.to(device).requires_grad_(True)

    with torch.no_grad():
        scores_input_all, boxes_input_all = model.predict_with_boxes(data.detach())
    scores_input = scores_input_all[0] if scores_input_all.ndim == 3 else scores_input_all
    boxes_input = boxes_input_all[0] if boxes_input_all.ndim == 3 else boxes_input_all
    num_input_detections = int(boxes_input.shape[0])
    if num_input_detections == 0:
        raise ValueError("No detections found for the input image.")
    if prediction_num < 0 or prediction_num >= num_input_detections:
        raise IndexError(
            f"Requested prediction_num={prediction_num}, but the input image has "
            f"{num_input_detections} detections."
        )

    prediction_idx = int(prediction_num)

    # attribution on input
    attribution.take_prediction = prediction_idx
    attr = attribution(
        data,
        condition,
        composite,
        record_layer=[layer_name],
        init_rel=1)

    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)
    channel_rels = channel_rels.detach().cpu().float()

    # --- sample fit (mixture) ---
    score_sample = gmm.score_samples(channel_rels.detach().cpu())  # shape (1,)

    # === PREP ARRAYS & CHOOSE PROTOTYPE (γ) ===
    x_star = channel_rels.detach().cpu().numpy()  # (1, C)
    A = attributions.detach().cpu().numpy().astype(np.float32)  # (N, C)

    post = gmm.predict_proba(x_star)  # (1, K) responsibilities
    chosen_proto = int(post.argmax(axis=1)[0])

    # === CLASS-LEVEL (mixture) PERCENTILE ===
    scores = gmm.score_samples(A)  # mixture log-likelihoods, (N,)
    score_star = float(score_sample[0])
    p_mix = ((scores < score_star).sum() + 0.5) / (len(scores) + 1)

    # === COMPONENT-LOCAL PERCENTILE ===
    lbl = gmm.predict(A)  # argmax γ for each train sample
    idx_k = np.where(lbl == chosen_proto)[0]
    A_k = A[idx_k]
    g_k = prototype_gmms[chosen_proto]  # single-Gaussian for component k
    scores_k = g_k.score_samples(A_k)  # (n_k,)
    s_star_k = float(g_k.score_samples(x_star)[0])
    p_local = ((scores_k < s_star_k).sum() + 0.5) / (len(scores_k) + 1)
    coverage = len(idx_k) / max(1, len(A))  # component frequency in class

    # === MEAN & MAHALANOBIS NEAREST SAMPLE ===
    mean = torch.from_numpy(gmm.means_[chosen_proto])

    mu = gmm.means_[chosen_proto].astype(np.float32)  # (C,)
    L = gmm.precisions_cholesky_[chosen_proto].astype(np.float32)  # (C, C), upper chol of precision
    diff = A - mu[None, :]
    y = diff @ L.T
    m2 = np.sum(y * y, axis=1)  # Mahalanobis^2
    closest_row = int(np.argmin(m2))  # row in attributions
    ds_idx = int(meta[closest_row]["dataset_idx"])  # NEW

    box_idx_proto = int(meta[closest_row]["box_idx"])  # NEW

    # DEBUG: Print and verify prototype selection
    print(f"\n{'=' * 60}")
    print(f"[DEBUG] PROTOTYPE SELECTION:")
    print(f"  closest_row in meta: {closest_row}")
    print(f"  ds_idx (dataset index): {ds_idx}")
    print(f"  box_idx_proto: {box_idx_proto}")
    print(f"  meta entry: {meta[closest_row]}")
    print(f"{'=' * 60}\n")

    # DEBUG: Visually verify both datasets return same image
    import matplotlib.pyplot as plt

    data_p_debug, _ = dataset[ds_idx]
    orig_p_debug, _ = orig_dataset[ds_idx]

    # ========== FIXED: Use orig_dataset as source of truth ==========
    orig_img_p_raw, _ = orig_dataset[ds_idx]

    # Convert to PIL
    if torch.is_tensor(orig_img_p_raw):
        orig_img_p_pil = F.to_pil_image(orig_img_p_raw.byte() if orig_img_p_raw.dtype == torch.uint8
                                        else (orig_img_p_raw * 255).byte())
    else:
        orig_img_p_pil = Image.fromarray(np.array(orig_img_p_raw).astype(np.uint8))

    orig_np_p = np.array(orig_img_p_pil)
    original_shape_p = orig_np_p.shape[:2]

    # Apply letterbox for model
    from YOLOV6.yolov6.data.data_augment import letterbox
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 64
    img_lb_p = letterbox(orig_np_p, new_shape=640, stride=stride)[0]

    # Convert to tensor
    img_tensor_p = torch.from_numpy(img_lb_p.transpose((2, 0, 1)).copy()).float() / 255.0
    data_p = img_tensor_p[None, ...].to(device).requires_grad_(True)

    # ============ ADD THIS BLOCK RIGHT BEFORE LINE 329 ============
    # Re-verify box_idx_proto immediately before attribution
    # (model output order can vary between calls)
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
    print(f"[ATTR FIX] Using box_idx_proto={box_idx_proto} (IoU={best_iou_v:.3f}) for attribution")
    # ============ END OF ADDED BLOCK ============

    # Model inference
    with torch.no_grad():
        scores_p_all, boxes_p_all = model.predict_with_boxes(data_p)
    scores_p = scores_p_all[0] if scores_p_all.ndim == 3 else scores_p_all
    boxes_p = boxes_p_all[0] if boxes_p_all.ndim == 3 else boxes_p_all
    n_det = boxes_p.shape[0]

    # IoU matching
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

    # Detection crop from SAME source
    ctx = 2.0 if class_id == 0 else 0.4
    predicted_box_p = np.array(meta[closest_row]["box"], dtype=np.float32)
    # Use stored original_shape from metadata to ensure consistency
    stored_orig_shape = meta[closest_row]["original_shape"]  # (H, W)
    input_size_p = (stored_orig_shape[1], stored_orig_shape[0])  # (W, H)

    print(f"[DEBUG] original_shape_p (computed): {original_shape_p}")
    print(f"[DEBUG] stored original_shape: {meta[closest_row]['original_shape']}")
    print(f"[DEBUG CROP] orig_img_p_pil.size (W,H): {orig_img_p_pil.size}")
    print(f"[DEBUG CROP] predicted_box_p: {predicted_box_p}")
    print(f"[DEBUG CROP] input_size_p: {input_size_p}")
    detection_crop_p = get_detection_crop_input(
        orig_img=orig_img_p_pil,
        box=predicted_box_p,
        input_size=input_size_p,
        context=ctx,
        draw_box=True,
    )

    # --- extra diagnostics precompute (after Mahalanobis 'closest_idx') ---
    gamma = post[0]  # responsibilities for the sample, shape (K,)
    pi_k = float(gmm.weights_[chosen_proto])  # mixture weight of chosen component

    # Mahalanobis distribution for TRAIN points of the chosen component
    diff_k = A_k - mu[None, :]
    y_k = diff_k @ L.T
    m2_k = np.sum(y_k * y_k, axis=1)  # Mahalanobis^2 of comp members

    y_star = (x_star - mu[None, :]) @ L.T
    m2_star = float(np.sum(y_star * y_star))
    # upper-tail (large M^2 more atypical; use >= to include ties)
    p_m2 = ((m2_k >= m2_star).sum() + 0.5) / (len(m2_k) + 1)

    # ------ TOP-K by combined score + ΔR fill-in ------
    # elementwise max of R and mean
    rel = channel_rels[0].detach().cpu()
    proto = mean.detach().cpu()
    combined = torch.max(rel, proto)

    #rel = channel_rels[0].detach().cpu()

    # # how many to take initially vs. by ΔR
    # init_k = int(np.ceil(n_concepts / 2))
    # fill_k = n_concepts - init_k

    # Take ALL concepts from sample's top relevances
    init_k = n_concepts
    fill_k = 0

    # initial top-k by combined score
    # init_idxs = torch.topk(combined, init_k).indices.tolist()

    init_idxs = torch.topk(rel, init_k).indices.tolist()

    #init_idxs = torch.topk(mean, init_k).indices.tolist()

    # ΔR = |R – mean|
    delta = (channel_rels[0].detach().cpu() - mean).abs()
    cand = torch.argsort(delta, descending=True).tolist()

    # fill with prototype’s top-|μ|, skipping duplicates
    proto_rank = torch.topk(proto.abs(), k=proto.numel()).indices.tolist()
    fill_idxs = [i for i in proto_rank if i not in init_idxs][:fill_k]

    topk_ind = init_idxs + fill_idxs

    # get reference images
    ref_imgs = get_ref_images(fv, topk_ind, layer_name,
                              composite=composite, class_id=class_id,
                              n_ref=n_refimgs,
                              ref_imgs_save_path=f"{ref_imgs_path}/ref_imgs_12/")

    # conditional heatmaps
    conditions = [{"y": class_id, layer_name: c} for c in topk_ind]
    attribution.take_prediction = prediction_idx
    cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions,
                                        composite, exclude_parallel=True)

    # ─── define cache dir & files ─────────────────────────────────────────────
    cache_dir = os.path.join(output_dir_pcx, "cache", layer_name,
                             f"class_{class_id}_protos_{num_prototypes}")
    os.makedirs(cache_dir, exist_ok=True)

    heatmap_cache = os.path.join(cache_dir, f"heatmap_p_protos{num_prototypes}.npy")
    cond_cache = os.path.join(cache_dir, f"cond_heatmap_p_protos{num_prototypes}.npy")
    predcls_cache = os.path.join(cache_dir, f"predicted_classes_p{num_prototypes}.npy")
    sortidxs_cache = os.path.join(cache_dir, f"sorted_idxs_p{num_prototypes}.npy")

    # # ─── load or compute & cache prot ──────────────────────────────────
    # if (os.path.exists(heatmap_cache) and os.path.exists(cond_cache)
    #         and os.path.exists(predcls_cache) and os.path.exists(sortidxs_cache) and use_cache):
    #
    #     # load heatmaps
    #     attr_p_heatmap = torch.from_numpy(np.load(heatmap_cache))
    #     cond_heatmap_p = torch.from_numpy(np.load(cond_cache))
    #
    #     # load predicted_classes_p & sorted_idxs_p
    #     predicted_classes_p = torch.from_numpy(np.load(predcls_cache))
    #     sorted_idxs_p = torch.from_numpy(np.load(sortidxs_cache))
    #
    #     print("Loaded attr_p_heatmap, cond_heatmap_p, predicted_classes_p, sorted_idxs_p from cache")
    #
    # else:

    # compute heatmaps
    attribution.take_prediction = box_idx_proto
    cond_heatmap_p, _, _, _ = attribution(
        data_p.requires_grad_(),
        conditions,
        composite,
        exclude_parallel=True
    )

    # compute attr_p for heatmap & predictions
    attribution.take_prediction = box_idx_proto
    attr_p = attribution(
        data_p.requires_grad_(),
        condition,
        composite,
        record_layer=[layer_name],
        init_rel=1
    )

    # detach → CPU → numpy
    heatmap_p_tensor = attr_p.heatmap.detach().cpu()
    cond_heatmap_p_tensor = cond_heatmap_p.detach().cpu()

    # compute predicted_classes and sorted_idxs_p
    predicted_classes_p = attr_p.prediction.argmax(dim=2)[0]
    sorted_idxs_p = attr_p.prediction.max(dim=2)[0].argsort(descending=True)[0]

        # # save all to disk
        # np.save(heatmap_cache, heatmap_p_tensor.numpy())
        # np.save(cond_cache, cond_heatmap_p_tensor.numpy())
        # np.save(predcls_cache, predicted_classes_p.cpu().numpy())
        # np.save(sortidxs_cache, sorted_idxs_p.cpu().numpy())
        #
    # also keep tensors around for immediate use
    attr_p_heatmap = heatmap_p_tensor
    cond_heatmap_p = cond_heatmap_p_tensor
        #
        # print("Saved attr_p_heatmap, cond_heatmap_p, predicted_classes_p, sorted_idxs_p to cache")

    if plot_prot_crops:
        import torchvision.transforms as T

        # ----- params for the proto+concept figure -----
        NUM_SAMPLES_PER_PROTO = 6  # thumbnails per prototype column
        K_CONCEPTS_PER_PROTO = 3  # how many concepts to take per prototype (union merged)
        N_REF_PER_CONCEPT = 6

        # --- arrays ---
        A_np = attributions.detach().cpu().numpy()  # [N, C]
        means = gmm.means_.astype(np.float32)  # [K, C]
        K, C = means.shape

        # --- nearest samples to each prototype mean ---
        diff = A_np[:, None, :] - gmm.means_[None, :, :]  # [N, K, C]
        L = getattr(gmm, "precisions_cholesky_", None)  # [K, C, C] (upper chol of precision)
        if L is not None:
            # y = diff @ L^T ; m2 = sum(y^2)
            y = np.einsum("nkc,kdc->nkd", diff, L)  # [N, K, C]
            m2 = np.einsum("nkd,nkd->nk", y, y)  # [N, K]
        else:
            # fallback: Euclidean if no precisions available
            m2 = np.einsum("nkc,nkc->nk", diff, diff)  # [N, K]
        ranked = np.argsort(m2, axis=0)  # smaller is closer
        num_take = min(NUM_SAMPLES_PER_PROTO, ranked.shape[0])
        top_idxs = ranked[:num_take, :]  # [num_take, K]

        # --- build crop strips per prototype (graceful fallback if no detection) ---
        all_crops = [[] for _ in range(K)]
        for pj in range(K):
            taken = 0
            k = 0
            max_try = min(ranked.shape[0], NUM_SAMPLES_PER_PROTO * 50)
            while taken < NUM_SAMPLES_PER_PROTO and k < max_try:
                # inside the for-pj loop, where you build `all_crops` from `ranked`:
                row_idx = int(ranked[k, pj]);
                k += 1
                ds_idx = int(meta[row_idx]["dataset_idx"])
                bx_idx = int(meta[row_idx]["box_idx"])
                img_data, _ = orig_dataset[ds_idx]
                orig_img_p = T.ToPILImage()(img_data) if torch.is_tensor(img_data) else img_data.copy()

                try:
                    if "box" in meta[row_idx]:
                        # Box is already in original coordinates
                        if "original_shape" in meta[row_idx]:
                            orig_shape_crop = meta[row_idx]["original_shape"]  # [H, W]
                            input_size_crop = (orig_shape_crop[1], orig_shape_crop[0])  # (W, H)
                        elif original_shape is not None:
                            input_size_crop = (original_shape[1], original_shape[0])
                        else:
                            input_size_crop = (640, 640)

                        crop = get_detection_crop_input(
                            orig_img=orig_img_p,
                            box=np.asarray(meta[row_idx]["box"], dtype=np.float32),
                            input_size=input_size_crop,
                            context=(2.0 if class_id == 0 else 0.4),
                            draw_box=True
                        )
                    else:
                        # fallback: re-detect and index
                        crop = get_detection_crop_exact(
                            model=model, orig_dataset=orig_dataset,
                            ds_idx=ds_idx, box_idx=bx_idx, device=device,
                            input_size=640, context=(2.0 if class_id == 0 else 0.4),
                            draw_box=True
                        )
                    t = torchvision.transforms.ToTensor()(crop)
                    all_crops[pj].append(t);
                    taken += 1
                except Exception:
                    # graceful fallback
                    fallback_tensor, _ = dataset[ds_idx]
                    fallback_uint8 = dataset.reverse_normalization(fallback_tensor).clamp(0, 255).byte()
                    fallback_pil = F.to_pil_image(fallback_uint8)
                    all_crops[pj].append(torchvision.transforms.ToTensor()(fallback_pil));
                    taken += 1

            # pad if still short
            while len(all_crops[pj]) < NUM_SAMPLES_PER_PROTO:
                row0 = int(ranked[0, pj])  # per-det row id
                ds0 = int(meta[row0]["dataset_idx"])  # map to dataset index
                fallback_tensor, _ = dataset[ds0]
                fallback_uint8 = dataset.reverse_normalization(fallback_tensor).clamp(0, 255).byte()
                fallback_pil = F.to_pil_image(fallback_uint8)
                all_crops[pj].append(torchvision.transforms.ToTensor()(fallback_pil))

        # --- select top concepts across prototypes (ONLY positives; if none, use top negatives) ---
        M = torch.from_numpy(means).float()  # [K, C]
        k_pick = int(min(K_CONCEPTS_PER_PROTO, C))

        per_proto_lists = []
        for pj in range(K):
            row = M[pj]  # [C]
            pos_mask = row > 0

            if torch.any(pos_mask):
                # take only positives (no mixing)
                num_pos = int(pos_mask.sum().item())
                take = min(k_pick, num_pos)
                masked = row.masked_fill(~pos_mask, float('-inf'))
                _, idx = torch.topk(masked, k=take, largest=True)
            else:
                # no positives -> take most negative entries
                _, idx = torch.topk(row, k=k_pick, largest=False)

            per_proto_lists.append(idx.tolist())

        # order-preserving unique merge across prototypes
        flat = [int(i) for lst in per_proto_lists for i in lst]
        seen = set();
        top_concepts = []
        for i in flat:
            if i not in seen:
                seen.add(i);
                top_concepts.append(i)

        top_concepts = [int(i) for i in top_concepts]
        N_CONCEPTS = len(top_concepts)

        # --- fetch concept reference images (cached) ---
        ref_imgs_concepts = get_ref_images(
            fv, top_concepts, layer_name,
            composite=composite, class_id=class_id,
            n_ref=N_REF_PER_CONCEPT,
            ref_imgs_save_path=f"{ref_imgs_path}/ref_imgs_6/"
        )

        # --- per-prototype coverage and similarity to class mean ---
        labels = gmm.predict(A_np)  # [N]
        counts = np.bincount(labels, minlength=K).astype(float)  # [K]
        coverage_pct = (counts / max(1, A_np.shape[0])) * 100.0  # [K]

        mu = A_np.mean(axis=0).astype(np.float32)  # [C]
        mu_n = mu / (np.linalg.norm(mu) + 1e-12)
        Pn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
        sim_mean = (Pn @ mu_n)  # [K], cosine-like similarity

        # --- concept weight matrix for current K ---
        concept_matrix = torch.from_numpy(means[:, top_concepts]).T  # [N_CONCEPTS, K]

        # --- make all thumbnails uniform square + vertical strips ---
        THUMB = 120
        resize_square = T.Compose([
            T.Resize(THUMB),  # smaller side -> THUMB (keeps aspect)
            T.CenterCrop((THUMB, THUMB))
        ])
        all_resized = [[resize_square(t.clamp(0, 1)) for t in col] for col in all_crops]

        # --- figure: (N_CONCEPTS+1) × (K+1); taller top row for vertical columns ---
        top_row_ratio = max(8, NUM_SAMPLES_PER_PROTO + 2)
        fig_pc, axs = plt.subplots(
            nrows=N_CONCEPTS + 1, ncols=K + 1,
            figsize=(K + 6, N_CONCEPTS + 6), dpi=170,
            gridspec_kw={'width_ratios': [6] + [1] * K, 'height_ratios': [top_row_ratio] + [1] * N_CONCEPTS}
        )

        # --- top row: prototype strips (VERTICAL) ---
        for pj in range(K):
            # nrow=1 -> one image per row => vertical strip
            grid = torchvision.utils.make_grid(all_resized[pj], nrow=1, padding=1)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)  # float[0,1] -> uint8

            axs[0, pj + 1].imshow(grid_np, aspect='auto')
            axs[0, pj + 1].set_title(
                f"Prototype {pj}\nCovers {coverage_pct[pj]:.0f}%\nSim. {sim_mean[pj]:.2f}",
                fontsize=9
            )
            axs[0, pj + 1].axis("off")
        axs[0, 0].axis("off")

        # --- first column: concept reference-image grids (also square + scaled) ---
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
                row0 = int(ranked[0, 0])  # per-det row id
                ds0 = int(meta[row0]["dataset_idx"])  # map to dataset index
                full_tensor, _ = dataset[ds0]
                full_uint8 = dataset.reverse_normalization(full_tensor).clamp(0, 255).byte()
                full_pil = F.to_pil_image(full_uint8)
                tiles = [resize_square(F.to_tensor(full_pil).clamp(0, 1))]

            nrow = max(1, min(len(tiles), N_REF_PER_CONCEPT))
            grid = make_grid(tiles, nrow=nrow, padding=0)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)

            axs[i + 1, 0].imshow(grid_np)
            axs[i + 1, 0].set_ylabel(f"concept {int(cidx)}", rotation=90, labelpad=8)
            axs[i + 1, 0].set_yticks([]);
            axs[i + 1, 0].set_xticks([])

        # --- inner cells: concept weight heat (signed; value printed) ---
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

        # save next to your other PCX plots
        plot_dir = f"../output_{dataset_type}/pcx/pcx_plots"
        out_png = os.path.join(plot_dir, f"{layer_name}_class{class_id}_K{K}_proto_concepts.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig_pc.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig_pc)
        print(f"➡️ prototype+concept figure: {out_png}")

    # --- use the SAME detection index you are looping over ---
    scores_det = scores_input  # [N, C]
    boxes_det = boxes_input  # [N, 4]

    if letterbox_shape is not None and original_shape is not None and rescale_boxes_fn is not None:
        boxes_det = rescale_boxes_fn(
            boxes_det,
            letterbox_shape=letterbox_shape,
            original_shape=original_shape
        )
        boxes_det = torch.from_numpy(boxes_det)

    scores_attr = attr.prediction[0].detach().cpu()  # [N_attr, C]
    # cosine-similarity between per-detection class score vectors
    A = scores_det / (scores_det.norm(dim=1, keepdim=True) + 1e-9)
    B = scores_attr / (scores_attr.norm(dim=1, keepdim=True) + 1e-9)
    sim = A @ B.T  # [N, N_attr]
    mapped = sim.argmax(dim=1)  # det idx (drawn) -> attr idx

    print(f"[DBG] prediction_num(draw)={prediction_idx}  mapped_to_attr_idx={int(mapped[prediction_idx])}")
    print(f"[DBG] class(draw)={int(scores_det[prediction_idx].argmax())}  "
          f"class(attr)={int(scores_attr[mapped[prediction_idx]].argmax())}")

    # Take that exact detection (no re-sorting, no class filtering)
    predicted_box = boxes_det[prediction_idx]

    # Confidence for the class you're explaining (could be non-max; that’s fine)
    pred_confidence = float(scores_det[prediction_idx, class_id].item())

    # thresholds
    MIX_THR = 0.01  # mixture lower-tail → "class outlier"
    LOCAL_THR = 0.01  # local lower-tail → "typical if >="
    M2_THR = 0.05  # Mahalanobis upper-tail → "typical if >="
    SMALL_THR = 0.10  # component is small if coverage or π_k <= 10%
    CONF_THR = 0.50  # detection confidence
    GAMMA_THR = 0.85

    global_outlier = (p_mix < MIX_THR)
    locally_typical = (p_local >= LOCAL_THR) or (p_m2 >= M2_THR) or (gamma[chosen_proto] >= GAMMA_THR)
    small_component = (coverage <= SMALL_THR) or (pi_k <= SMALL_THR)
    is_correct = (pred_confidence >= CONF_THR)

    show_extra_diag = global_outlier and locally_typical and small_component and is_correct

    # if you have a class‐name list somewhere, e.g. dataset.class_names:
    pred_label = dataset.class_names[class_id]

    # Get the box in letterbox coordinates (before rescaling)
    with torch.no_grad():
        _, boxes_raw = model.predict_with_boxes(data)
    boxes_raw = boxes_raw[0]  # [N, 4]
    box_letterbox = boxes_raw[prediction_idx].clone().detach().float()[None]

    colors = ["#ffcc00"]
    result = draw_bounding_boxes(
        (dataset.reverse_normalization(data[0])).type(torch.uint8),
        box_letterbox,
        colors=colors,
        width=8
    )
    img_ = F.to_pil_image(result)

    # model-input spatial size (W,H) = (data.shape[-1], data.shape[-2])
    inW, inH = int(data.shape[-1]), int(data.shape[-2])

    # predicted_box is already rescaled to original coords (done at line 510-516)
    detection_crop = get_detection_crop_input(
        orig_img=orig_img,
        box=predicted_box.detach().cpu().numpy() if torch.is_tensor(predicted_box) else predicted_box,  # ADD .detach()
        input_size=(original_shape[1], original_shape[0]) if original_shape is not None else (inW, inH),  # (W, H)
        context=ctx,
        draw_box=True,
    )

    # Draw a red rectangle on a copy to verify box location
    from PIL import ImageDraw
    debug_img = orig_img_p_pil.copy()
    draw = ImageDraw.Draw(debug_img)
    draw.rectangle(predicted_box_p.tolist(), outline="red", width=10)

    # # ---- export interactive HTML (per detection!) ----
    # safe_layer = layer_name.replace('.', '_')
    # print('[DBG] exporting HTML with test_predicted_box in input coords');
    # export_gmm_view_html(
    #     attributions=attributions,
    #     gmm=gmm,
    #     dataset=dataset,
    #     orig_dataset=orig_dataset,
    #     model=model,
    #     class_id=class_id,
    #     get_detection_crop_fn=get_detection_crop_exact,
    #     get_detection_crop_input_fn=get_detection_crop_input,
    #     meta=meta,
    #     test_channel_rels=channel_rels[0],
    #     test_orig_img=orig_img,
    #     test_predicted_box=predicted_box,
    #     test_context=ctx,
    #     device=device,
    #     input_size=(letterbox_shape[1], letterbox_shape[0]) if letterbox_shape is not None else (inW, inH),
    #     # USE ORIGINAL SHAPE (W, H)
    #     score_thresh=0.4,
    #     input_prediction_num=prediction_num,
    #     export_dir=f"../output_{dataset_type}/pcx/export_html/gmm_export_layer_{safe_layer}_class_{class_id}_det{int(prediction_num):02d}",
    #     max_points=2000,
    #     title=f"GMM 3D — {layer_name} | class {class_id} | det #{int(prediction_num)}",
    #     reducer="umap",
    #     reducer_n_components=3,
    #     reducer_kwargs=dict(n_neighbors=30, min_dist=0.05, metric="cosine")
    # )

    # from src.pcx_helper import export_gmm_view_2d
    #
    # fig = export_gmm_view_2d(
    #     attributions=attributions,
    #     gmm=gmm,
    #     dataset=dataset,
    #     orig_dataset=orig_dataset,
    #     model=model,
    #     class_id=class_id,
    #     get_detection_crop_fn=get_detection_crop_exact,
    #     get_detection_crop_input_fn=get_detection_crop_input,
    #     meta=meta,
    #     test_channel_rels=channel_rels[0],
    #     test_orig_img=orig_img,
    #     test_predicted_box=predicted_box,
    #     test_context=ctx,
    #     device=device,
    #     export_dir=f"../output_{dataset_type}/pcx/gmm_2d/{layer_name}",
    #     title=f"GMM 2D — {layer_name} | class {class_id}",
    #     reducer="umap",
    #     show_kde=True,
    # )
    # plt.close(fig)

    # ------- PLOTTING -------
    width_ratios = [1, 1, n_refimgs / 4, 1, 1, 1]
    n_rows = max(n_concepts, 5 if show_extra_diag else 4)
    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(4 * n_refimgs / 4, 1.8 * n_rows),
        gridspec_kw={'width_ratios': width_ratios},
        dpi=200
    )
    resize = torchvision.transforms.Resize((150, 150))

    print(f"[LL] mixture p={p_mix:.3f} | local p={p_local:.3f} | M2 p={p_m2:.3f} | "
          f"γ={gamma[chosen_proto]:.2f} | π={pi_k:.2f} | coverage={coverage * 100:.1f}% "
          f"| show_extra_diag={show_extra_diag}")

    # signed vs abs channel relevance (the one used for concept picking)
    chan_rel_signed = cc.attribute(attr.relevances[layer_name], abs_norm=False).detach().cpu()[0]
    chan_rel_abs = cc.attribute(attr.relevances[layer_name], abs_norm=True).detach().cpu()[0]

    print(f"[DBG] heatmap sum (signed): {float(attr.heatmap.detach().cpu().sum()):+.3f}")
    print("[DBG] sample of top channels by abs:",
          [(int(i), float(chan_rel_signed[i])) for i in torch.topk(chan_rel_abs, k=8).indices.tolist()])

    # Check model output on prototype image
    with torch.no_grad():
        scores_test, boxes_test = model.predict_with_boxes(data_p)

    scores_t = scores_test[0] if scores_test.ndim == 3 else scores_test
    boxes_t = boxes_test[0] if boxes_test.ndim == 3 else boxes_test

    print(f"\nModel found {boxes_t.shape[0]} detections on prototype image")
    print(f"Stored box_idx_proto: {box_idx_proto} (need at least {box_idx_proto + 1} detections)")
    print(f"Stored box_letterbox: {meta[closest_row]['box_letterbox']}")

    # Check if stored index is valid
    if box_idx_proto >= boxes_t.shape[0]:
        print(f"❌ ERROR: box_idx_proto={box_idx_proto} is OUT OF RANGE!")
    else:
        # Check what's at that index
        cls_at_idx = scores_t[box_idx_proto].argmax().item()
        conf_at_idx = scores_t[box_idx_proto].max().item()
        box_at_idx = boxes_t[box_idx_proto].cpu().tolist()
        print(f"\nDetection at index {box_idx_proto}:")
        print(f"  class={cls_at_idx}, conf={conf_at_idx:.3f}")
        print(f"  box={[f'{x:.1f}' for x in box_at_idx]}")

        # Compare with stored
        stored = meta[closest_row]['box_letterbox']
        print(f"\nStored box: {[f'{x:.1f}' for x in stored]}")

        # IoU
        b = boxes_t[box_idx_proto]
        sb = torch.tensor(stored, device=b.device)
        x1, y1 = max(sb[0], b[0]), max(sb[1], b[1])
        x2, y2 = min(sb[2], b[2]), min(sb[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (sb[2] - sb[0]) * (sb[3] - sb[1])
        area2 = (b[2] - b[0]) * (b[3] - b[1])
        iou = inter / (area1 + area2 - inter + 1e-6)
        print(f"  IoU with stored: {iou:.3f}")

        if iou < 0.5:
            print(f"❌ LOW IoU - detection at index {box_idx_proto} is NOT the same box!")

    # Find best matching detection
    print(f"\nSearching for best IoU match...")
    stored_box = torch.tensor(meta[closest_row]['box_letterbox'], device=boxes_t.device)
    best_iou, best_idx = 0, 0
    for i in range(boxes_t.shape[0]):
        b = boxes_t[i]
        x1, y1 = max(stored_box[0], b[0]), max(stored_box[1], b[1])
        x2, y2 = min(stored_box[2], b[2]), min(stored_box[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (stored_box[2] - stored_box[0]) * (stored_box[3] - stored_box[1])
        area2 = (b[2] - b[0]) * (b[3] - b[1])
        iou = float(inter / (area1 + area2 - inter + 1e-6))
        cls = scores_t[i].argmax().item()
        if iou > best_iou:
            best_iou, best_idx = iou, i
        if iou > 0.3:
            print(f"  Det {i}: IoU={iou:.3f}, class={cls}")

    print(f"\n✓ Best match: idx={best_idx}, IoU={best_iou:.3f}")
    print(f"  Should use box_idx_proto={best_idx} instead of {box_idx_proto}")

    # Populate the subplots with relevant visualizations for each selected concept
    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):

            if c in (1, 2, 3, 4) and r >= n_concepts:
                ax.axis("off")
                continue

            if c == 0:
                if r == 0:
                    ax.set_title("input")
                    img_ = img_.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_)
                elif r == 1:
                    ax.set_title("heatmap")
                    img = imgify(attr.heatmap.detach().cpu(), cmap="bwr", symmetric=True, level=5)
                    img = img.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img)
                elif r == 2:
                    ax.set_title("Detection", fontsize=10)
                    label_str = f"{pred_label} {pred_confidence * 100:.1f}%"
                    ax.text(
                        0.02, 0.98,  #
                        label_str,
                        transform=ax.transAxes,
                        fontsize=7,
                        fontweight="bold",
                        color="yellow",
                        va="top", ha="left",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="black",
                            alpha=0.3,
                            edgecolor="none")
                    )
                    ax.imshow(detection_crop)
                elif r == 3:
                    # density histogram of mixture log-likelihoods
                    a = ax.hist(scores, bins=30, density=True, color='k', alpha=0.85)
                    ax.vlines(score_star, 0, (a[0].max() if len(a[0]) else 1),
                              linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density");
                    ax.set_xlabel("log-likelihood")
                    ax.set_yticks([]);
                    ax.set_xticks([])

                    # Define threshold for outlier detection
                    lower_threshold = np.percentile(scores, 5)

                    # Determine if the sample is an outlier
                    outlier_text = "Outlier" if score_sample < lower_threshold else "Ordinary"
                    bbox_props = dict(boxstyle="round,pad=0.3",
                                      edgecolor="red" if outlier_text == "Outlier" else "green",
                                      facecolor="red" if outlier_text == "Outlier" else "green", alpha=0.3, linewidth=4)
                    ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10,
                            fontweight='bold', color="red" if outlier_text == "Outlier" else "green", bbox=bbox_props)

                else:
                    ax.axis("off")

            if c == 1:
                if r == 0:
                    ax.set_title("Input localization")
                cond_h =imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True, level=5)
                cond_h = cond_h.resize((150, 150), Image.BILINEAR)
                ax.imshow(cond_h)
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}")

            elif c == 2:
                if r == 0 and c == 2:
                    ax.set_title("concept visualization")
                grid = make_grid(
                    [resize(torch.from_numpy(np.asarray(i).copy()).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs / 2), padding=0)
                grid_np = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                ax.imshow(grid_np)
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
                    edge_color = "#ff0000"  # red for over-used
                elif delta_R < -2.5:
                    textstr = f"ΔR = {delta_R:+2.1f}\n⚠ under-used"
                    edge_color = "#ff0000"  # red for under-used
                else:
                    textstr = f"ΔR = {delta_R:+2.1f}\n✓ similar"
                    edge_color = "#00cc00"  # green for similar

                # Add a rectangle patch
                rect = patches.Rectangle((0, 0), 150, 150, linewidth=3, edgecolor=edge_color, facecolor='white')
                ax.add_patch(rect)
                # Split the text to handle the symbol and text separately
                lines = textstr.split('\n')
                symbol_line = lines[1]
                text_line = lines[0]

                # Add text with separate properties for the symbol
                ax.text(75, 60, text_line, fontsize=10, verticalalignment='center', horizontalalignment='center',
                        bbox=dict(facecolor=edge_color, edgecolor='none'))
                ax.text(75, 90, symbol_line, fontproperties=bold_font, verticalalignment='center',
                        horizontalalignment='center', color=edge_color)

                ax.set_xlim([0, 150])
                ax.set_ylim([0, 150])
                ax.axis("off")

            elif c == 5:
                if r == 0:
                    ax.set_title("prototype")
                    img_prototype = img_prototype.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_prototype)
                elif r == 1:
                    ax.set_title("heatmap")
                    img = imgify(attr_p_heatmap, cmap="bwr", symmetric=True, level=5)
                    img = img.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img)
                elif r == 2:
                    ax.set_title("detection")
                    ax.imshow(detection_crop_p)
                else:
                    ax.axis("off")
            elif c == 4:
                if r == 0:
                    ax.set_title("Prot localization")
                cond_h_p =imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True, level=5)
                cond_h_p = cond_h_p.resize((150, 150), Image.BILINEAR)
                ax.imshow(cond_h_p)
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(mean[topk_ind[r]] * 100):2.1f}")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig
