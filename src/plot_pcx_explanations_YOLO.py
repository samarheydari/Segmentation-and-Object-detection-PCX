import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# Add the parent directory to the Python path - bad practice, but it's just for the example
import sys
import logging
from PIL import Image

# Configure logging to display debug information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure parent directory is in path
sys.path.append("..")

# Helper imports for PCX and CRP explanations

#from src.minio_client import MinIOClient
from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
from sklearn.mixture import GaussianMixture
from LCRP.utils.render import vis_opaque_img_border
from crp.image import imgify
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import joblib
from PIL import ImageDraw
from src.pcx_helper import get_ref_images, get_detection_crop, get_detection_crop_input
from src.letterbox_utils import rescale_boxes

def plot_pcx_explanations(
    class_id, model_name, model, dataset, sample_id, n_concepts, n_refimgs, num_prototypes, prediction_num, layer_name,
        ref_imgs_path, output_dir_pcx, output_dir_crp):

    # Load the input image and label
    img, t = dataset[sample_id]

    # Generate the explanation figure
    fig = plot_one_image_pcx_explanation(
        model_name, model, img, dataset, class_id, n_concepts, n_refimgs, num_prototypes, prediction_num, layer_name,
        ref_imgs_path, output_dir_pcx, output_dir_crp)

    # Display and save the plot
    plt.figure(fig)
    plt.tight_layout()
    plot_dir = f"{output_dir_pcx}/pcx_plots"
    os.makedirs(plot_dir, exist_ok=True)
    safe_layer = layer_name.replace('.', '_')
    fname = (
        f"pcx_class{class_id}"
        f"_layer{safe_layer}"
        f"_sample{sample_id}"
        f"_n_prot{num_prototypes}"
        f"_nconc{n_concepts}.png"
    )
    fullpath = os.path.join(plot_dir, fname)
    fig.savefig(fullpath, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_one_image_pcx_explanation(
        model_name, model, img, orig_img, dataset, orig_dataset, class_id, n_concepts, n_refimgs, num_prototypes, prediction_num, layer_name,
        ref_imgs_path, output_dir_pcx, output_dir_crp, outside_logger=None
):
    import logging
    logger = outside_logger if outside_logger is not None else logging.getLogger(__name__)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model has to be in eval state
    model.to(device)
    model.eval()

    # Prepare layer names for CRP attribution
    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])
    num_prototypes = num_prototypes[class_id]

    # Setting up CRP
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_names,
                                    preprocess_fn=lambda x: x,
                                    path=output_dir_crp,
                                    max_target="max")
    cc = ChannelConcept()

    # Getting the sample we selected
    data = img
    data = data[None, ...].to(device)

    # Loading relevances for this layer
    folder = f"{output_dir_pcx}/{layer_name}/"
    attributions = torch.from_numpy(np.load(folder + f"attributions_{class_id}.npy"))

    meta_path = os.path.join(folder, f"meta_class_{class_id}.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        # Fallback: assume 1:1 mapping (old format)
        meta = [{"dataset_idx": i, "box_idx": 0} for i in range(len(attributions))]
        logger.warning(f"No metadata file found at {meta_path}, assuming 1:1 mapping")

    # Training GMM based on relevances if not done already
    # Initialize Gaussian Mixture Model (GMM) with specified number of prototypes as components
    cache_path = f'{output_dir_pcx}/gmms/gmm_cache_{layer_name}_class_{class_id}.pkl'
    prototype_cache_path = f'{output_dir_pcx}/gmm_prototypes/prototype_gmms_cache_{layer_name}_class_{class_id}.pkl'

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True)

    if os.path.exists(cache_path) and os.path.exists(prototype_cache_path):
        # Load the GMM and individual GMMs from the cache files
        gmm = joblib.load(cache_path)
        prototype_gmms = joblib.load(prototype_cache_path)
    else:
        # Fit the GMM
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions)
        # Create individual GMMs for each prototype and store them in a list
        prototype_gmms = [GaussianMixture(n_components=1, covariance_type='full', ) for p in range(num_prototypes)]
        # Save the GMM and individual GMMs to cache files
        joblib.dump(gmm, cache_path)
        joblib.dump(prototype_gmms, prototype_cache_path)

    for p, g_ in enumerate(prototype_gmms):
        g_._set_parameters([
            param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
            for j, param in enumerate(gmm._get_parameters())])

    # Calculating scores of the dataset, used further for outlier detection
    scores = gmm.score_samples(attributions)

    data = data.to(device).requires_grad_(True)

    # Running attribution on the input image
    attribution.take_prediction = prediction_num
    logger.debug(f"Running attribution on the input image, {attribution.take_prediction}") 
    attr = attribution(
            data,
            condition,
            composite,
            record_layer=[layer_name],
            init_rel=1)

    # Channel (neuron) relevance on the given layer for this image
    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)

    # --- sample fit (mixture) ---
    score_sample = gmm.score_samples(channel_rels.detach().cpu())  # shape (1,)

    # === PREP ARRAYS & CHOOSE PROTOTYPE (γ) ===
    x_star = channel_rels.detach().cpu().numpy()  # (1, C)
    A = attributions.detach().cpu().numpy().astype(np.float32)  # (N, C)

    post = gmm.predict_proba(x_star)  # (1, K) responsibilities
    chosen_proto = int(post.argmax(axis=1)[0])

    score_sample = float(score_sample[0])

    # === MEAN & MAHALANOBIS NEAREST SAMPLE ===
    mean = torch.from_numpy(gmm.means_[chosen_proto])

    mu = gmm.means_[chosen_proto].astype(np.float32)  # (C,)
    L = gmm.precisions_cholesky_[chosen_proto].astype(np.float32)  # (C, C), upper chol of precision
    diff = A - mu[None, :]
    y = diff @ L.T
    m2 = np.sum(y * y, axis=1)  # Mahalanobis^2

    closest_row = int(np.argmin(m2))
    ds_idx = int(meta[closest_row]["dataset_idx"])

    # Validate before accessing
    if ds_idx >= len(dataset):
        logger.error(f"ds_idx {ds_idx} >= dataset size {len(dataset)}")
        raise IndexError(f"Dataset index {ds_idx} out of range. Regenerate attributions with current dataset.")

    data_p, target_p = dataset[ds_idx]
    data_p = data_p[None, ...].to(device)

    # Getting top concepts/neurons for the given image in the given layer
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()

    # Getting reference images for those concepts
    ref_imgs = get_ref_images(fv, topk_ind, layer_name, composite=composite, class_id=class_id, n_ref=n_refimgs,
                              ref_imgs_save_path=ref_imgs_path)

    # This part is supposed to calculate conditional heatmaps and prototype heatmaps
    conditions = [{"y": class_id, layer_name: c} for c in topk_ind]

    attribution.take_prediction = prediction_num
    cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions, composite, exclude_parallel=True)
    logger.debug(f"Running conditional attribution on the input image, {attribution.take_prediction}")

    # ─── define cache dir & files ────────────────────────────────
    cache_dir = os.path.join(output_dir_pcx, "cache", layer_name, f"class_{class_id}_protos_{num_prototypes}")
    os.makedirs(cache_dir, exist_ok=True)
    heatmap_cache = os.path.join(cache_dir, f"attr_p_heatmap_protos{num_prototypes}.npy")
    cond_cache = os.path.join(cache_dir, f"cond_heatmap_p_protos{num_prototypes}.npy")

    # ─── load or compute & cache raw heatmap arrays ────────────────
    if os.path.exists(heatmap_cache) and os.path.exists(cond_cache):
        # load back into torch
        logger.debug("Loading prototype heatmaps from cache")
        attr_p_heatmap = torch.from_numpy(np.load(heatmap_cache))
        cond_heatmap_p = torch.from_numpy(np.load(cond_cache))
        logger.debug("Loaded prototype heatmaps from cache")
    else:
        logger.debug("Cache not found, computing fresh")
        # compute them fresh
        attribution.take_prediction = 0
        cond_heatmap_p, _, _, _ = attribution(
            data_p.requires_grad_(),
            conditions,
            composite,
            exclude_parallel=True
        )
        attribution.take_prediction = 0
        attr_p = attribution(
            data_p.requires_grad_(),
            condition,
            composite,
            record_layer=[layer_name],
            init_rel=1
        )

        # detach → CPU → numpy
        heatmap_p_tensor      = attr_p.heatmap.detach().cpu()
        cond_heatmap_p_tensor = cond_heatmap_p.detach().cpu()

        # save as .npy
        np.save(heatmap_cache,      heatmap_p_tensor.numpy())
        np.save(cond_cache,         cond_heatmap_p_tensor.numpy())
        attr_p_heatmap = heatmap_p_tensor

        logger.debug("Saved prototype heatmaps to cache")


    # This was here previously
    # predicted_boxes = model.predict_with_boxes(data)[1][0]
    # Rewriting for clarity
    input_scores, batch_predicted_boxes = model.predict_with_boxes(data)
    pred_confidence = input_scores[0, prediction_num, class_id].item()
    sample_predicted_boxes = batch_predicted_boxes[0]

    # This is already predicted as class_id
    predicted_boxes = sample_predicted_boxes[prediction_num]

    # predicted_classes = attr.prediction.argmax(dim=2)[0]
    # sorted = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
    # predicted_classes = predicted_classes[sorted]
    # predicted_boxes = predicted_boxes[sorted]
    # # Filter boxes for the desired class.
    # filtered_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id]

    # try:
    #     predicted_boxes = filtered_boxes[prediction_num]
    # except IndexError:
    #     print(f"Warning: No bounding box found for class {class_id} at index {prediction_num}.")
    #     raise IndexError(f"No bounding box found for class {class_id} at index {prediction_num}.")

    pred_label = dataset.class_names[class_id]
    boxes = predicted_boxes.clone().detach().float()[None]
    colors = ["#ffcc00" for _ in boxes]
    result = draw_bounding_boxes((dataset.reverse_normalization(data[0])).type(torch.uint8),
                                 boxes, colors=colors, width=8)

    img_ = F.to_pil_image(result)

    # Get bounding box coordinates.
    box_coords = predicted_boxes.clone().detach().cpu().numpy()
    x_min, y_min, x_max, y_max = box_coords.astype(int)
    orig_img_tensor = dataset.reverse_normalization(data[0])
    orig_img_pil = F.to_pil_image(orig_img_tensor.type(torch.uint8))

    # Zoom out by adding a margin
    orig_width, orig_height = orig_img_pil.size
    box_width = x_max - x_min
    box_height = y_max - y_min
    # choose zoom factor based on class
    zoom_factor = 0.4 if class_id == 1 else 2.0
    # compute margin
    margin_x = int(zoom_factor * box_width)
    margin_y = int(zoom_factor * box_height)
    crop_x_min = max(0, x_min - margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_x_max = min(orig_width, x_max + margin_x)
    crop_y_max = min(orig_height, y_max + margin_y)

    # Crop the detection region with extra context.
    cropped_img = orig_img_pil.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    # Draw the original bounding box (adjusted to the cropped image coordinates) with a thinner outline.
    draw = ImageDraw.Draw(cropped_img)
    adjusted_box = (x_min - crop_x_min, y_min - crop_y_min, x_max - crop_x_min, y_max - crop_y_min)
    draw.rectangle(adjusted_box, outline="yellow", width=2)

    # This was here previously
    # predicted_boxes = model.predict_with_boxes(data_p)[1][0]
    # Rewriting for clarity
    _, batch_predicted_boxes = model.predict_with_boxes(data_p)
    sample_predicted_boxes = batch_predicted_boxes[0]
    predicted_boxes_p = sample_predicted_boxes[0]

    # predicted_classes = attr_p.prediction.argmax(dim=2)[0]
    # print(f"Predicted boxes: {predicted_boxes}")
    # print(f"Predicted boxes shape: {predicted_boxes.shape}")

    # sorted = attr_p.prediction.max(dim=2)[0].argsort(descending=True)[0]
    # predicted_classes = predicted_classes[sorted]
    # predicted_boxes = predicted_boxes[sorted]
    # # Filter boxes for the d esired class.
    # filtered_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id]
    # predicted_boxes = filtered_boxes[prediction_num]

    boxes = predicted_boxes_p.clone().detach().float()[None]
    colors = ["#ffcc00" for _ in boxes]
    result = draw_bounding_boxes((dataset.reverse_normalization(data_p[0])).type(torch.uint8),
                                 boxes, colors=colors, width=8)

    img_prototype = F.to_pil_image(result)

    # Get bounding box coordinates.
    box_coords = predicted_boxes_p.clone().detach().cpu().numpy()
    x_min, y_min, x_max, y_max = box_coords.astype(int)
    orig_img_tensor = dataset.reverse_normalization(data_p[0])
    orig_img_pil = F.to_pil_image(orig_img_tensor.type(torch.uint8))

    # Zoom out by adding a margin
    orig_width, orig_height = orig_img_pil.size
    box_width = x_max - x_min
    box_height = y_max - y_min
    # choose zoom factor based on class
    zoom_factor = 0.4 if class_id == 1 else 2.0
    # compute margin
    margin_x = int(zoom_factor * box_width)
    margin_y = int(zoom_factor * box_height)
    crop_x_min = max(0, x_min - margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_x_max = min(orig_width, x_max + margin_x)
    crop_y_max = min(orig_height, y_max + margin_y)

    # Crop the detection region with extra context.
    cropped_img_prot = orig_img_pil.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    # Draw the original bounding box (adjusted to the cropped image coordinates) with a thinner outline.
    draw = ImageDraw.Draw(cropped_img_prot)
    adjusted_box = (x_min - crop_x_min, y_min - crop_y_min, x_max - crop_x_min, y_max - crop_y_min)
    draw.rectangle(adjusted_box, outline="yellow", width=2)

    # model-input spatial size (W,H) = (data.shape[-1], data.shape[-2])
    inW, inH = int(data.shape[-1]), int(data.shape[-2])

    # class-specific context (matches your earlier idea: more zoom for class 0)
    ctx = 2.0 if class_id == 0 else 0.4

    detection_crop_input = get_detection_crop_input(
        orig_img=orig_img,  # the original full-resolution image you passed into the function
        box=predicted_boxes,  # the chosen detection in model-input pixels
        input_size=(inW, inH),
        context=ctx,
        draw_box=True,
    )

    # Get original prototype image from orig_dataset
    orig_img_p, _ = orig_dataset[ds_idx]

    # Get original shape for prototype
    if isinstance(orig_img_p, np.ndarray):
        original_shape_p = orig_img_p.shape[:2]  # (H, W)
    elif isinstance(orig_img_p, Image.Image):
        original_shape_p = (orig_img_p.size[1], orig_img_p.size[0])  # PIL (W,H) -> (H,W)
    else:
        # Assume it's a tensor (C, H, W) or similar - unlikely for orig_dataset
        original_shape_p = orig_img_p.shape[-2:]

    # Get letterbox shape for prototype from transformed dataset
    letterbox_shape_p = (data_p.shape[2], data_p.shape[3])  # (H, W) from (B, C, H, W)

    # Get boxes in letterbox coordinates
    _, batch_predicted_boxes_p = model.predict_with_boxes(data_p)
    sample_predicted_boxes_p = batch_predicted_boxes_p[0]

    # Rescale boxes to original prototype image coordinates
    boxes_np_p = sample_predicted_boxes_p.cpu().detach().numpy()
    boxes_rescaled_p = rescale_boxes(boxes_np_p, letterbox_shape_p, original_shape_p)
    predicted_boxes_p_original = boxes_rescaled_p[0]  # First detection for prototype

    # Convert BGR to RGB if needed (orig_dataset might return BGR)
    if isinstance(orig_img_p, np.ndarray):
        orig_img_p_rgb = orig_img_p[:, :, ::-1].copy()
    else:
        orig_img_p_rgb = orig_img_p

    # Get prototype crop from original high-res image
    ctx_p = 2.0 if class_id == 0 else 0.4
    detection_crop_p = get_detection_crop_input(
        orig_img=orig_img_p_rgb,
        box=predicted_boxes_p_original,
        input_size=original_shape_p[::-1],  # (W, H)
        context=ctx_p,
        draw_box=True,
    )

    # --- Defining plot ---
    width_ratios = [1, 1, n_refimgs/4, 1, 1, 1]
    n_rows = max(n_concepts, 4)  # always at least 4 rows
    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(4 * n_refimgs / 4, 1.8 * n_rows),
        gridspec_kw={'width_ratios': width_ratios},
        dpi=200
    )
    resize = torchvision.transforms.Resize((150, 150))

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):

            if r >= n_concepts:
                if not (r == 3 and c == 0):
                    ax.axis("off")
                    continue

            # --- col 0: input / heatmap / detection / histogram ---
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
                    label_str = f"{pred_label} {pred_confidence*100:.1f}%"
                    ax.text(
                        0.02, 0.98,                       #
                        label_str,
                        transform=ax.transAxes,
                        fontsize= 7,
                        fontweight="bold",
                        color="yellow",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.3, edgecolor="none"))
                    ax.imshow(detection_crop_input)
                elif r == 3:
                    ax.set_title("class likelihood")
                    h = ax.hist(scores, bins=20, color='k')
                    ax.vlines(score_sample, 0, h[0].max(),
                              linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_xticks([]); ax.set_yticks([])

                    # Define threshold for outlier detection
                    lower_threshold = np.percentile(scores, 1)
                    logger.debug(
                        f"[OUTLIER] strategy=global_p5 | p5={lower_threshold:.3f} | score_sample={float(score_sample):.3f}")
                    outlier_text = "Outlier" if score_sample < lower_threshold else "Ordinary"
                    logger.debug(f"[OUTLIER] result={outlier_text}")

                    # Determine if the sample is an outlier
                    outlier_text = "Outlier" if score_sample < lower_threshold else "Ordinary"
                    bbox_props = dict(boxstyle="round,pad=0.3",
                                      edgecolor="red" if outlier_text == "Outlier" else "green",
                                      facecolor="red" if outlier_text == "Outlier" else "green", alpha=0.3, linewidth=4)
                    ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10,
                            fontweight='bold', color="red" if outlier_text == "Outlier" else "green", bbox=bbox_props)

                else:
                    ax.axis("off")

            # --- col 1: input localization ---
            elif c == 1:
                if r == 0:
                    ax.set_title("Input localization")
                cond_h =imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True, level=5)
                cond_h = cond_h.resize((150, 150), Image.BILINEAR)
                ax.imshow(cond_h)
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}")

            # --- col 2: reference imgs grid ---
            elif c == 2:
                if r == 0:
                    ax.set_title("concept visualization")
                grid = make_grid(
                    [resize(torch.from_numpy(np.asarray(i).copy())
                             .permute(2, 0, 1)) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs/2), padding=0
                )
                ax.imshow(grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                ax.yaxis.set_label_position("right")

            # --- col 3: ΔR boxes ---
            elif c == 3:
                if r == 0:
                    ax.set_title("Difference to prot")
                ax.imshow(np.zeros((150,150,3)), alpha=0.2)
                delta_R = ((channel_rels[0,topk_ind[r]].item()
                            - mean[topk_ind[r]].item())*100)
                if delta_R > 2.5:
                    txt, ec = f"ΔR = {delta_R:+2.1f}\n⚠ over-used", "#ff0000"
                elif delta_R < -2.5:
                    txt, ec = f"ΔR = {delta_R:+2.1f}\n⚠ under-used", "#ff0000"
                else:
                    txt, ec = f"ΔR = {delta_R:+2.1f}\n✓ similar", "#00cc00"
                rect = patches.Rectangle((0,0),150,150,
                                         linewidth=3, edgecolor=ec, facecolor="white")
                ax.add_patch(rect)
                l0, l1 = txt.split("\n")
                ax.text(75, 60, l0, ha="center", va="center",
                        bbox=dict(facecolor=ec, edgecolor="none"))
                ax.text(75, 90, l1, ha="center", va="center",
                        fontproperties=FontProperties(weight='bold'), color=ec)
                ax.axis("off")

            # --- col 4: proto localization ---
            elif c == 4:
                if r == 0:
                    ax.set_title("Prot localization")
                ax.imshow(imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True, level=5))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"concept {topk_ind[r]}\n"f"relevance: {mean[topk_ind[r]]*100:2.1f}")

            # --- col 5: prototype image / heatmap / detection ---
            elif c == 5:
                if r == 0:
                    ax.set_title("prototype")
                    img_prototype = img_prototype.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_prototype)
                elif r == 1:
                    img = imgify(attr_p_heatmap, cmap="bwr", symmetric=True, level=5)
                    img = img.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img)
                elif r == 2:
                    ax.set_title("detection")
                    ax.imshow(detection_crop_p)
                else:
                    ax.axis("off")

            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()

    return fig
