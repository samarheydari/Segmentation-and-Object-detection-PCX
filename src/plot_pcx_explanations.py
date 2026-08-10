import os
import torch
import copy
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the parent directory to the Python path - bad practice, but it's just for the example
import sys
sys.path.append("..")

from src.glocal_analysis import run_analysis 
from datasets.flood_dataset import FloodDataset
from src.datasets.DLR_dataset import DatasetDLR
from src.plot_crp_explanations import plot_explanations, plot_one_image_explanation
from src.minio_client import MinIOClient
from LCRP.models import get_model 
from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
from sklearn.mixture import GaussianMixture
from LCRP.utils.render import vis_opaque_img_border
from crp.image import imgify
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage
import h5py 
from PIL import Image
import torchvision.transforms.functional as F

from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import joblib
import plotly.graph_objects as go

device = "cuda" if torch.cuda.is_available() else "cpu"


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



def plot_pcx_explanations(model_name, model, dataset, sample_id, n_concepts=5, n_refimgs=12, num_prototypes=2, layer_name="decoder.center.0.0", ref_imgs_path="output/ref_imgs/", output_dir_pcx="output/pcx/unet_flood/", output_dir_crp="output/crp/unet_flood_old/"):
    # Model has to be in eval state
    model.eval()
    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])

    # Setting up CRP 
    attribution = ATTRIBUTORS["unet"](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": 1}]    
    fv = VISUALIZATIONS[model_name](attribution,
                                        dataset,
                                        layer_names,
                                        preprocess_fn=lambda x: x,
                                        path=output_dir_crp,
                                        max_target="max")
    cc = ChannelConcept()

    # Getting the sample we selected
    data, _ = fv.get_data_sample(sample_id, preprocessing=False)    

    # Loading relevances for this layer,
    folder = f"{output_dir_pcx}/{layer_name}/"
    attributions = torch.from_numpy(np.load(folder + "attributions.npy"))

    
    
    # Training GMM based on relevances if not done already
    # Initialize Gaussian Mixture Model (GMM) with specified number of prototypes as components
    # Fit the GMM fresh each run so prototypes can adapt
    gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions.numpy())

    # Create individual GMMs for each prototype and store them in a list
    prototype_gmms = [GaussianMixture(n_components=1, covariance_type='full',) for p in range(num_prototypes)]
    for p, g_ in enumerate(prototype_gmms):
        g_._set_parameters([
            param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
            for j, param in enumerate(gmm._get_parameters())])

    # Calculating scores of the dataset, used further for outlier detection
    scores = gmm.score_samples(attributions.numpy())

    # Running attribution on the input image
    attr = attribution(data.requires_grad_(), condition, composite, record_layer=[layer_name],
                           init_rel=1)
    
    # Channel (neuron) relevance on the given layer for this image
    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)
    
    # Finding how well given sample fits to prototypes, finding the closest prototype
    score_sample = gmm.score_samples(channel_rels.detach().cpu())
    likelihoods = [g_.score_samples(channel_rels.detach().cpu()) for g_ in prototype_gmms]
    mean = gmm.means_[np.argmax(likelihoods)]
    mean = torch.from_numpy(mean)
    closest_sample_to_mean = ((attributions - mean[None])).pow(2).sum(dim=1).argmin().item()


    #saving stuff
    joblib.dump((attributions, gmm, channel_rels.cpu(), mean), "output/pcx/gmm_data.pkl")

    # Closest prototype
    data_p, target_p = dataset[closest_sample_to_mean]
    data_p = data_p.to(device)[None]

    # Getting top concepts/neurons for the given image in the given layer
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    
    # Getting reference images for those concepts
    ref_imgs = get_ref_images(fv, topk_ind, layer_name, composite=composite, n_ref=n_refimgs, ref_imgs_save_path=ref_imgs_path)

    # This part is supposed to calculate conditional heatmaps and prototype heatmaps
    conditions = [{"y": 1, layer_name: c} for c in topk_ind]

    attr_p = attribution(data_p.requires_grad_(),[{"y": 1}], composite, record_layer=[layer_name])
    cond_heatmap_p, _, _, _ = attribution(data_p.requires_grad_(), conditions, composite)
    cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions, composite)

    # Mask for plotting segmentation
    mask = (attr.prediction[0].argmax(dim=0) == 1).detach().cpu()
    sample_ = dataset.reverse_augmentation(data)
    # Resizing mask in pidnet
    if "pidnet" in model_name:
        # Convert mask to float and add batch + channel dims
        mask = mask.float().unsqueeze(0).unsqueeze(0)  # shape (1, 1, 60, 60)

        # Interpolate to match sample_ spatial size
        resized_mask = torch.nn.functional.interpolate(mask, size=(sample_[:3, :, :][0].shape[1], sample_[:3, :, :][0].shape[2]), mode='nearest')  # keep it binary

        # Remove batch + channel dims and convert back to bool
        mask = resized_mask.bool().squeeze().squeeze()  # shape (480, 480)
    img_ = F.to_pil_image(draw_segmentation_masks(sample_[:3, :, :][0], masks=mask, alpha=0.3, colors=["red"]))

    # mask_prototype = (attr_p.prediction[0].argmax(dim=0) == 1).detach().cpu()
    mask_prototype = (((target_p - target_p.min()) / (target_p.max() - target_p.min())) > 0.5)[0]
    sample_prototype = dataset.reverse_augmentation(data_p)
    img_prototype = F.to_pil_image(draw_segmentation_masks(sample_prototype[:3, :, :][0], masks=mask_prototype, alpha=0.3, colors=["red"]))

    # Defining plot
    if n_concepts > 3:
        fig, axs = plt.subplots(n_concepts, 6, gridspec_kw={'width_ratios': [1, 1, n_refimgs / 4, 1, 1, 1]}, figsize=(4 * n_refimgs / 4, 1.8 * n_concepts), dpi=200)
    else: 
        fig, axs = plt.subplots(3, 6, gridspec_kw={'width_ratios': [1, 1, n_refimgs / 4, 1, 1, 1]}, figsize=(4 * n_refimgs / 4, 1.8 * 3), dpi=200)
    resize = torchvision.transforms.Resize((150, 150), antialias=True)

    # Populate the subplots with relevant visualizations for each selected concept
    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):

            if c == 0:
                if r == 0:
                    ax.set_title("input")
                    img = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
                    ax.imshow(img)
                    ax.imshow(np.asarray(img_))
                    ax.contour(mask, colors="black", linewidths=[1])
                elif r == 1:
                    ax.set_title("heatmap")
                    img = imgify(attr.heatmap.detach().cpu(), cmap="bwr", symmetric=True)
                    ax.imshow(img)
                elif r == 2:
                    ax.set_title("class likelihood")
                    a = ax.hist(scores, bins=20, color='k')
                    ax.vlines(score_sample, 0, a[0].max(), linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_yticks([])
                    ax.set_xticks([])

                    # Define threshold for outlier detection (e.g., below the 5th percentile or above the 95th percentile)
                    lower_threshold = np.percentile(scores, 1)
                    upper_threshold = np.percentile(scores, 99)

                    # Determine if the sample is an outlier
                    outlier_text = "Outlier" if score_sample < lower_threshold or score_sample > upper_threshold else "Ordinary"
                    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="red" if outlier_text == "Outlier" else "green",
                                        facecolor="red" if outlier_text == "Outlier" else "green", alpha=0.3, linewidth=4)
                    ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10, fontweight='bold', color="red" if outlier_text == "Outlier" else "green", bbox=bbox_props)
                else:
                    ax.axis("off")
            try:
                if c == 1:
                    if r == 0:
                        ax.set_title("Input localization")
                    ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True))
                    ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}\%")

                elif c == 2:
                    if r == 0 and c == 2:
                        ax.set_title("concept visualization")
                    grid = make_grid([resize(torch.from_numpy(np.asarray(i).copy()).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]], nrow=int(n_refimgs / 2), padding=0)
                    grid = np.array(zimage.imgify(grid.detach().cpu()))
                    img = imgify(ref_imgs[topk_ind[r]][c - 2], padding=True)
                    ax.imshow(grid)
                    ax.yaxis.set_label_position("right")

                elif c == 3:
                    plt.rc('text', usetex=False)
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    bold_font = FontProperties(weight='bold')

                    if r == 0:
                        ax.set_title("Difference to prot.")
                    ax.imshow(np.zeros((150, 150, 3)), alpha=0.2, cmap=None)
                    delta_R = (channel_rels[0][topk_ind[r]].round(decimals=3) - mean[topk_ind[r]].round(decimals=3)) * 100
                    if delta_R > 2:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n⚠ over-used"
                        edge_color = "#ff0000"  # red for over-used
                    elif delta_R < -2:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n⚠ under-used"
                        edge_color = "#ff0000"  # red for under-used
                    else:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n✓ similar"
                        edge_color = "#00cc00"  # green for similar
            
                    # Add a rectangle patch
                    rect = patches.Rectangle((0, 0), 150, 150, linewidth=3, edgecolor=edge_color, facecolor='white')
                    ax.add_patch(rect)
                    # Split the text to handle the symbol and text separately
                    lines = textstr.split('\n')
                    symbol_line = lines[1]
                    text_line = lines[0]

                    # Add text with separate properties for the symbol
                    ax.text(75, 60, text_line, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor=edge_color, edgecolor='none'))
                    ax.text(75, 90, symbol_line, fontproperties=bold_font, verticalalignment='center', horizontalalignment='center', color=edge_color)

                    ax.set_xlim([0, 150])
                    ax.set_ylim([0, 150])
                    ax.axis("off")

                elif c == 5:
                    if r == 0:
                        ax.set_title("prototype")
                        fv.dataset = dataset
                        img = imgify(fv.get_data_sample(closest_sample_to_mean, preprocessing=False)[0][0])
                        fv.dataset = dataset
                        ax.imshow(img)
                        ax.imshow(np.asarray(img_prototype))
                        ax.contour(mask_prototype, colors="black", linewidths=[1])
                    elif r == 1:
                        ax.set_title("heatmap")
                        img = imgify(attr_p.heatmap, cmap="bwr", symmetric=True)
                        ax.imshow(img)
                    else:
                        ax.axis("off")
                elif c == 4:
                    if r == 0:
                        ax.set_title("Prot localization")
                    ax.imshow(imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True))
                    ax.yaxis.set_label_position("right")

                    ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(mean[topk_ind[r]] * 100):2.1f}\%")
            except IndexError:
                axs[r][c].axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

    # add horizontal line
    #ax.plot([1/6, 5/6], [2/5 - 0.01, 2/5 - 0.01], color='lightgray', lw=1.5, ls="--",
    #        transform=plt.gcf().transFigure, clip_on=False)
    #ax.text(5/6, 2/5 - 0.008, "concepts sorted by $|R|$", transform=plt.gcf().transFigure, fontsize=10,
    #        verticalalignment='bottom', ha="right", clip_on=False, in_layout=False, color="gray")
    #ax.text(5/6, 2/5 - 0.013, "remaining concepts sorted by $|\\Delta R|$", transform=plt.gcf().transFigure, fontsize=10,
    #        verticalalignment='top', ha="right", clip_on=False, in_layout=False, color="gray")

    # Save and show the generated figures.
    plt.tight_layout()

    plt.show()

    return gmm, mean, channel_rels



def compute_outlier_scores(model_name, model, dataset, layer_name="decoder.center.0.0", num_prototypes=2, output_dir_pcx="output/pcx/unet_flood/"):  #automate the task of finding outlier samples

    #setting model to eval state
    model.eval()
    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])

    #Setting up for CRP
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_names,
                                     preprocess_fn=lambda x: x,
                                     path=output_dir_pcx,
                                     max_target="max")

    # Load or compute the GMM
    folder = f"{output_dir_pcx}/{layer_name}/"
    attributions = torch.from_numpy(np.load(folder + "attributions.npy"))
    gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions.numpy())

    #log-likelihood scores
    scores = gmm.score_samples(attributions.numpy())

    # Define outlier thresholds (e.g., 1st and 99th percentiles)
    lower_threshold = np.percentile(scores, 1)
    upper_threshold = np.percentile(scores, 99)

    # Get outlier indices
    outliers = [i for i, score in enumerate(scores)
                if score < lower_threshold or score > upper_threshold]

    return outliers, scores, lower_threshold, upper_threshold
