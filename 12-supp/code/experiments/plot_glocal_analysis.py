import copy
import os

import click
import numpy as np
import torch
from PIL import Image
from crp.helper import get_layer_names
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms.functional as F

from datasets import get_dataset
from models import get_model
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage

from utils.crp import ChannelConcept, ReceptiveFieldLocalization
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from utils.render import slice_img, gauss_p_norm


@click.command()
@click.option("--model_name", default="yolov6")
@click.option("--dataset_name", default="coco2017")
@click.option("--sample_id", default=2351, type=int)
@click.option("--img_path", default=None) #"/hardd/datasets/coco2017/coco/images/val2017/000000547383.jpg")
@click.option("--class_id", default=37)
@click.option("--layer", default="backbone.ERBlock_5.0.rbr_dense.conv", type=str)
@click.option("--prediction_num", default=0)
@click.option("--mode", default="relevance")
def main(model_name, dataset_name, sample_id, img_path, class_id, layer, prediction_num, mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()
    model = get_model(model_name=model_name, classes=n_classes)

    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    if img_path:
        img = Image.open(img_path)
        width = img.width
        height = img.height
        pad_y = max(0, int(width / 2 - height / 2))
        pad_x = max(0, int(height / 2 - width / 2))

        padding = torchvision.transforms.Pad((pad_x, pad_y), fill=1)
        img = torch.from_numpy(np.array(img) / 255).permute((2, 0, 1)).float()[None, :].to(device)
        img = padding(img)
    elif sample_id:
        img, t = dataset[sample_id]
        img = img[None, ...].to(device)
        width = img.shape[-2]
        height = img.shape[-1]
        pad_y = 0
        pad_x = 0

    ratio = height / width
    plt.figure(figsize=(3, 3 * ratio))
    plt.xticks([])
    plt.yticks([])
    plt.axes().set_aspect(ratio)

    if "deeplab" in model_name or "unet" in model_name:

        attr = attribution(copy.deepcopy(img).requires_grad_(), condition, composite, record_layer=[layer],
                           init_rel=1)
        heatmap = np.array(zimage.imgify(attr.heatmap, symmetric=True))
        heatmap = slice_img(heatmap, pad_x, pad_y)
        heatmap = zimage.imgify(heatmap, symmetric=True)

        predicted = attr.prediction[0]
        mask = predicted.argmax(dim=0) == class_id
        mask = mask.detach().cpu()


        sample_ = dataset.reverse_augmentation(img[0])
        sample_ = slice_img(sample_, pad_x=pad_x, pad_y=pad_y)
        mask_ = slice_img(mask, pad_x=pad_x, pad_y=pad_y)

        masked = draw_segmentation_masks(sample_, masks=mask_, alpha=0.5, colors=["red"])


        img_ = F.to_pil_image(masked)
        plt.imshow(np.asarray(img_))
        plt.contour(mask_, colors="black", linewidths=[2])
    elif "yolo" in model_name:
        attribution.take_prediction = prediction_num
        attr = attribution(img.requires_grad_(), condition, composite, record_layer=[layer], init_rel=1)
        heatmap = np.array(zimage.imgify(attr.heatmap, symmetric=True))
        heatmap = slice_img(heatmap, pad_x, pad_y)
        heatmap = zimage.imgify(heatmap, symmetric=True)

        predicted_boxes = model.predict_with_boxes(img)[1][0]
        print(attr.prediction.max(dim=2))
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        predicted_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]
        boxes = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]
        result = draw_bounding_boxes((img[0] * 255).type(torch.uint8), boxes, colors=colors, width=8)

        img_ = slice_img(result, pad_x=pad_x, pad_y=pad_y)
        img_ = F.to_pil_image(img_)
        attribution.take_prediction = 0

        plt.imshow(np.asarray(img_))
    else:
        raise NameError

    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    dir = f"results/glocal_analysis/{dataset_name}/{sample_id}"
    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/input_prediction_{model_name}_{class_id}.pdf", dpi=300,
                transparent=True, pad_inches=0)
    plt.savefig(f"{dir}/input_prediction_{model_name}_{class_id}.png", dpi=300,
                transparent=True, pad_inches=0)

    heatmap.save(f"{dir}/heatmap_{model_name}_{class_id}.png")

    if layer:
        cc = ChannelConcept()
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        layer_map = {layer: cc for layer in layer_names}
        fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x,
                                        path=f"{model_name}_{dataset_name}",
                                        max_target="max", device=device)
        single_sample = fv.get_data_sample(111)[0]
        rf = ReceptiveFieldLocalization(attribution, single_sample, path=f"{model_name}_{dataset_name}")
        print("RECEPTIVE FIELD COMPUTATION ONLY HAS TO RUN ONCE PER LAYER. COMMENT CODE AFTER IT HAS RUN.")
        rf.run({layer: cc}, canonizers=[CANONIZERS[model_name]()], batch_size=8)
        fv.add_receptive_field(rf)

        topk_c = 10

        if mode == "relevance":
            channel_rels = cc.attribute(attr.relevances[layer], abs_norm=True)
        else:

            channel_rels = attr.activations[layer].detach().cpu().flatten(start_dim=2).max(2)[0]
            channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]
        topk = torch.topk(channel_rels[0], topk_c).indices.detach().cpu().numpy()

        print(torch.topk(channel_rels[0], topk_c), channel_rels[0][343])
        conditions = [{"y": class_id, layer: c} for c in topk]
        if mode == "relevance":
            attribution.take_prediction = prediction_num
            heatmaps, _, _, _ = attribution(img.requires_grad_(), conditions, composite)
            attribution.take_prediction = 0
        else:
            heatmaps = torch.stack([attr.activations[layer][0][t] for t in topk]).detach().cpu()
        inp = img[0]
        for hm, ind in zip(heatmaps, topk):

            img = np.asarray(F.to_pil_image(inp))
            aspect_ratio = img.shape[0] / img.shape[1]
            alpha = (gauss_p_norm(hm) > 0.2)[..., None].astype(int)

            plt.close()
            if mode == "relevance":
                plt.figure(figsize=(3, 3 * aspect_ratio))
                mean_pixel_value = (img * alpha).sum() / (3 * alpha.sum())
                plt.imshow(img * alpha + 255 * (mean_pixel_value < 110) * (1 - alpha), alpha=1)
                plt.imshow(img, alpha=0.5)
                for i in [0, 1]:
                    plt.contour((gauss_p_norm(hm) > 0.2) == i, colors="black", levels=[0.5], linewidths=2)
                plt.xticks([])
                plt.yticks([])
                plt.axes().set_aspect(aspect_ratio)
                plt.gca().set_axis_off()
                plt.tight_layout()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(f"{dir}/channel_input_masked_{ind}_{model_name}_{class_id}.png",
                            dpi=300, transparent=True, pad_inches=0)

            heatmap = np.array(zimage.imgify(hm, symmetric=True))
            heatmap = slice_img(heatmap, pad_x, pad_y)
            heatmap = zimage.imgify(heatmap, symmetric=True)
            heatmap.save(f"{dir}/channel_heatmap_{ind}_{model_name}_{class_id}_{mode}.png")

        topk_img = 8
        ref_imgs = fv.get_max_reference(topk, layer, mode, (0, topk_img), rf=True)
        resize = torchvision.transforms.Resize((150, 150))
        for c in topk:
            grid = make_grid([resize(i) for i in ref_imgs[c]], nrow=int(topk_img / 2))
            grid = zimage.imgify(grid.detach().cpu())
            grid.save(f"{dir}/channel_samples_{c}_{model_name}_{class_id}_{mode}.png")


if __name__ == "__main__":
    main()
