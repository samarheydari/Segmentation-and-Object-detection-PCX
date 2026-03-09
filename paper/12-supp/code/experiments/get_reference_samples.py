import os

import click
import numpy as np
import torch
from crp.helper import get_layer_names
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode

from datasets import get_dataset
from models import get_model
from torchvision.utils import make_grid
import zennit.image as zimage

from utils.crp import ChannelConcept, ReceptiveFieldLocalization
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from utils.render import get_masks


@click.command()
@click.option("--model_name", default="yolov6", type=str)
@click.option("--dataset_name", default="coco2017", type=str)
@click.option("--layer", default="backbone.ERBlock_5.0.rbr_dense.conv", type=str)
@click.option("--neurons", default="177,445,485", type=str)
@click.option("--mode", default="relevance", type=str)
@click.option("--n_images", default=10, type=int)
@click.option("--n_rows", default=1, type=int)
@click.option("--masked", default=True, type=bool)  # currently only works with True
@click.option("--batch_size", default=5, type=int)
@click.option("--class_id", default=0, type=int)
@click.option("--recfield", default=True, type=bool)
def main(model_name, dataset_name, layer, neurons, mode, n_images, n_rows, masked, batch_size, class_id, recfield):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()
    model = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)

    dir = f"results/reference_samples/{dataset_name}/{model_name}/{layer}"
    os.makedirs(dir, exist_ok=True)

    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x,
                                    path=f"{model_name}_{dataset_name}", max_target="max", device=device)

    single_sample = fv.get_data_sample(111)[0]
    rf = ReceptiveFieldLocalization(attribution, single_sample, path=f"{model_name}_{dataset_name}")
    fv.add_receptive_field(rf)
    neurons = [int(n) for n in neurons.split(",")]
    try:
        if class_id is not None:
            ref_imgs = {}
            for c in neurons:
                ref_imgs[c] = fv.get_stats_reference(c, layer, [class_id], mode, (0, n_images), rf=recfield)[class_id]
        else:
            ref_imgs = fv.get_max_reference(neurons, layer, mode, (0, n_images), rf=recfield)

    except FileNotFoundError:
        print(f"Layer {layer} is analyzed regarding the receptive field of neurons.")
        rf.run({layer: cc}, canonizers=[CANONIZERS[model_name]()], batch_size=batch_size)
        ref_imgs = fv.get_max_reference(neurons, layer, mode, (0, n_images), rf=recfield)

    for c in neurons:
        ref_imgs[c] = [dataset.reverse_augmentation(img.detach().cpu()).int() for img in ref_imgs[c]]

    if masked:
        if class_id is not None:
            hms = {}
            for c in neurons:
                hms[c] = fv.get_stats_reference(c, layer, [class_id], mode, (0, n_images), rf=recfield,
                                                heatmap=True,
                                                composite=COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()]),
                                                batch_size=batch_size)[class_id]
        else:
            hms = fv.get_max_reference(neurons, layer, mode, (0, n_images), heatmap=True,
                                       composite=COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()]),
                                       rf=recfield,
                                       batch_size=batch_size)

        masks = get_masks({k: [x.numpy() for x in v] for k, v in hms.items()})

    s = 150 if recfield else 350
    resize = torchvision.transforms.Resize((s, s))
    resizem = torchvision.transforms.Resize((s, s), interpolation=InterpolationMode.NEAREST)
    for c in neurons:
        grid = make_grid([resize(i) for i in ref_imgs[c]], nrow=int(n_images / n_rows), padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        maskgrid = make_grid([resizem(torch.from_numpy(i) * 1.0) for i in masks[c]], nrow=int(n_images / n_rows),
                             padding=0)
        maskgrid = maskgrid[0].numpy()[..., None].astype(int) #* 0 + 1

        def make_alpha(m, img):
            m = resizem((torch.from_numpy(m) * 1.0))
            img = resize(img)
            mean_val = (m * img).sum() / m.sum() / 3
            return torch.ones_like(m) * (mean_val < 110) * 255

        alphagrid = make_grid([make_alpha(i, img) for i, img in zip(masks[c], ref_imgs[c])],
                              nrow=int(n_images / n_rows), padding=0)
        alphagrid = alphagrid[0].numpy()[..., None].astype(int)


        plt.close()
        w = 1.5 * n_images // n_rows
        plt.figure(figsize=(w, w * maskgrid.shape[0] / maskgrid.shape[1]), dpi=300)
        plt.imshow(grid * maskgrid + alphagrid * (1 - maskgrid), alpha=1)
        plt.imshow(grid, alpha=0.5)
        for i in [0, 1]:
            plt.contour(maskgrid[..., 0] == i, colors="black", levels=[0.5], linewidths=1.0)
        plt.xticks([])
        plt.yticks([])
        plt.axes().set_aspect(maskgrid.shape[0] / maskgrid.shape[1])
        plt.gca().set_axis_off()
        plt.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if class_id is None:
            fname = f"{dir}/reference_samples_{c}_{mode}_{model_name}_{layer}_{mode}.png"
        else:
            fname = f"{dir}/reference_samples_{c}_{mode}_{model_name}_{layer}_{class_id}_{mode}.png"
        plt.savefig(fname, dpi=300, transparent=True, pad_inches=0)
        plt.show()


if __name__ == "__main__":
    main()
