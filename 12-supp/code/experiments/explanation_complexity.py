import os

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import get_dataset
from models import get_model
from crp.helper import get_layer_names

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


@click.command()
@click.option("--model_name", default="yolov6")
@click.option("--dataset_name", default="coco2017")
@click.option("--rel_init", default="logits", type=str)
def main(model_name, dataset_name, rel_init):
    path = f"results/complexity/{dataset_name}/{model_name}"

    _, _, n_classes = get_dataset(dataset_name=dataset_name).values()
    model = get_model(model_name=model_name, classes=n_classes)
    model.eval()

    experiments = [{"label": "LRP-zplus",
                    "name": "crvs_zplus"},
                   {"label": "LRP-gamma",
                    "name": "crvs_gamma"},
                   {"label": "LRP-eps",
                    "name": "crvs_eps"},
                   {"label": "GradCAM",
                    "name": "crvs_gradcam"},
                   {"label": "Gradient",
                    "name": "crvs_grad"},
                   {"label": "activation",
                    "name": "cavs_max"}
                   ]

    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    layer_names_all = get_layer_names(model, [torch.nn.Conv2d])
    if "yolov5" in model_name:
        layer_names = layer_names[::4]
        mn = "YOLOv5"
        ds = "MS COCO 2017"
    elif "yolov6" in model_name:
        layer_names = layer_names[::4][:-2]
        mn = "YOLOv6"
        ds = "MS COCO 2017"
    elif "deeplab" in model_name:
        layer_names = layer_names[::3]
        mn = "DeepLabV3+"
        ds = "Pascal VOC 2012"
    elif model_name == "unet":
        mn = "UNet"
        ds = "CityScapes"
    else:
        mn = model_name
        ds = dataset_name

    print(" ".join(layer_names))

    for exp in experiments:
        exp["stds"] = []
        exp["nc80"] = []
    layers = []
    for layer in layer_names[:]:
        for exp in experiments:
            exp["vecs"] = []
        print(layer)
        for c in range(n_classes):
            for exp in experiments:
                try:
                    data = torch.load(
                        f"results/global_class_concepts/{dataset_name}/{model_name}/{rel_init}/{layer}_class_{c}.pth")
                    exp["vecs"].append(torch.stack(data[exp["name"]]).detach().cpu())
                except:
                    continue
        if len(exp["vecs"]) == 0:
            print("break?")
            break
        layers.append(layer)
        for exp in experiments:
            exp["stds"].append(torch.stack(
                [(v / (v.abs().sum(1)[:, None] + 1e-12)).std(0).mean() for v in exp["vecs"] if len(v) > 1]).mean())
            rels = torch.cat(exp["vecs"])
            rels = rels / rels.abs().sum(1)[:, None]
            rels_abs_sorted = torch.sort(rels.abs(), 1, descending=True).values
            cumulative = torch.cumsum(rels_abs_sorted, 1)
            num_concepts = (cumulative.numpy() <= 0.8).sum(1).mean(0) / rels.shape[-1] * 100
            exp["nc80"].append(num_concepts)

    labels = ["LRP-z$^+$", "LRP-$\gamma$", "LRP-$\\varepsilon$", "GradCAM", "gradient", "activation"]

    plt.figure(dpi=300, figsize=(10, 3))
    for k, exp in enumerate(experiments):
        plt.subplot(1, 2, 1)
        x = np.array(exp["stds"])*100
        plt.plot(x, '.-', label=f"{labels[k]} ({np.mean(x):2.2f})")
        plt.xticks(np.arange(len(x)), [layer_names_all.index(l) for l in layer_names], rotation=90)
        plt.xlabel("convolutional layer")
        plt.ylabel("relevance deviation (\%)")
        # plt.title("explanation variation")
        plt.title(f"{mn} - {ds}")

        plt.legend()
        plt.subplot(1, 2, 2)
        z = np.array(exp["nc80"])
        pl, = plt.plot(z, '.-', label=f"{labels[k]} ({np.mean(z):2.1f})")
        plt.xticks(np.arange(len(z)), [layer_names_all.index(l) for l in layer_names], rotation=90)
        # plt.title("explanation quantity")
        plt.title(f"{mn} - {ds}")
        plt.ylabel("concepts to form 80\% of relevance (\%)")
        plt.xlabel("convolutional layer")
        plt.legend()
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/complexity_{model_name}_{dataset_name}_{rel_init}.pdf", dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
