import os 
import sys
import glob

# Allow running this file directly from IDE "Run" button.
# When executed as a script, Python's import root is this file's folder
# (`.../experiments`), so sibling top-level packages like `datasets` are not
# visible unless we add project root (`.../code`) to sys.path.
if __package__ is None or __package__ == "":
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
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
@click.option("--model_name", default="yolov6s6")
@click.option("--dataset_name", default="person_car")
@click.option("--layer_name", default="detect.reg_convs.2.conv")
@click.option("--insertion", default=True, type=bool)
@click.option("--rel_init", default="logits", type=str)
@click.option("--class_tag", default="c0", type=str, help="Class suffix used in filenames, e.g. c0, c1, c0-1")
def main(model_name, dataset_name, layer_name, insertion, rel_init, class_tag):

    if "yolo" in model_name:
        path_new = f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}/data"
        path_old = f"results/instance_perturbation/{dataset_name}/{model_name}/data"
        path = path_new if os.path.isdir(path_new) else path_old
    else:
        path = f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}/data"

    labels = ["LRP-z$^+$", "LRP-$\gamma$", "LRP-$\\varepsilon$", "GradCAM", "gradient", "activation",
              "random"]

    _, _, n_classes = get_dataset(dataset_name=dataset_name).values()
    model = get_model(model_name=model_name, classes=n_classes)
    model.eval()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    if "yolo" in model_name:
        layer_names = layer_names[::8]
    if "deeplab" in model_name:
        layer_names = layer_names[::3]
    print(" ".join(layer_names))

    def _load_layer_data(use_insertion):
        loaded_layers = []
        loaded_data = []
        suffix = "_insertion.pth" if use_insertion else ".pth"
        for layer_name in layer_names:
            # New filename format for YOLO experiments with class split.
            fname_new = f"{path}/instance_perturbation_{layer_name}_{class_tag}{suffix}"
            # Backward-compatible legacy format.
            fname_old = f"{path}/instance_perturbation_{layer_name}{suffix}"
            fname = fname_new if os.path.exists(fname_new) else fname_old
            if os.path.exists(fname):
                loaded_data.append(torch.load(fname, map_location="cpu", weights_only=False))
                loaded_layers.append(layer_name)
        return loaded_layers, loaded_data

    layers, data = _load_layer_data(insertion)
    if not data:
        other_mode = not insertion
        all_candidates = glob.glob(f"{path}/instance_perturbation_*.pth")
        if other_mode:
            other_available = [p for p in all_candidates if p.endswith("_insertion.pth")]
        else:
            other_available = [p for p in all_candidates if not p.endswith("_insertion.pth")]
        if other_available:
            click.echo(
                f"No files found for insertion={insertion} in '{path}'. "
                f"Falling back to insertion={other_mode}."
            )
            insertion = other_mode
            layers, data = _load_layer_data(insertion)

    if not data:
        expected = (
            f"{path}/instance_perturbation_<layer>_insertion.pth"
            if insertion else
            f"{path}/instance_perturbation_<layer>.pth"
        )
        raise click.ClickException(
            f"No perturbation files found in '{path}'. Expected pattern: {expected}"
        )

    plt.figure(figsize=(5, 3), dpi=300)
    methods = [m for m in data[0] if m != "steps"]
    for k, method in enumerate(methods):
        vals = []
        steps = data[0]["steps"]
        steps = np.concatenate([[0], steps]) / steps[-1] if not insertion else steps / steps[-1]
        steps = np.diff(steps)
        for i, layer in enumerate(layers):
            x = data[i][method]
            x = np.concatenate([x[0][None] * 0, x]) if not insertion else x - x[0, :][None]
            trapz = np.trapz(x, dx=steps[:, None], axis=0)
            trapz = trapz[trapz != 0]
            x = trapz
            if not len(x):
                x = data[i][method]
                x = np.concatenate([x[0][None] * 0, x]) if not insertion else x - x[0, :][None]
                x = np.trapz(x, dx=steps[:, None], axis=0)
            vals.append(x)
        valerrs = np.array([v.std() for v in vals]) / np.sqrt(np.array([len(v) for v in vals]))
        vals = np.array([v.mean() for v in vals])
        if not insertion:
            vals = - vals
        err = np.sqrt(np.sum(valerrs ** 2)) / len(valerrs)

        p, = plt.plot(layers, vals, '.-', label=f"{labels[k]} (${vals.mean():.4f}\\pm{err:.4f}$)", zorder=k / len(methods))
        plt.fill_between(layers, vals - valerrs, vals + valerrs, alpha=0.2, zorder=k - len(methods), color=p.get_color())

    plt.legend(fontsize="small")
    if model_name == "yolov5":
        mn = "YOLOv5"
        ds = "MS COCO 2017"
    elif model_name in ("yolov6", "yolov6s6"):
        mn = "YOLOv6"
        ds = "person_car"
    elif model_name == "deeplabv3plus":
        mn = "DeepLabV3+"
        ds = "Pascal VOC 2012"
    elif model_name == "unet":
        mn = "UNet"
        ds = "CityScapes"
    else:
        mn = model_name
        ds = dataset_name
    plt.title(f"{mn} - {ds}")
    plt.ylabel("AOC concept flipping" if not insertion else "AUC concept insertion")
    plt.xlabel("convolutional layer")
    plt.xticks(rotation=90)
    plt.xticks(layers, [get_layer_names(model, [torch.nn.Conv2d]).index(l) for l in layers])
    plt.tight_layout()
    if insertion:
        fname = f"{path}/concept_perturbation_{model_name}_{dataset_name}_{rel_init}_{class_tag}_insertion.pdf"
    else:
        fname = f"{path}/concept_perturbation_{model_name}_{dataset_name}_{rel_init}_{class_tag}.pdf"
    plt.savefig(fname, dpi=300, transparent=True)
    print(f"Saved plot: {fname}")
    plt.show()


if __name__ == "__main__":
    main()
