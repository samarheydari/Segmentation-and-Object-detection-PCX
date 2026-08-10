import os
import random

import click
import numpy as np
import torch
from crp.helper import get_layer_names
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from zennit.composites import COMPOSITES
from zennit.core import Composite

from datasets import get_dataset
from models import get_model

from utils.crp import ChannelConcept
from utils.crp_configs import ATTRIBUTORS, CANONIZERS
from utils.zennit_composites import EpsilonPlusFlat
import torch.nn.functional as F

random.seed(10)


@click.command()
# @click.option("--model_name", default="unet")
# @click.option("--dataset_name", default="cityscapes")
# @click.option("--layer_name", default="encoder.features.10")
# @click.option("--model_name", default="deeplabv3plus")
# @click.option("--dataset_name", default="voc2012")
# @click.option("--layer_name", default="backbone.layer3.0.conv3") #backbone.layer4.0.conv3
@click.option("--model_name", default="pidnet")
@click.option("--dataset_name", default="flood")
@click.option("--layer_name", default="layer3.1.conv1") #backbone.layer4.0.conv3
@click.option("--num_samples", default=40)
@click.option("--batch_size", default=10)
@click.option("--num_workers", default=0, type=int)
@click.option("--insertion", default=False, type=bool)
@click.option("--rel_init", default="logits", type=str)
@click.option("--result_tag", default="only_flood", type=str,
              help="Suffix for output folder, e.g. only_flood_120")
@click.option("--show_plot", default=False, type=bool,
              help="If True, call plt.show() at the end. Keep False for batch runs.")
def main(model_name, dataset_name, layer_name, num_samples, batch_size, num_workers, insertion, rel_init, result_tag,
         show_plot):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model_masked = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model_masked = model_masked.to(device)
    model.eval()
    model_masked.eval()

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
                    "name": "cavs_max"},
                   {"label": "random",
                    "name": "random"},
                   ]

    for exp in experiments:
        exp['vecs'] = []
        exp['samples'] = []

    print("Loading concept vectors...")

    classes_unique = []
    for c in np.arange(0, n_classes):
        # Flood is a binary task where class 1 is the foreground (flood).
        # For strict flood-quality evaluation, restrict perturbation to class 1.
        if dataset_name == "flood" and int(c) != 1:
            continue
        try:
            data = torch.load(
                f"results/global_class_concepts/{dataset_name}_onlyflood/{model_name}/{rel_init}/{layer_name}_class_{c}.pth",
                map_location="cpu",
                weights_only=False)
            for exp in experiments:
                if exp["label"] != "random":
                    exp['vecs'].append(torch.stack(data[exp['name']], 0))
                    exp['samples'].append(data["samples"])
                else:
                    exp['vecs'].append(torch.rand_like(torch.stack(data[experiments[0]['name']], 0)))
                    exp['samples'].append(data["samples"])
            classes_unique.append(c)
        except Exception as e:
            print(f"[warn] could not load class {c}: {e}")
            continue
    print(classes_unique)
    N = int(exp['vecs'][0].shape[-1])
    hooks = []
    neuron_indices = []

    composite = Composite(canonizers=[CANONIZERS[model_name]()])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    COLORS = prop_cycle.by_key()['color']

    plt.figure(dpi=300)
    steps = np.round(np.linspace(0, N, 15)).astype(int)
    steps = steps[1:] if not insertion else steps

    with composite.context(model_masked) as model_masked_mod:

        def hook(m, i, o):
            for b, batch_indices in enumerate(neuron_indices):
                if insertion:
                    indices_v = [x not in batch_indices for x in range(o.shape[1])]
                else:
                    indices_v = [x in batch_indices for x in range(o.shape[1])]
                o[b][indices_v] = o[b][indices_v] * 0

        for n, m in model_masked_mod.named_modules():
            if n == layer_name:
                hooks.append(m.register_forward_hook(hook))

        class1 = np.array([random.choice(classes_unique) for _ in range(num_samples)])
        samples = [experiments[0]["samples"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)] for j, c
                   in
                   enumerate(class1)]

        all_samples = np.array(samples)
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs_df = {}

        output = []
        data_ = []
        targets_ = []

        subset = Subset(dataset, all_samples)
        # Keep worker count configurable. In restricted/runtime-limited setups,
        # multiprocessing workers can fail due semaphore permissions.
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for b in tqdm(dl):
            # data = torch.stack([dataset[s][0] for s in all_samples[b * batch_size: (b + 1) * batch_size]])
            # targets_.append(torch.stack([dataset[s][1] for s in all_samples[b * batch_size: (b + 1) * batch_size]]))
            data = b[0].to(device)
            targets_.append(b[1])
            data_.append(data)
            output.append(model(data).detach().cpu())

        for i, exp in enumerate(tqdm(experiments)):
            vecs = torch.stack([exp["vecs"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)] for j, c
                                in enumerate(class1)])
            diffs = []
            diffs_all = []
            topk1 = torch.topk(vecs, N)
            for k in steps:
                diff = []
                neuron_indices_all = topk1.indices[:, :k]
                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    targets = targets_[b]
                    out_diff = (model_masked_mod(data_[b]).detach().cpu() - output[b])
                    # Some segmentation models (e.g., PIDNet) output logits at lower
                    # spatial resolution than GT masks. Align logits to mask size
                    # here so the perturbation metric can be computed consistently.
                    if out_diff.ndim == 4 and targets.ndim == 3 and out_diff.shape[-2:] != targets.shape[-2:]:
                        out_diff = F.interpolate(
                            out_diff, size=targets.shape[-2:], mode="bilinear", align_corners=False
                        )
                    diff.extend([(out_diff[i, j] * (targets[i] == j) / ((1 * (targets[i] == j)).sum() + 1e-12)).sum().numpy() for i, j in
                                 enumerate(torch.tensor(class1)[b * batch_size: (b + 1) * batch_size].numpy())])
                diffs.append(np.mean(diff))
                diffs_all.append(diff)
                mean_diff = float(np.mean(diff))
                raw_std = float(np.std(diff))
                sem = raw_std / np.sqrt(len(diff))
                print(f"mean={mean_diff:.6f} raw_std={raw_std:.6f} sem={sem:.6f} n={len(diff)}")
            diffs = np.array(diffs)
            diffs_all = np.array(diffs_all)
            plt.plot(steps, diffs, 'o--', label=exp["label"])
            diffs_df[exp["label"]] = diffs_all
            diffs_df["steps"] = steps

    plt.legend()
    plt.xlabel("flipped concepts")
    plt.ylabel("mean logit change")
    # Keep outputs isolated from generic runs and allow separate folders
    # for different sample counts (e.g., only_flood_120).
    rel_dir = f"{rel_init}_{result_tag}" if result_tag else rel_init
    path = f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_dir}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/data", exist_ok=True)
    if insertion:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}_insertion.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}_insertion.pth")
    else:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}.pth")
    if show_plot:
        plt.show()
    else:
        plt.close()
    [hook.remove() for hook in hooks]
    print("debug")


if __name__ == "__main__":
    main()
