import os
import random
import sys

# Allow running this file directly from IDE "Run" button.
# When executed as a script, Python's import root is this file's folder
# (`.../experiments`), so sibling top-level packages like `datasets` are not
# visible unless we add project root (`.../code`) to sys.path.
if __package__ is None or __package__ == "":
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import click
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.core import Composite

from datasets import get_dataset
from models import get_model

from utils.crp_configs import CANONIZERS

import torch.nn.functional as F

random.seed(10)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

@click.command()
@click.option("--model_name", default="yolov6s6")
@click.option("--dataset_name", default="person_car")
@click.option("--layer_name", default="backbone.ERBlock_5.0.rbr_dense.conv")
@click.option("--num_samples", default=100)
@click.option("--batch_size", default=20)
@click.option("--class_id", default=0)
@click.option("--other_class_id", default=1)
@click.option("--concepts", default="177,478,445")
def main(model_name, dataset_name, layer_name, num_samples, batch_size, class_id, other_class_id, concepts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model_masked = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model_masked = model_masked.to(device)
    model.eval()
    model_masked.eval()

    experiments = [
                   {"label": "LRP-gamma",
                    "name": "crvs_gamma"},
                   ]

    for exp in experiments:
        exp['vecs'] = []
        exp['samples'] = []

    print("Loading concept vectors...")

    classes_unique = []
    for c in [class_id]:
        try:
            data = torch.load(
                f"results/global_class_concepts/{dataset_name}/{model_name}/{layer_name}_class_{c}.pth")
            for exp in experiments:
                if exp["label"] != "random":
                    exp['vecs'].append(torch.stack(data[exp['name']], 0))
                    exp['samples'].append(data["samples"])
                else:
                    exp['vecs'].append(torch.rand_like(torch.stack(data[experiments[0]['name']], 0)))
                    exp['samples'].append(data["samples"])
            classes_unique.append(c)
        except:
            continue
    print(classes_unique)
    N = int(exp['vecs'][0].shape[-1])
    hooks = []
    neuron_indices = []

    if model_name == "yolov6s6":
        setattr(model.detect, "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.detect, "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits)
    if model_name == "yolov5":
        setattr(model.model[24], "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.model[24], "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits)

    composite = Composite(canonizers=[CANONIZERS[model_name]()])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    COLORS = prop_cycle.by_key()['color']

    plt.figure(dpi=300, figsize=(3, 2.1))
    concepts = [int(c) for c in concepts.split(",")]
    steps = np.arange(len(concepts) + 1)


    with composite.context(model_masked) as model_masked_mod:

        def hook(m, i, o):
            for b, batch_indices in enumerate(neuron_indices):
                indices_v = [x in batch_indices for x in range(o.shape[1])]
                o[b][indices_v] = o[b][indices_v] * 0

        for n, m in model_masked_mod.named_modules():
            if n == layer_name:
                hooks.append(m.register_forward_hook(hook))

        class1 = np.array([class_id for _ in range(num_samples)])
        samples = np.array(experiments[0]["samples"]).flatten()

        all_samples = samples[:num_samples]
        print(len(samples), len(all_samples))
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs_df = {}

        output = []
        data_ = []
        targets_ = []
        other_classes = []
        for b in range(batches):
            other_class = torch.tensor(
                [other_class_id in dataset[s][1][:, 1].long() for s in all_samples[b * batch_size: (b + 1) * batch_size]]).cpu()
            other_classes.append(other_class)
            data = torch.stack([dataset[s][0] for s in all_samples[b * batch_size: (b + 1) * batch_size]])
            data_.append(data.cpu())
            out = model(data.to(device)).detach().cpu()
            confidences = torch.nn.functional.sigmoid(out[..., 5:]) * torch.nn.functional.sigmoid(out[..., 4:5])

            t = [confidences[i, :, class1[b * batch_size: (b + 1) * batch_size][i]].argmax().item() for i in range(len(confidences))]
            targets_.append(t)
            output.append(torch.stack([out[i, t[i], 5 + class1[b * batch_size: (b + 1) * batch_size][i]] for i in range(len(confidences))]))


        for i, exp in enumerate(tqdm(experiments)):
            vecs = torch.stack([exp["vecs"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)] for j, c
                                in enumerate(class1)])
            diffs = []
            diffs_all = []
            topk1 = torch.topk(vecs, N)
            for j, c in enumerate(concepts):
                topk1.indices[:, j] = c

            for k in steps:
                diff = []

                neuron_indices_all = topk1.indices[:, :k]
                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    out = model_masked_mod(data_[b].to(device)).detach().cpu()
                    t = targets_[b]
                    out_diff = (torch.stack([out[i, t[i], 5 + class1[b * batch_size: (b + 1) * batch_size][i]] for i in range(len(t))]) - output[b])  / (output[b].abs() + 1e-12)
                    diff.extend([o for o in out_diff.clamp(min=-1, max=1).numpy()]) #!!
                diffs.append(np.mean(diff))
                diffs_all.append(diff)
                print(np.mean(diff), np.std(diff) / np.sqrt(len(diff)))
            diffs = np.array(diffs)
            diffs_all = np.array(diffs_all)
            other_classes = torch.cat(other_classes)
            print(other_classes.sum())
            serr = diffs_all[:, other_classes].std(1) / np.sqrt(diffs_all[:, other_classes].shape[1])
            mean = diffs_all[:, other_classes].mean(1)
            mean = mean - mean[0]
            mean *= 100
            serr *= 100
            plt.fill_between(steps, mean - serr, mean + serr, alpha=0.2)
            plt.plot(steps, mean, 'o--', label="imgs with context object")
            serr = diffs_all[:, other_classes == False].std(1) / np.sqrt(diffs_all[:, other_classes == False].shape[1])
            mean = diffs_all[:, other_classes == False].mean(1)
            mean = mean - mean[0]
            mean *= 100
            serr *= 100
            plt.fill_between(steps, mean - serr, mean + serr, alpha=0.2)
            plt.plot(steps, mean, 'o--', label="imgs without context object")
            diffs_df[exp["label"]] = diffs_all
            diffs_df["steps"] = steps

    plt.legend()
    plt.xlabel("flipped context concepts")
    plt.ylabel("logit change (\%)")
    plt.xticks(steps)
    plt.yticks((-np.arange(7) + 1)[1::2])
    plt.ylim(-6.1, 1)
    plt.tight_layout()
    path = f"results/instance_perturbation/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/data", exist_ok=True)
    plt.savefig(f"{path}/instance_perturbation_{layer_name}_{class_id}_{other_class_id}.pdf", dpi=300, transparent=True)
    plt.show()
    [hook.remove() for hook in hooks]
    print("debug")


if __name__ == "__main__":
    main()
