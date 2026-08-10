import os
import random
import glob

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

from LCRP.utils.crp import ChannelConcept
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS
from LCRP.utils.zennit_composites import EpsilonPlusFlat
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
@click.option("--layer_name", default="layer5.0.conv1") #backbone.layer4.0.conv3
@click.option("--num_samples", default=40)
@click.option("--batch_size", default=10)
@click.option("--insertion", default=False, type=bool)
@click.option("--rel_init", default="ones", type=str)
@click.option("--concept_root", default="results/global_class_concepts", type=str)
def main(model_name, dataset_name, layer_name, num_samples, batch_size, insertion, rel_init, concept_root):
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
    expected_pattern = os.path.join(
        concept_root, dataset_name, model_name, rel_init, f"{layer_name}_class_*.pth"
    )
    for c in np.arange(0, n_classes):
        try:
            data = torch.load(os.path.join(
                concept_root, dataset_name, model_name, rel_init, f"{layer_name}_class_{c}.pth"
            ))
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
    if not classes_unique:
        discovered = glob.glob(os.path.join(concept_root, dataset_name, model_name, rel_init, "*_class_*.pth"))
        discovered_preview = "\n".join(discovered[:10]) if discovered else "(none)"
        raise RuntimeError(
            "No concept vectors were found for this run.\n"
            f"Expected pattern:\n  {expected_pattern}\n\n"
            "This experiment needs precomputed global class concept files.\n"
            "Discovered files under the selected concept root:\n"
            f"{discovered_preview}\n\n"
            "If your concept files are in another folder, pass --concept_root <path>."
        )
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
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=8)

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
                    diff.extend([(out_diff[i, j] * (targets[i] == j) / ((1 * (targets[i] == j)).sum() + 1e-12)).sum().numpy() for i, j in
                                 enumerate(torch.tensor(class1)[b * batch_size: (b + 1) * batch_size].numpy())])
                diffs.append(np.mean(diff))
                diffs_all.append(diff)
                print(np.mean(diff), np.std(diff) / np.sqrt(len(diff)))
            diffs = np.array(diffs)
            diffs_all = np.array(diffs_all)
            plt.plot(steps, diffs, 'o--', label=exp["label"])
            diffs_df[exp["label"]] = diffs_all
            diffs_df["steps"] = steps

    plt.legend()
    plt.xlabel("flipped concepts")
    plt.ylabel("mean logit change")
    path = f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/data", exist_ok=True)
    if insertion:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}_insertion.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}_insertion.pth")
    else:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}.pth")
    plt.show()
    [hook.remove() for hook in hooks]
    print("debug")


if __name__ == "__main__":
    main()
