import os
import random
import os
import sys
import importlib

# Allow running this file directly from IDE "Run" button.
# When executed as a script, Python's import root is this file's folder
# (`.../experiments`), so sibling top-level packages like `datasets` are not
# visible unless we add project root (`.../code`) to sys.path.
project_root = os.path.abspath("/home/heydari/paper/Segmentation-and-Object-detection-PCX/")

if project_root not in sys.path:
    sys.path.insert(0, project_root)



import click
import numpy as np
import torch
import torch.nn.functional as F
from crp.helper import get_layer_names
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from zennit.composites import COMPOSITES
from zennit.core import Composite

from datasets import get_dataset
from LCRP.models import get_model

from LCRP.utils.crp import ChannelConcept
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS
from LCRP.utils.zennit_composites import EpsilonPlusFlat

random.seed(10)


def _torch_load_compat(path):
    """Load tensors saved by either NumPy 1.x or 2.x environments."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("numpy._core"):
            raise
        aliases = {
            "numpy._core": "numpy.core",
            "numpy._core.multiarray": "numpy.core.multiarray",
            "numpy._core.numeric": "numpy.core.numeric",
            "numpy._core.umath": "numpy.core.umath",
            "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        }
        for new_name, old_name in aliases.items():
            try:
                sys.modules.setdefault(new_name, importlib.import_module(old_name))
            except ModuleNotFoundError:
                pass
        return torch.load(path, map_location="cpu", weights_only=False)


@click.command()
# @click.option("--model_name", default="unet")
# @click.option("--dataset_name", default="cityscapes")
# @click.option("--layer_name", default="encoder.features.10")
# @click.option("--model_name", default="deeplabv3plus")
# @click.option("--dataset_name", default="voc2012")
# @click.option("--layer_name", default="backbone.layer3.0.conv3") #backbone.layer4.0.conv3
@click.option("--model_name", default="pidnet")
@click.option("--dataset_name", default="flood")
@click.option("--layer_name", default="final_layer.conv1")
@click.option("--num_samples", default=100)
@click.option("--batch_size", default=10)
@click.option("--insertion", default=False, type=bool)
@click.option("--rel_init", default="logits", type=str)
@click.option("--num_steps", default=23, type=int, help="Number of perturbation steps.")
@click.option("--balanced_sampling", default=True, type=bool, help="Sample target classes uniformly.")
@click.option("--min_mask_area", default=64, type=int, help="Minimum mask area in pixels; smaller masks contribute 0.")
@click.option("--seed", default=10, type=int, help="Random seed for reproducibility.")
@click.option(
    "--checkpoint_path",
    default=os.path.join(project_root, "LCRP/models/flood_model.pt"),
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    show_default=True,
    help="PIDNet checkpoint. The same weights are loaded into both models.",
)
def main(model_name, dataset_name, layer_name, num_samples, batch_size, insertion, rel_init,
         num_steps, balanced_sampling, min_mask_area, seed, checkpoint_path):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset_kwargs = {}
    if dataset_name == "flood" and model_name == "pidnet":
        # Match the paper/canonizer geometry: 720x1280 input -> 90x160 logits.
        dataset_kwargs["resize_hw"] = (720, 1280)
        dataset_kwargs["mask_downsample"] = 8
    dataset = test_dataset(**dataset_kwargs)

    model_kwargs = {"classes": n_classes}
    if model_name == "pidnet":
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"Loading PIDNet checkpoint into both models: {checkpoint_path}")
        model_kwargs["ckpt_path"] = checkpoint_path

    model = get_model(model_name=model_name, **model_kwargs)
    model_masked = get_model(model_name=model_name, **model_kwargs)
    if model_name == "pidnet":
        # Match the paper pipeline exactly: canonize and hook the native PIDNet,
        # rather than an adapter wrapper that changes the visible module graph.
        model = model.base_model
        model_masked = model_masked.base_model
    model = model.to(device)
    model_masked = model_masked.to(device)
    model.eval()
    model_masked.eval()

    def _to_logits(out):
        # PIDNet with augment=True returns [aux_p, main, aux_d].
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                raise RuntimeError("Model returned an empty list/tuple output.")
            out = out[1] if len(out) > 1 else out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Unsupported model output type: {type(out)}")
        return out

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
        try:
            data = _torch_load_compat(
                os.path.join(project_root, "results/global_class_concepts", dataset_name,
                             model_name, rel_init, f"{layer_name}_class_{c}.pth")
            )
            for exp in experiments:
                if exp["label"] != "random":
                    exp['vecs'].append(torch.stack(data[exp['name']], 0))
                    exp['samples'].append(data["samples"])
                else:
                    exp['vecs'].append(torch.rand_like(torch.stack(data[experiments[0]['name']], 0)))
                    exp['samples'].append(data["samples"])
            classes_unique.append(c)
        except Exception as e:
            print(f"[warn] could not load class {c} for layer '{layer_name}': {e}")
            continue
    print(classes_unique)
    if not classes_unique:
        raise click.ClickException(
            f"No concept vectors found for layer='{layer_name}', model='{model_name}', "
            f"dataset='{dataset_name}', rel_init='{rel_init}'."
        )
    N = int(exp['vecs'][0].shape[-1])
    hooks = []
    neuron_indices = []

    composite = Composite(canonizers=[CANONIZERS[model_name]()])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    COLORS = prop_cycle.by_key()['color']

    plt.figure(dpi=300)
    steps = np.round(np.linspace(0, N, max(2, int(num_steps)))).astype(int)
    steps = np.unique(steps)
    steps = steps[1:] if not insertion else steps
    if len(steps) == 0:
        steps = np.array([N], dtype=int)

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

        class_to_pos = {c: i for i, c in enumerate(classes_unique)}
        class_capacity = {c: len(experiments[0]["samples"][class_to_pos[c]]) for c in classes_unique}
        for c in classes_unique:
            if class_capacity[c] == 0:
                raise click.ClickException(f"Class {c} has zero samples in concept vectors.")

        if balanced_sampling:
            base = num_samples // len(classes_unique)
            rem = num_samples % len(classes_unique)
            class1 = []
            class1.extend([c for c in classes_unique for _ in range(base)])
            rem_classes = classes_unique.copy()
            random.shuffle(rem_classes)
            class1.extend(rem_classes[:rem])
            random.shuffle(class1)
            class1 = np.array(class1, dtype=int)
        else:
            class1 = np.array([random.choice(classes_unique) for _ in range(num_samples)], dtype=int)

        draw_map = {}
        for c in classes_unique:
            cnt = int((class1 == c).sum())
            if cnt == 0:
                continue
            cap = class_capacity[c]
            replace = cnt > cap
            if replace:
                print(f"[info] class {c}: requested {cnt} but only {cap} available. Sampling with replacement.")
            draw_map[c] = np.random.choice(cap, size=cnt, replace=replace)

        class_offsets = {c: 0 for c in classes_unique}
        sample_sel = []
        all_samples = []
        for c in class1:
            c = int(c)
            pos = class_to_pos[c]
            idx_in_class = int(draw_map[c][class_offsets[c]])
            class_offsets[c] += 1
            sample_sel.append((pos, idx_in_class))
            all_samples.append(experiments[0]["samples"][pos][idx_in_class])

        all_samples = np.array(all_samples)
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs_df = {}

        output = []
        data_ = []
        targets_ = []

        subset = Subset(dataset, all_samples)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=8)

        for b in tqdm(dl):
            data = b[0].to(device)
            targets_.append(b[1])
            data_.append(data)
            output.append(_to_logits(model(data)).detach().cpu())

        for i, exp in enumerate(tqdm(experiments)):
            vecs = torch.stack([exp["vecs"][pos][idx_in_class] for pos, idx_in_class in sample_sel])
            diffs = []
            diffs_all = []
            topk1 = torch.topk(vecs, N)
            for k in steps:
                diff = []
                neuron_indices_all = topk1.indices[:, :k]
                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    targets = targets_[b]
                    out_diff = (_to_logits(model_masked_mod(data_[b])).detach().cpu() - output[b])

                    cls_batch = torch.tensor(class1)[b * batch_size: (b + 1) * batch_size].numpy()
                    for ii, jj in enumerate(cls_batch):
                        logit_map = out_diff[ii, int(jj)]
                        mask = (targets[ii] == int(jj)).to(dtype=logit_map.dtype)
                        if tuple(mask.shape) != tuple(logit_map.shape[-2:]):
                            mask = F.interpolate(
                                mask[None, None],
                                size=logit_map.shape[-2:],
                                mode="nearest",
                            )[0, 0]
                        area = float(mask.sum().item())
                        if area < float(min_mask_area):
                            diff.append(0.0)
                            continue
                        val = (logit_map * mask / (area + 1e-12)).sum().item()
                        diff.append(float(val))
                diff = np.array(diff, dtype=float)
                diffs.append(float(diff.mean()))
                diffs_all.append(diff)
                print(float(diff.mean()), float(diff.std() / np.sqrt(len(diff))))
            diffs = np.array(diffs)
            diffs_all = np.array(diffs_all, dtype=float)
            plt.plot(steps, diffs, 'o--', label=exp["label"])
            diffs_df[exp["label"]] = diffs_all
            diffs_df["steps"] = steps

    plt.legend()
    plt.xlabel("flipped concepts")
    plt.ylabel("mean logit change")
    path = f"/home/heydari/paper/Segmentation-and-Object-detection-PCX/results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}"
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
