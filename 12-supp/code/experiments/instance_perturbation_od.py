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

random.seed(10)


def _parse_class_spec(class_spec):
    aliases = {"person": 0, "car": 1}
    out = []
    for tok in class_spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in aliases:
            out.append(aliases[tok])
        else:
            out.append(int(tok))
    if not out:
        raise click.ClickException("No valid classes provided.")
    return sorted(set(out))


def run_single(model_name, dataset_name, layer_name, num_samples, batch_size, insertion, rel_init,
               class_ids, sampling_mode):
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
    for c in class_ids:
        try:
            data = torch.load(
                f"results/global_class_concepts/{dataset_name}/{model_name}/{rel_init}/{layer_name}_class_{c}.pth"
            )
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
    if not classes_unique:
        raise RuntimeError(
            f"No concept vectors found for dataset='{dataset_name}', model='{model_name}', layer='{layer_name}'."
        )
    class_ids = classes_unique
    N = int(exp['vecs'][0].shape[-1])
    hooks = []
    neuron_indices = []

    if model_name == "yolov6s6":
        # Support both plain YOLO modules and wrapper objects (e.g. YoloV6S6_Model).
        for m in (model, model_masked):
            detect_module = None
            if hasattr(m, "detect"):
                detect_module = m.detect
            elif hasattr(m, "module") and hasattr(m.module, "detect"):
                detect_module = m.module.detect

            if detect_module is not None:
                setattr(detect_module, "sm", torch.nn.Identity())

            # Only override forward if a logits-forward exists.
            if hasattr(m, "forward_logits"):
                setattr(m, "forward", m.forward_logits)
            elif hasattr(m, "module") and hasattr(m.module, "forward_logits"):
                setattr(m, "forward", m.module.forward_logits)
    if model_name == "yolov5":
        setattr(model.model[24], "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.model[24], "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits) #self.explainable_output

    composite = Composite(canonizers=[CANONIZERS[model_name]()])

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
        if not hooks:
            raise RuntimeError(f"Layer '{layer_name}' not found in model '{model_name}'.")

        if len(class_ids) == 1:
            class1 = np.full(num_samples, class_ids[0], dtype=int)
        else:
            # Balanced split across classes for the "both classes" run.
            base = num_samples // len(class_ids)
            rem = num_samples % len(class_ids)
            class1 = []
            for i, c in enumerate(class_ids):
                class1.extend([c] * (base + (1 if i < rem else 0)))
            class1 = np.array(class1, dtype=int)
            np.random.shuffle(class1)

        # Select an index within each class bucket.
        per_item_pos = np.zeros(len(class1), dtype=int)
        for c in class_ids:
            idxs = np.where(class1 == c)[0]
            if len(idxs) == 0:
                continue
            c_idx = class_ids.index(c)
            n_available = len(experiments[0]["samples"][c_idx])
            if n_available <= 0:
                raise RuntimeError(f"No stored samples for class {c} in loaded concept vectors.")

            if sampling_mode == "best":
                vec = experiments[0]["vecs"][c_idx]
                scores = vec.abs().sum(dim=1).detach().cpu().numpy()
                rank = np.argsort(-scores)
                chosen = [int(rank[j % len(rank)]) for j in range(len(idxs))]
            else:
                chosen = [random.randrange(n_available) for _ in range(len(idxs))]
            per_item_pos[idxs] = np.array(chosen, dtype=int)

        samples = [experiments[0]["samples"][class_ids.index(c)][per_item_pos[j]] for j, c in enumerate(class1)]

        all_samples = np.array(samples)
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs_df = {}

        output = []
        data_ = []
        targets_ = []
        logit_offsets_ = []
        for b in range(batches):
            data = torch.stack([dataset[s][0] for s in all_samples[b * batch_size: (b + 1) * batch_size]])
            data = data.to(device)
            data_.append(data)
            out = model(data).detach().cpu()
            # Support both layouts:
            # 1) detection layout [..., 4=obj, 5: classes]
            # 2) class-score-only layout [..., classes]
            if out.shape[-1] > n_classes:
                logit_offset = 5
                confidences = torch.sigmoid(out[..., 5:]) * torch.sigmoid(out[..., 4:5])
            else:
                logit_offset = 0
                confidences = torch.sigmoid(out)
            logit_offsets_.append(logit_offset)

            t = [confidences[i, :, class1[b * batch_size: (b + 1) * batch_size][i]].argmax().item() for i in range(len(confidences))]
            targets_.append(t)
            output.append(torch.stack([
                out[i, t[i], logit_offset + class1[b * batch_size: (b + 1) * batch_size][i]]
                for i in range(len(confidences))
            ]))


        for i, exp in enumerate(tqdm(experiments)):
            vecs = torch.stack([exp["vecs"][class_ids.index(c)][per_item_pos[j]] for j, c in enumerate(class1)])
            diffs = []
            diffs_all = []
            topk1 = torch.topk(vecs, N)
            for k in steps:
                diff = []
                neuron_indices_all = topk1.indices[:, :k]
                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    out = model_masked_mod(data_[b]).detach().cpu()
                    t = targets_[b]
                    logit_offset = logit_offsets_[b]
                    cls_batch = class1[b * batch_size: (b + 1) * batch_size]
                    selected_logits = []
                    for i in range(len(t)):
                        cls_i = int(cls_batch[i])
                        # If detection count changed after masking, fallback to best
                        # available detection for the same class in current output.
                        if out.shape[1] == 0:
                            selected_logits.append(torch.tensor(0.0))
                            continue
                        det_idx = int(t[i])
                        if det_idx >= out.shape[1]:
                            if out.shape[-1] > n_classes:
                                conf_i = torch.sigmoid(out[i, :, 5 + cls_i]) * torch.sigmoid(out[i, :, 4])
                            else:
                                conf_i = torch.sigmoid(out[i, :, cls_i])
                            det_idx = int(conf_i.argmax().item())
                        selected_logits.append(out[i, det_idx, logit_offset + cls_i])

                    out_diff = torch.stack(selected_logits) - output[b]
                    diff.extend([o for o in out_diff.numpy()])
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
    class_tag = "c" + "-".join(str(int(c)) for c in class_ids)
    if insertion:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}_{class_tag}_insertion.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}_{class_tag}_insertion.pth")
    else:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}_{class_tag}.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}_{class_tag}.pth")
    plt.show()
    [hook.remove() for hook in hooks]
    print(f"done rel_init={rel_init} insertion={insertion} layer={layer_name} classes={class_ids} sampling={sampling_mode}")


@click.command()
@click.option("--model_name", default="yolov6s6")
@click.option("--dataset_name", default="person_car")
@click.option("--layer_name", default="module.backbone.ERBlock_2.0.rbr_1x1.conv")
@click.option("--num_samples", default=100)
@click.option("--batch_size", default=10)
@click.option("--insertion", default=False, type=bool)
@click.option("--rel_init", default="logits", help="Single init or comma-separated list, e.g. ones,prob,logits")
@click.option("--run_all_rel_init", default=False, type=bool, help="Run for ones,prob,logits")
@click.option("--run_both_insertions", default=False, type=bool, help="Run insertion=False and insertion=True")
@click.option("--class_spec", default="person,car", help="Classes as names/ids, e.g. person | car | person,car | 0,1")
@click.option("--run_class_splits", default=False, type=bool, help="Run person-only, car-only, and both")
@click.option("--sampling_mode", default="best", type=click.Choice(["best", "random"]), help="Sample selection mode")
def main(model_name, dataset_name, layer_name, num_samples, batch_size, insertion, rel_init,
         run_all_rel_init, run_both_insertions, class_spec, run_class_splits, sampling_mode):
    rel_inits = ["ones", "prob", "logits"] if run_all_rel_init else [r.strip() for r in rel_init.split(",") if r.strip()]
    if not rel_inits:
        raise click.ClickException("No valid rel_init values provided.")
    insertions = [False, True] if run_both_insertions else [insertion]
    class_runs = [[0], [1], [0, 1]] if run_class_splits else [_parse_class_spec(class_spec)]

    for rel in rel_inits:
        for ins in insertions:
            for cls in class_runs:
                print(f"Running rel_init={rel} insertion={ins} layer={layer_name} classes={cls} sampling={sampling_mode}")
                run_single(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    layer_name=layer_name,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    insertion=ins,
                    rel_init=rel,
                    class_ids=cls,
                    sampling_mode=sampling_mode,
                )


if __name__ == "__main__":
    main()
