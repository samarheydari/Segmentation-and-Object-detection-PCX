import os
import random
import os
import sys

# Allow running this file directly from IDE "Run" button.
# When executed as a script, Python's import root is this file's folder
# (`.../experiments`), so sibling top-level packages like `datasets` are not
# visible unless we add project root (`.../code`) to sys.path.
if __package__ is None or __package__ == "":
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import json
import datetime
import logging

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


def setup_logger(save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"log_{tag}_{ts}.txt")

    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # prevent duplicate handlers in IDE reruns

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path


@click.command()
@click.option("--model_name", default="pidnet")
@click.option("--dataset_name", default="flood")
@click.option("--layer_name", default="conv1.0")  # backbone.layer4.0.conv3
@click.option("--num_samples", default=100)
@click.option("--batch_size", default=10)
@click.option("--insertion", default=False, type=bool)
@click.option("--rel_init", default="ones", type=str)
@click.option(
    "--mask_mode",
    default="gt",
    type=str,
    help="Evaluation mask: 'gt' (default) or 'pred0' (baseline predicted pixels).",
)
@click.option(
    "--log_dir",
    default="/home/heydari/paper/12-supp/code/results/instance_perturbation_logs",
    type=str,
    help="Directory to write run logs.",
)
def main(model_name, dataset_name, layer_name, num_samples, batch_size, insertion, rel_init, mask_mode, log_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model_masked = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model_masked = model_masked.to(device)
    model.eval()
    model_masked.eval()

    # Logging (does not change logic)
    run_tag = f"{dataset_name}_{model_name}_{layer_name}_{rel_init}_{'ins' if insertion else 'del'}_{mask_mode}"
    logger, log_path = setup_logger(log_dir, run_tag)
    logger.info(f"LOG FILE: {log_path}")
    logger.info(f"device={device}")
    logger.info(f"model.eval={not model.training} model_masked.eval={not model_masked.training}")
    logger.info(f"dataset={dataset_name} model={model_name} layer={layer_name}")
    logger.info(f"num_samples={num_samples} batch_size={batch_size} insertion={insertion} rel_init={rel_init}")
    logger.info(f"mask_mode={mask_mode} (gt=ground-truth mask, pred0=baseline predicted mask)")

    experiments = [{"label": "LRP-zplus", "name": "crvs_zplus"},
                   {"label": "LRP-gamma", "name": "crvs_gamma"},
                   {"label": "LRP-eps", "name": "crvs_eps"},
                   {"label": "GradCAM", "name": "crvs_gradcam"},
                   {"label": "Gradient", "name": "crvs_grad"},
                   {"label": "activation", "name": "cavs_max"},
                   {"label": "random", "name": "random"}]

    for exp in experiments:
        exp['vecs'] = []
        exp['samples'] = []

    logger.info("Loading concept vectors...")

    classes_unique = []
    for c in np.arange(0, n_classes):
        try:
            data = torch.load(
                f"/home/heydari/paper/12-supp/code/experiments/results/global_class_concepts/{dataset_name}_new2/{model_name}/{rel_init}/{layer_name}_class_{c}.pth",
                map_location="cpu",
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
            # keep original behavior (skip) but log it so you can debug later
            logger.info(f"[skip class {c}] cannot load concept file: {repr(e)}")
            continue

    logger.info(f"Loaded classes_unique={classes_unique}")
    print(classes_unique)

    # Keep original behavior (will still crash if nothing loaded)
    N = int(exp['vecs'][0].shape[-1])
    hooks = []
    neuron_indices = []

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

        class1 = np.array([random.choice(classes_unique) for _ in range(num_samples)])
        samples = [experiments[0]["samples"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)]
                   for j, c in enumerate(class1)]

        all_samples = np.array(samples)
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs_df = {}

        output = []
        pred0_ = []     # baseline predictions (for mask_mode=pred0)
        data_ = []
        targets_ = []

        subset = Subset(dataset, all_samples)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=8)

        logger.info("Running baseline forward pass...")
        for b in tqdm(dl):
            data = b[0].to(device)
            targets_.append(b[1])
            data_.append(data)
            out0 = model(data).detach().cpu()
            output.append(out0)
            pred0_.append(out0.argmax(dim=1))  # baseline predicted labels

        # Baseline reference for your metric (no perturbation):
        # This DOES NOT change perturbation logic; it's just an extra diagnostic.
        logger.info("Computing baseline reference (no perturbation) for mean logit-change metric...")
        neuron_indices = []  # ensures hook does nothing even if called
        baseline_vals = []
        for b in range(batches):
            targets = targets_[b]
            out_diff0 = (model_masked_mod(data_[b]).detach().cpu() - output[b])  # ideally ~0 everywhere

            # Downsample masks to logit res (same fix as perturbation loop)
            if targets.ndim == 3 and out_diff0.ndim == 4 and targets.shape[-2:] != out_diff0.shape[-2:]:
                targets_ds = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=out_diff0.shape[-2:],
                    mode="nearest"
                ).squeeze(1).long()
            else:
                targets_ds = targets

            pred0 = pred0_[b]
            if pred0.ndim == 3 and out_diff0.ndim == 4 and pred0.shape[-2:] != out_diff0.shape[-2:]:
                pred0_ds = F.interpolate(
                    pred0.unsqueeze(1).float(),
                    size=out_diff0.shape[-2:],
                    mode="nearest"
                ).squeeze(1).long()
            else:
                pred0_ds = pred0

            cls_batch = torch.tensor(class1)[b * batch_size: (b + 1) * batch_size].numpy()
            for ii, jj in enumerate(cls_batch):
                jj = int(jj)
                if mask_mode == "pred0":
                    mask = (pred0_ds[ii] == jj)
                else:
                    mask = (targets_ds[ii] == jj)
                den = mask.sum().clamp_min(1)
                val = ((out_diff0[ii, jj] * mask).sum() / den).item()
                baseline_vals.append(val)

        baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) else float("nan")
        baseline_std = float(np.std(baseline_vals)) if len(baseline_vals) else float("nan")
        logger.info(f"BASELINE metric (no perturbation): mean={baseline_mean:.6f} std={baseline_std:.6f} n={len(baseline_vals)}")

        # Main experiment loop (logic unchanged, just more explicit + logs + optional mask)
        for i, exp in enumerate(tqdm(experiments)):
            vecs = torch.stack([exp["vecs"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)]
                                for j, c in enumerate(class1)])
            diffs = []
            diffs_all = []
            topk1 = torch.topk(vecs, N)

            logger.info(f"=== EXPERIMENT {exp['label']} ===")

            for k in steps:
                diff = []
                neuron_indices_all = topk1.indices[:, :k]

                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    targets = targets_[b]
                    pred0 = pred0_[b]
                    out_diff = (model_masked_mod(data_[b]).detach().cpu() - output[b])

                    # --- downsample GT mask to logits resolution (nearest) ---
                    if targets.ndim == 3 and out_diff.ndim == 4 and targets.shape[-2:] != out_diff.shape[-2:]:
                        targets_ds = F.interpolate(
                            targets.unsqueeze(1).float(),
                            size=out_diff.shape[-2:],
                            mode="nearest"
                        ).squeeze(1).long()
                    else:
                        targets_ds = targets

                    # --- downsample pred0 to logits resolution (nearest), if needed ---
                    if pred0.ndim == 3 and out_diff.ndim == 4 and pred0.shape[-2:] != out_diff.shape[-2:]:
                        pred0_ds = F.interpolate(
                            pred0.unsqueeze(1).float(),
                            size=out_diff.shape[-2:],
                            mode="nearest"
                        ).squeeze(1).long()
                    else:
                        pred0_ds = pred0

                    cls_batch = torch.tensor(class1)[b * batch_size: (b + 1) * batch_size].numpy()
                    for ii, jj in enumerate(cls_batch):
                        jj = int(jj)
                        if mask_mode == "pred0":
                            mask = (pred0_ds[ii] == jj)
                        else:
                            mask = (targets_ds[ii] == jj)

                        den = mask.sum().clamp_min(1)
                        val = ((out_diff[ii, jj] * mask).sum() / den).item()
                        diff.append(val)

                mean_k = float(np.mean(diff))
                raw_std = float(np.std(diff))
                sem = raw_std / np.sqrt(len(diff)) if len(diff) else float("nan")
                diffs.append(mean_k)
                diffs_all.append(diff)

                # Console prints preserved
                print(mean_k, sem)

                # Requested checks: baseline vs small k, and small vs extreme
                if k == steps[0]:
                    logger.info(f"{exp['label']} small-k (k={k}): mean={mean_k:.6f} sem={sem:.6f} | baseline_mean={baseline_mean:.6f}")
                if k == steps[-1]:
                    logger.info(f"{exp['label']} last-k  (k={k}): mean={mean_k:.6f} sem={sem:.6f} | baseline_mean={baseline_mean:.6f}")

            diffs = np.array(diffs)
            diffs_all = np.array(diffs_all, dtype=object)  # keep ragged lists

            # Plot raw metric(k) curve (this is your diffs, unchanged meaning)
            plt.plot(steps, diffs, 'o--', label=exp["label"])
            diffs_df[exp["label"]] = diffs_all
            diffs_df["steps"] = steps

            # Save per-experiment raw curve summary JSON (extra, no logic change)
            # This is what you want for diagnosing "improves early vs late".
            curve_summary = {
                "dataset": dataset_name,
                "model": model_name,
                "layer": layer_name,
                "rel_init": rel_init,
                "insertion": bool(insertion),
                "mask_mode": mask_mode,
                "steps": [int(x) for x in steps.tolist()],
                "curve_mean_metric": [float(x) for x in diffs.tolist()],
                "baseline_mean_metric": baseline_mean,
                "baseline_std_metric": baseline_std,
                "n_samples": int(num_samples),
                "note": "curve_mean_metric is mean logit-change over mask pixels (same as original diffs).",
            }

            # Path is defined below, but we want to save now as well without changing flow:
            tmp_path = f"/home/heydari/paper/12-supp/code/results/instance_perturbation/{dataset_name}_new2/{model_name}/{rel_init}"
            os.makedirs(tmp_path, exist_ok=True)
            os.makedirs(tmp_path + "/data", exist_ok=True)
            json_path = os.path.join(
                tmp_path, "data",
                f"curve_{layer_name}_{exp['label']}_{'insertion' if insertion else 'deletion'}_{mask_mode}.json"
            )
            with open(json_path, "w") as f:
                json.dump(curve_summary, f, indent=2)
            logger.info(f"Saved curve JSON: {json_path}")

    plt.legend()
    plt.xlabel("flipped concepts")
    plt.ylabel("mean logit change (raw metric(k))")
    path = f"/home/heydari/paper/12-supp/code/results/instance_perturbation/{dataset_name}_new22/{model_name}/{rel_init}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/data", exist_ok=True)

    # Save plot + diffs_df (original behavior)
    if insertion:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}_insertion.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}_insertion.pth")
        logger.info(f"Saved plot: {path}/instance_perturbation_{layer_name}_insertion.pdf")
        logger.info(f"Saved data: {path}/data/instance_perturbation_{layer_name}_insertion.pth")
    else:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}.pth")
        logger.info(f"Saved plot: {path}/instance_perturbation_{layer_name}.pdf")
        logger.info(f"Saved data: {path}/data/instance_perturbation_{layer_name}.pth")

    # Also save a run config snapshot (extra)
    run_cfg = {
        "device": device,
        "dataset": dataset_name,
        "model": model_name,
        "layer": layer_name,
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "insertion": bool(insertion),
        "rel_init": rel_init,
        "mask_mode": mask_mode,
        "classes_unique": [int(x) for x in classes_unique],
        "steps": [int(x) for x in steps.tolist()],
        "baseline_mean_metric": baseline_mean,
        "baseline_std_metric": baseline_std,
        "log_file": log_path,
        "model_eval": bool(not model.training),
        "model_masked_eval": bool(not model_masked.training),
    }
    cfg_path = os.path.join(path, "data", f"run_config_{layer_name}_{'insertion' if insertion else 'deletion'}_{mask_mode}.json")
    with open(cfg_path, "w") as f:
        json.dump(run_cfg, f, indent=2)
    logger.info(f"Saved run config: {cfg_path}")

    plt.show()
    [hook.remove() for hook in hooks]
    logger.info("debug")


if __name__ == "__main__":
    main()