import json
import os
import random
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from zennit.core import Composite

from datasets import get_dataset
from models import get_model
from utils.crp_configs import CANONIZERS

random.seed(10)
np.random.seed(10)


def layer_branch(layer_name: str) -> str:
    if layer_name.startswith("layer3_.") or layer_name.startswith("layer4_.") or layer_name.startswith("layer5_."):
        return "P-branch"
    if (
        layer_name.startswith("layer3_d.")
        or layer_name.startswith("layer4_d.")
        or layer_name.startswith("layer5_d.")
        or layer_name.startswith("diff")
        or layer_name.startswith("spp.")
        or layer_name.startswith("dfm.")
    ):
        return "D-branch"
    if (
        layer_name.startswith("layer1.")
        or layer_name.startswith("layer2.")
        or layer_name.startswith("layer3.")
        or layer_name.startswith("layer4.")
        or layer_name.startswith("layer5.")
        or layer_name.startswith("conv1.")
    ):
        return "I/trunk branch"
    if layer_name.startswith("pag") or layer_name.startswith("compression"):
        return "fusion block"
    if layer_name.startswith("final_layer."):
        return "final merged head"
    return "other"


def layer_merge_status(layer_name: str) -> dict:
    # "merged" here means the tensor at this layer already contains information
    # from more than one branch by design.
    if layer_name.startswith("dfm.") or layer_name.startswith("final_layer."):
        return {"is_merged": True, "merge_type": "three-branch merged (P+I+D)"}
    if layer_name.startswith("pag"):
        return {"is_merged": True, "merge_type": "two-branch merged (I+P)"}
    return {"is_merged": False, "merge_type": "not merged at this layer"}


def sample_metrics(logits_hw: torch.Tensor, target_hw: torch.Tensor, class_id: int = 1) -> dict:
    # logits_hw: [C, H, W], target_hw: [H, W]
    eps = 1e-12
    pred = torch.argmax(logits_hw, dim=0)

    target_pos = target_hw == class_id
    pred_pos = pred == class_id

    tp = torch.logical_and(pred_pos, target_pos).sum().item()
    fp = torch.logical_and(pred_pos, ~target_pos).sum().item()
    fn = torch.logical_and(~pred_pos, target_pos).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_acc = (pred == target_hw).float().mean().item()

    flood_logit = logits_hw[class_id]
    if target_pos.any():
        flood_logit_on_target = flood_logit[target_pos].mean().item()
    else:
        flood_logit_on_target = float("nan")

    return {
        "pixel_acc": float(pixel_acc),
        "flood_iou": float(iou),
        "flood_precision": float(precision),
        "flood_recall": float(recall),
        "flood_f1": float(f1),
        "flood_logit_on_target": float(flood_logit_on_target),
    }


def collect_layer_names_from_concepts(dataset_name: str, model_name: str, rel_init: str):
    base = Path(f"results/global_class_concepts/{dataset_name}/{model_name}/{rel_init}")
    layer_files = sorted(base.glob("*_class_1.pth"))
    return [p.name.replace("_class_1.pth", "") for p in layer_files]


@click.command()
@click.option("--model_name", default="pidnet")
@click.option("--dataset_name", default="flood")
@click.option("--rel_init", default="ones", type=str)
@click.option("--num_samples", default=100, type=int)
@click.option("--batch_size", default=2, type=int)
@click.option("--num_workers", default=0, type=int)
@click.option("--insertion", default=False, type=bool)
@click.option("--result_tag", default="only_flood_100_perf", type=str)
@click.option("--max_layers", default=0, type=int, help="0 means all layers.")
def main(model_name, dataset_name, rel_init, num_samples, batch_size, num_workers, insertion, result_tag, max_layers):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes).to(device)
    model_masked = get_model(model_name=model_name, classes=n_classes).to(device)
    model.eval()
    model_masked.eval()

    out_dir = Path(f"results/instance_perturbation_performance/{dataset_name}/{model_name}/{rel_init}_{result_tag}")
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    layer_names = collect_layer_names_from_concepts(dataset_name, model_name, rel_init)
    if max_layers > 0:
        layer_names = layer_names[:max_layers]

    experiments = [
        {"label": "LRP-zplus", "name": "crvs_zplus"},
        {"label": "LRP-gamma", "name": "crvs_gamma"},
        {"label": "LRP-eps", "name": "crvs_eps"},
        {"label": "GradCAM", "name": "crvs_gradcam"},
        {"label": "Gradient", "name": "crvs_grad"},
        {"label": "activation", "name": "cavs_max"},
        {"label": "random", "name": "random"},
    ]

    summary_rows = []

    for layer_name in layer_names:
        print(f"[run] layer={layer_name}")
        # Strict only-flood: class 1 only concept file.
        concept_path = Path(f"results/global_class_concepts/{dataset_name}/{model_name}/{rel_init}/{layer_name}_class_1.pth")
        if not concept_path.exists():
            print(f"[warn] missing concept file for layer: {layer_name}")
            continue

        data = torch.load(concept_path, map_location="cpu", weights_only=False)
        if len(data.get("samples", [])) == 0:
            print(f"[warn] empty samples for layer: {layer_name}")
            continue

        per_exp_vecs = {}
        for exp in experiments:
            if exp["label"] != "random":
                vec_list = data.get(exp["name"], [])
                if len(vec_list) == 0:
                    per_exp_vecs[exp["label"]] = None
                    continue
                per_exp_vecs[exp["label"]] = torch.stack(vec_list, 0)
            else:
                # Random baseline matched to zplus shape.
                z = data.get("crvs_zplus", [])
                if len(z) == 0:
                    per_exp_vecs[exp["label"]] = None
                    continue
                per_exp_vecs[exp["label"]] = torch.rand_like(torch.stack(z, 0))

        if per_exp_vecs["LRP-zplus"] is None:
            print(f"[warn] no usable concept vectors for layer: {layer_name}")
            continue

        num_concepts = int(per_exp_vecs["LRP-zplus"].shape[-1])
        steps = np.round(np.linspace(0, num_concepts, 15)).astype(int)
        steps = steps if insertion else steps[1:]

        sample_pool = np.array(data["samples"])
        # Allow replacement to support > available concept samples.
        sampled_indices = np.random.choice(np.arange(len(sample_pool)), size=num_samples, replace=True)
        all_samples = sample_pool[sampled_indices]

        subset = Subset(dataset, all_samples.tolist())
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Cache base outputs/targets once.
        cached_inputs = []
        cached_targets = []
        baseline_logits = []
        baseline_metrics = []
        for b in dl:
            x = b[0].to(device)
            y = b[1]
            with torch.no_grad():
                out = model(x).detach().cpu()

            if out.ndim == 4 and y.ndim == 3 and out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)

            cached_inputs.append(x)
            cached_targets.append(y)
            baseline_logits.append(out)
            for i in range(out.shape[0]):
                baseline_metrics.append(sample_metrics(out[i], y[i], class_id=1))

        # Forward hook to mask channels at selected layer output.
        hooks = []
        neuron_indices = []
        composite = Composite(canonizers=[CANONIZERS[model_name]()])

        with composite.context(model_masked) as model_masked_mod:
            def hook(_m, _i, o):
                for b_ix, batch_indices in enumerate(neuron_indices):
                    if insertion:
                        zero_mask = [ch not in batch_indices for ch in range(o.shape[1])]
                    else:
                        zero_mask = [ch in batch_indices for ch in range(o.shape[1])]
                    o[b_ix][zero_mask] = o[b_ix][zero_mask] * 0

            for n, m in model_masked_mod.named_modules():
                if n == layer_name:
                    hooks.append(m.register_forward_hook(hook))

            if not hooks:
                print(f"[warn] hook not registered for layer: {layer_name}")
                continue

            layer_result = {
                "layer_name": layer_name,
                "branch": layer_branch(layer_name),
                **layer_merge_status(layer_name),
                "insertion": bool(insertion),
                "num_samples": int(num_samples),
                "steps": steps.tolist(),
                "methods": {},
            }

            # Run all methods
            for exp in experiments:
                label = exp["label"]
                vecs_full = per_exp_vecs.get(label)
                if vecs_full is None:
                    print(f"[warn] skip method={label}, layer={layer_name} (no vecs)")
                    continue

                # Align to selected sampled rows.
                vecs = vecs_full[sampled_indices]
                topk = torch.topk(vecs, num_concepts)

                # Store per-step per-sample metrics
                metrics_store = {
                    "pixel_acc": [],
                    "flood_iou": [],
                    "flood_precision": [],
                    "flood_recall": [],
                    "flood_f1": [],
                    "flood_logit_on_target": [],
                    "delta_pixel_acc": [],
                    "delta_flood_iou": [],
                    "delta_flood_precision": [],
                    "delta_flood_recall": [],
                    "delta_flood_f1": [],
                    "delta_flood_logit_on_target": [],
                }

                for k in tqdm(steps, desc=f"{layer_name}::{label}", leave=False):
                    step_metrics = {m: [] for m in metrics_store.keys()}
                    neuron_indices_all = topk.indices[:, :k]

                    sample_offset = 0
                    for b_ix in range(len(cached_inputs)):
                        x = cached_inputs[b_ix]
                        y = cached_targets[b_ix]
                        neuron_indices = neuron_indices_all[sample_offset: sample_offset + x.shape[0]]

                        with torch.no_grad():
                            out = model_masked_mod(x).detach().cpu()
                        if out.ndim == 4 and y.ndim == 3 and out.shape[-2:] != y.shape[-2:]:
                            out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)

                        for i in range(out.shape[0]):
                            s_idx = sample_offset + i
                            cur = sample_metrics(out[i], y[i], class_id=1)
                            base = baseline_metrics[s_idx]
                            step_metrics["pixel_acc"].append(cur["pixel_acc"])
                            step_metrics["flood_iou"].append(cur["flood_iou"])
                            step_metrics["flood_precision"].append(cur["flood_precision"])
                            step_metrics["flood_recall"].append(cur["flood_recall"])
                            step_metrics["flood_f1"].append(cur["flood_f1"])
                            step_metrics["flood_logit_on_target"].append(cur["flood_logit_on_target"])
                            step_metrics["delta_pixel_acc"].append(cur["pixel_acc"] - base["pixel_acc"])
                            step_metrics["delta_flood_iou"].append(cur["flood_iou"] - base["flood_iou"])
                            step_metrics["delta_flood_precision"].append(cur["flood_precision"] - base["flood_precision"])
                            step_metrics["delta_flood_recall"].append(cur["flood_recall"] - base["flood_recall"])
                            step_metrics["delta_flood_f1"].append(cur["flood_f1"] - base["flood_f1"])
                            step_metrics["delta_flood_logit_on_target"].append(
                                cur["flood_logit_on_target"] - base["flood_logit_on_target"]
                            )

                        sample_offset += x.shape[0]

                    for mk in metrics_store.keys():
                        metrics_store[mk].append(step_metrics[mk])

                # Convert to arrays [steps, samples]
                layer_result["methods"][label] = {k: np.array(v, dtype=float) for k, v in metrics_store.items()}

                # Add compact summary rows (mean at last step).
                for mk, arr in layer_result["methods"][label].items():
                    arr = np.array(arr, dtype=float)
                    summary_rows.append(
                        {
                            "layer_name": layer_name,
                            "branch": layer_result["branch"],
                            "is_merged": layer_result["is_merged"],
                            "merge_type": layer_result["merge_type"],
                            "method": label,
                            "metric": mk,
                            "last_step_mean": float(np.nanmean(arr[-1])),
                            "last_step_std": float(np.nanstd(arr[-1])),
                            "n_samples": int(arr.shape[1]),
                        }
                    )

            # Save one file per layer
            torch.save(layer_result, data_dir / f"performance_{layer_name}{'_insertion' if insertion else ''}.pth")
            # Remove hooks for this layer before next layer
            [h.remove() for h in hooks]

    # Save global summary
    with open(data_dir / f"performance_summary{'_insertion' if insertion else ''}.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"[done] wrote layer files + summary to: {data_dir}")


if __name__ == "__main__":
    main()
