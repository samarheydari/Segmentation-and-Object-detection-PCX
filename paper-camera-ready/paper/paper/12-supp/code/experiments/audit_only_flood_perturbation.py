import json
import re
from pathlib import Path

import click
import numpy as np
import torch

from crp.helper import get_layer_names
from datasets import get_dataset
from datasets.flood import _get_data_root, _resolve_dataset_root, flood_test
from models import get_model


def _canonical_id(path_stem: str) -> str:
    # Flood naming can differ between RGB and mask files:
    # image_0_   vs   image_0Ids_
    # Canonicalize by extracting the numeric token.
    m = re.search(r"image_(\d+)", path_stem)
    return m.group(1) if m else path_stem


def _load_pth(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


@click.command()
@click.option("--model_name", default="pidnet")
@click.option("--dataset_name", default="flood")
@click.option("--rel_init", default="ones")
@click.option("--result_tag", default="only_flood_120")
@click.option("--expected_samples", default=120, type=int)
@click.option("--strict_min_samples", default=101, type=int)
def main(model_name, dataset_name, rel_init, result_tag, expected_samples, strict_min_samples):
    out_dir = Path(f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}_{result_tag}/data")
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Missing result dir: {out_dir}")

    # Dataset checks
    root = _get_data_root()
    base = _resolve_dataset_root(root)
    split = "val" if (base / "RGB" / "val" / "JPEG").is_dir() else "test"
    ds = flood_test()

    pair_examples = []
    stem_match_count = 0
    canon_match_count = 0
    for i, (ip, mp) in enumerate(ds.samples):
        stem_match = ip.stem == mp.stem
        canon_match = _canonical_id(ip.stem) == _canonical_id(mp.stem)
        stem_match_count += int(stem_match)
        canon_match_count += int(canon_match)
        if i < 20:
            pair_examples.append(
                {
                    "image": str(ip),
                    "mask": str(mp),
                    "stem_match": stem_match,
                    "canonical_id_match": canon_match,
                }
            )

    img_shape_ok, img_dtype_ok = True, True
    mask_shape_ok, mask_dtype_ok = True, True
    img_stats = []
    unique_vals = set()
    flood_pixels = []
    for i in range(len(ds)):
        x, y = ds[i]
        if not (x.ndim == 3 and x.shape[0] in (3, 4)):
            img_shape_ok = False
        if x.dtype != torch.float32:
            img_dtype_ok = False
        img_stats.append((float(x.min()), float(x.max()), float(x.mean())))

        if y.ndim != 2:
            mask_shape_ok = False
        if y.dtype != torch.long:
            mask_dtype_ok = False
        unique_vals.update(int(v) for v in torch.unique(y).cpu().numpy().tolist())
        flood_pixels.append(int((y == 1).sum().item()))

    # Result checks
    flip_files = sorted([p for p in out_dir.glob("instance_perturbation_*.pth") if "_insertion" not in p.name])
    ins_files = sorted(out_dir.glob("instance_perturbation_*_insertion.pth"))

    _, _, n_classes = get_dataset(dataset_name=dataset_name).values()
    model = get_model(model_name=model_name, classes=n_classes)
    model.eval()
    expected_layers = set(get_layer_names(model, [torch.nn.Conv2d]))

    flip_layer_map, ins_layer_map = {}, {}
    for p in flip_files:
        layer = re.sub(r"^instance_perturbation_|\.pth$", "", p.name)
        d = _load_pth(p)
        m = [k for k in d if k != "steps"][0]
        n = int(np.array(d[m]).shape[1])
        flip_layer_map[layer] = {"path": str(p), "n": n}
    for p in ins_files:
        layer = re.sub(r"^instance_perturbation_|_insertion\.pth$", "", p.name)
        d = _load_pth(p)
        m = [k for k in d if k != "steps"][0]
        n = int(np.array(d[m]).shape[1])
        ins_layer_map[layer] = {"path": str(p), "n": n}

    missing_flip = sorted(expected_layers - set(flip_layer_map))
    missing_ins = sorted(expected_layers - set(ins_layer_map))
    stale_flip = sorted([l for l, v in flip_layer_map.items() if v["n"] < expected_samples])
    stale_ins = sorted([l for l, v in ins_layer_map.items() if v["n"] < expected_samples])
    strict_layers = sorted(
        [
            l
            for l in (set(flip_layer_map) & set(ins_layer_map))
            if flip_layer_map[l]["n"] >= strict_min_samples and ins_layer_map[l]["n"] >= strict_min_samples
        ]
    )

    # Metric stats on strict layers (signed-logit areas, as implemented)
    methods = [k for k in _load_pth(Path(next(iter(flip_layer_map.values()))["path"])) if k != "steps"]
    metric_stats = {}
    for m in methods:
        auc_vals, aoc_vals = [], []
        for l in strict_layers:
            di = _load_pth(Path(ins_layer_map[l]["path"]))
            si = np.array(di["steps"], dtype=float)
            dxi = np.diff(si / si[-1])
            xi = np.array(di[m], dtype=float) - np.array(di[m], dtype=float)[0, :][None]
            auc_vals.extend(np.trapz(xi, dx=dxi[:, None], axis=0).tolist())

            df = _load_pth(Path(flip_layer_map[l]["path"]))
            sf = np.array(df["steps"], dtype=float)
            dxf = np.diff(np.concatenate([[0], sf]) / sf[-1])
            xf = np.array(df[m], dtype=float)
            xf = np.concatenate([xf[0][None] * 0, xf])
            aoc_vals.extend((-np.trapz(xf, dx=dxf[:, None], axis=0)).tolist())

        auc_vals, aoc_vals = np.array(auc_vals), np.array(aoc_vals)
        metric_stats[m] = {
            "AUC": {
                "n": int(auc_vals.size),
                "mean": float(np.mean(auc_vals)),
                "std": float(np.std(auc_vals)),
                "sem": float(np.std(auc_vals) / np.sqrt(max(1, auc_vals.size))),
                "negative_ratio": float(np.mean(auc_vals < 0)),
            },
            "AOC": {
                "n": int(aoc_vals.size),
                "mean": float(np.mean(aoc_vals)),
                "std": float(np.std(aoc_vals)),
                "sem": float(np.std(aoc_vals) / np.sqrt(max(1, aoc_vals.size))),
                "negative_ratio": float(np.mean(aoc_vals < 0)),
            },
        }

    report = {
        "meta": {
            "results_dir": str(out_dir),
            "data_root": str(base),
            "eval_split": split,
            "eval_samples": len(ds),
            "expected_samples": expected_samples,
            "strict_min_samples": strict_min_samples,
        },
        "dataset_integrity": {
            "stem_match_ratio": stem_match_count / max(len(ds.samples), 1),
            "canonical_id_match_ratio": canon_match_count / max(len(ds.samples), 1),
            "first_20_pair_examples": pair_examples,
            "image_shape_ok": img_shape_ok,
            "image_dtype_float32": img_dtype_ok,
            "mask_shape_ok": mask_shape_ok,
            "mask_dtype_long": mask_dtype_ok,
            "mask_unique_values": sorted(unique_vals),
            "mask_binary_01": set(unique_vals).issubset({0, 1}) and len(unique_vals) >= 2,
            "flood_zero_samples": int(np.sum(np.array(flood_pixels) == 0)),
            "flood_zero_ratio": float(np.mean(np.array(flood_pixels) == 0)),
            "image_stats": {
                "min_of_mins": float(np.min([s[0] for s in img_stats])),
                "max_of_maxs": float(np.max([s[1] for s in img_stats])),
                "mean_of_means": float(np.mean([s[2] for s in img_stats])),
            },
        },
        "result_integrity": {
            "flip_files": len(flip_files),
            "insertion_files": len(ins_files),
            "expected_conv_layers": len(expected_layers),
            "missing_flip_layers": missing_flip,
            "missing_insertion_layers": missing_ins,
            "stale_flip_layers_(n<expected_samples)": stale_flip,
            "stale_insertion_layers_(n<expected_samples)": stale_ins,
            "strict_layers_(both_n>=strict_min_samples)": strict_layers,
        },
        "metric_integrity": {
            "definition": {
                "delta_in_code": "perturbed - original logits",
                "insertion_baseline": "x - x[0]",
                "flipping_aoc_sign": "-trapz",
            },
            "per_method_stats_on_strict_layers": metric_stats,
        },
    }

    out_json = out_dir / f"audit_{result_tag}.json"
    out_md = out_dir / f"audit_{result_tag}.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        f"# Audit: {result_tag}",
        "",
        f"- Eval samples: **{report['meta']['eval_samples']}**",
        f"- Flip files: **{report['result_integrity']['flip_files']}**",
        f"- Insertion files: **{report['result_integrity']['insertion_files']}**",
        f"- Expected conv layers: **{report['result_integrity']['expected_conv_layers']}**",
        "",
        "## Dataset integrity",
        f"- stem_match_ratio: **{report['dataset_integrity']['stem_match_ratio']:.3f}**",
        f"- canonical_id_match_ratio: **{report['dataset_integrity']['canonical_id_match_ratio']:.3f}**",
        f"- mask values: **{report['dataset_integrity']['mask_unique_values']}**",
        f"- zero-flood ratio: **{report['dataset_integrity']['flood_zero_ratio']:.3f}**",
        "",
        "## Result integrity",
        f"- missing flip layers: **{len(report['result_integrity']['missing_flip_layers'])}**",
        f"- missing insertion layers: **{len(report['result_integrity']['missing_insertion_layers'])}**",
        f"- stale flip layers (n<{expected_samples}): **{len(report['result_integrity']['stale_flip_layers_(n<expected_samples)'])}**",
        f"- stale insertion layers (n<{expected_samples}): **{len(report['result_integrity']['stale_insertion_layers_(n<expected_samples)'])}**",
        f"- strict layers (both n>={strict_min_samples}): **{len(report['result_integrity']['strict_layers_(both_n>=strict_min_samples)'])}**",
        "",
        "## Metric integrity",
        "- Negative AUC/AOC can occur for signed-logit deltas; values are reported as diagnostic.",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print(f"WROTE {out_json}")
    print(f"WROTE {out_md}")
    print(f"strict_layers={len(strict_layers)} stale_flip={len(stale_flip)} stale_ins={len(stale_ins)}")


if __name__ == "__main__":
    main()
