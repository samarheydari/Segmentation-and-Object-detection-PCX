import argparse
import os
import re
import glob
import csv

import numpy as np
import torch
from matplotlib import pyplot as plt


def normalize_curve(y):
    y = np.asarray(y, dtype=float)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if np.isclose(ymax, ymin):
        return np.zeros_like(y)
    return (y - ymin) / (ymax - ymin)


def normalize_x(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if np.isclose(xmax, xmin):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def list_pairs(input_dir):
    files = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, "*.pth"))]
    base = [f for f in files if not f.endswith("_insertion.pth")]
    out = []
    for b in base:
        i = b[:-4] + "_insertion.pth"
        if i in files:
            out.append((os.path.join(input_dir, b), os.path.join(input_dir, i), b[:-4]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot AUC/AOC summary for skip-related layers.")
    parser.add_argument(
        "--input_dir",
        default="/home/heydari/paper/12-supp/code/experiments/results/instance_perturbation/person_car/yolov6s6/data",
    )
    parser.add_argument(
        "--layer_pattern",
        default=r"(rbr_1x1|Bifusion)",
        help="Regex to keep layers considered skip-related.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/heydari/paper/LCRP/results/plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pairs = list_pairs(args.input_dir)
    layer_re = re.compile(args.layer_pattern)
    pairs = [p for p in pairs if layer_re.search(p[2])]
    if not pairs:
        raise RuntimeError(f"No paired files matched pattern: {args.layer_pattern}")

    method_scores = {}
    layer_rows = []

    for del_path, ins_path, layer_name in pairs:
        d_del = torch.load(del_path, map_location="cpu", weights_only=False)
        d_ins = torch.load(ins_path, map_location="cpu", weights_only=False)
        methods = sorted([k for k in d_del.keys() if k != "steps" and k in d_ins])

        x_del = normalize_x(np.asarray(d_del["steps"], dtype=float))
        x_ins = normalize_x(np.asarray(d_ins["steps"], dtype=float))

        for m in methods:
            y_del = np.asarray(d_del[m], dtype=float).mean(axis=1)
            y_ins = np.asarray(d_ins[m], dtype=float).mean(axis=1)

            y_del_norm = normalize_curve(y_del)
            y_ins_norm = normalize_curve(y_ins)

            auc = float(np.trapezoid(y_ins_norm, x_ins))
            aoc = float(np.trapezoid(1.0 - y_del_norm, x_del))

            method_scores.setdefault(m, {"auc": [], "aoc": []})
            method_scores[m]["auc"].append(auc)
            method_scores[m]["aoc"].append(aoc)
            layer_rows.append((layer_name, m, auc, aoc))

    methods = sorted(method_scores.keys())
    auc_mean = [float(np.mean(method_scores[m]["auc"])) for m in methods]
    auc_std = [float(np.std(method_scores[m]["auc"])) for m in methods]
    aoc_mean = [float(np.mean(method_scores[m]["aoc"])) for m in methods]
    aoc_std = [float(np.std(method_scores[m]["aoc"])) for m in methods]

    x = np.arange(len(methods))
    w = 0.38
    plt.figure(figsize=(11, 5), dpi=220)
    plt.bar(x - w / 2, auc_mean, yerr=auc_std, width=w, label="AUC (Insertion)", capsize=3)
    plt.bar(x + w / 2, aoc_mean, yerr=aoc_std, width=w, label="AOC (Deletion)", capsize=3)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Score (0-1)")
    plt.title(f"YOLOv6s6 Skip-Related Layers: AUC/AOC Summary (n={len(pairs)} layers)")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(args.output_dir, "yolov6s6_skip_auc_aoc_summary.png")
    plt.savefig(fig_path)
    plt.close()

    summary_csv = os.path.join(args.output_dir, "yolov6s6_skip_auc_aoc_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "auc_mean", "auc_std", "aoc_mean", "aoc_std", "n_layers"])
        for i, m in enumerate(methods):
            writer.writerow([m, auc_mean[i], auc_std[i], aoc_mean[i], aoc_std[i], len(pairs)])

    layer_csv = os.path.join(args.output_dir, "yolov6s6_skip_auc_aoc_per_layer.csv")
    with open(layer_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "method", "auc", "aoc"])
        writer.writerows(layer_rows)

    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {layer_csv}")
    print(f"Matched paired layers: {len(pairs)}")


if __name__ == "__main__":
    main()
