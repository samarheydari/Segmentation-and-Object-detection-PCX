import argparse
import glob
import os
import re

import numpy as np
import torch
from matplotlib import pyplot as plt


LABELS = {
    "LRP-zplus": r"LRP-z$^+$",
    "LRP-gamma": r"LRP-$\gamma$",
    "LRP-eps": r"LRP-$\varepsilon$",
    "GradCAM": "GradCAM",
    "Gradient": "gradient",
    "activation": "activation",
    "random": "random",
}


def layer_pairs(input_dir):
    files = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, "*.pth"))]
    base = [f for f in files if not f.endswith("_insertion.pth")]
    pairs = []
    for b in base:
        i = b[:-4] + "_insertion.pth"
        if i in files:
            pairs.append((b[:-4], os.path.join(input_dir, b), os.path.join(input_dir, i)))
    return sorted(pairs, key=lambda x: x[0])


def integrate_samples(d, method, insertion):
    steps = np.asarray(d["steps"], dtype=float)
    if insertion:
        stepsn = steps / steps[-1]
    else:
        stepsn = np.concatenate([[0], steps]) / steps[-1]
    dx = np.diff(stepsn)

    x = np.asarray(d[method], dtype=float)
    if insertion:
        x = x - x[0, :][None]
    else:
        x = np.concatenate([x[0][None] * 0, x], axis=0)

    vals = np.trapezoid(x, dx=dx[:, None], axis=0)
    vals = vals[vals != 0]
    if vals.size == 0:
        vals = np.trapezoid(x, dx=dx[:, None], axis=0)
    if not insertion:
        vals = -vals
    return vals


def integrate_samples_normalized(d, method, insertion):
    steps = np.asarray(d["steps"], dtype=float)
    stepsn = steps / steps[-1]
    x = np.asarray(d[method], dtype=float)

    if insertion:
        yn = []
        for s in range(x.shape[1]):
            y = x[:, s]
            ymin, ymax = float(np.min(y)), float(np.max(y))
            if np.isclose(ymax, ymin):
                y_norm = np.zeros_like(y)
            else:
                y_norm = (y - ymin) / (ymax - ymin)
            yn.append(float(np.trapezoid(y_norm, x=stepsn)))
        return np.asarray(yn, dtype=float)

    # Deletion mode: prepend zero baseline and integrate 1 - normalized(y)
    stepsd = np.concatenate([[0], steps]) / steps[-1]
    yn = []
    for s in range(x.shape[1]):
        y = np.concatenate([[0.0], x[:, s]])
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if np.isclose(ymax, ymin):
            y_norm = np.zeros_like(y)
        else:
            y_norm = (y - ymin) / (ymax - ymin)
        yn.append(float(np.trapezoid(1.0 - y_norm, x=stepsd)))
    return np.asarray(yn, dtype=float)


def plot_mode(pairs, include_re, insertion, out_pdf, title, normalize_curve, summary_like_auc):
    methods = None
    rows = []
    for layer, d_path, i_path in pairs:
        if not include_re.search(layer):
            continue
        d = torch.load(i_path if insertion else d_path, map_location="cpu", weights_only=False)
        if methods is None:
            methods = [m for m in d.keys() if m != "steps"]
        rows.append((layer, d))

    if not rows:
        raise RuntimeError("No layers matched filter.")

    plt.figure(figsize=(5, 3), dpi=300)
    layers = [r[0] for r in rows]
    x_idx = np.arange(len(layers))

    for k, m in enumerate(methods):
        means = []
        sems = []
        for _, d in rows:
            if summary_like_auc and insertion:
                x = np.asarray(d["steps"], dtype=float)
                x = (x - x.min()) / (x.max() - x.min())
                y = np.asarray(d[m], dtype=float).mean(axis=1)
                ymin, ymax = float(np.min(y)), float(np.max(y))
                if np.isclose(ymax, ymin):
                    val = 0.0
                else:
                    y = (y - ymin) / (ymax - ymin)
                    val = float(np.trapezoid(y, x=x))
                vals = np.asarray([val], dtype=float)
            elif normalize_curve:
                vals = integrate_samples_normalized(d, m, insertion=insertion)
            else:
                vals = integrate_samples(d, m, insertion=insertion)
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals) / np.sqrt(max(len(vals), 1))))

        means = np.asarray(means)
        sems = np.asarray(sems)
        err = float(np.sqrt(np.sum(sems ** 2)) / len(sems))
        label = LABELS.get(m, m)
        p, = plt.plot(
            x_idx,
            means,
            ".-",
            label=f"{label} (${means.mean():.4f}\\pm{err:.4f}$)",
            zorder=k / max(len(methods), 1),
        )
        plt.fill_between(
            x_idx,
            means - sems,
            means + sems,
            alpha=0.2,
            zorder=k - len(methods),
            color=p.get_color(),
        )

    plt.legend(fontsize="small")
    plt.title(title)
    plt.ylabel("AUC concept insertion" if insertion else "AOC concept flipping")
    plt.xlabel("convolutional layer")
    plt.xticks(x_idx, x_idx, rotation=90)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot AUC/AOC in concept_perturbation style.")
    parser.add_argument(
        "--input_dir",
        default="/home/heydari/paper/12-supp/code/experiments/results/instance_perturbation/person_car/yolov6s6/data",
    )
    parser.add_argument(
        "--layer_pattern",
        default=r"^(?!.*(rbr_1x1|Bifusion)).*$",
        help="Regex for included layers. Default: all non-skip layers.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/heydari/FHHI-XAI/plots/non_skip_style",
    )
    parser.add_argument("--title", default="YOLOv6 - person_car")
    parser.add_argument(
        "--normalize_curve",
        action="store_true",
        help="Normalize each sample curve to [0,1] before integration.",
    )
    parser.add_argument(
        "--summary_like_auc",
        action="store_true",
        help="Use the same AUC definition as summary CSV (mean curve per layer, then normalize+integrate).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pairs = layer_pairs(args.input_dir)
    include_re = re.compile(args.layer_pattern)

    suffix = "_norm" if args.normalize_curve else ""
    out_ins = os.path.join(args.output_dir, f"concept_perturbation_yolov6s6_person_car_non_skip_insertion_style{suffix}.pdf")
    out_del = os.path.join(args.output_dir, f"concept_perturbation_yolov6s6_person_car_non_skip_deletion_style{suffix}.pdf")

    plot_mode(
        pairs,
        include_re,
        insertion=True,
        out_pdf=out_ins,
        title=args.title,
        normalize_curve=args.normalize_curve,
        summary_like_auc=args.summary_like_auc,
    )
    plot_mode(
        pairs,
        include_re,
        insertion=False,
        out_pdf=out_del,
        title=args.title,
        normalize_curve=args.normalize_curve,
        summary_like_auc=args.summary_like_auc,
    )

    print(f"Saved: {out_ins}")
    print(f"Saved: {out_del}")


if __name__ == "__main__":
    main()
