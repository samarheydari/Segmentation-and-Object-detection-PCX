"""Small, model-agnostic helpers used by the final-results notebooks."""

from __future__ import annotations

import os
import re
import shutil
import importlib
import sys
import base64
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "zplus": "LRP-z+",
    "gamma": "LRP-gamma",
    "epsilon": "LRP-epsilon",
    "eps": "LRP-epsilon",
    "gradcam": "GradCAM",
    "gradient": "Gradient",
    "activation": "Activation",
    "random": "Random",
}

PIDNET_POST_MERGE_LAYERS = [
    "dfm.conv_p.0",
    "dfm.conv_i.0",
    "final_layer.conv1",
    "final_layer.conv2",
]

PIDNET_METHODS = [
    "LRP-zplus", "LRP-gamma", "LRP-eps", "GradCAM",
    "Gradient", "activation", "random",
]


def find_repo_root(start: Path | None = None) -> Path:
    """Find the checkout root, whether the notebook runs from root or notebooks/."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "paper" / "12-supp" / "code").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repository root (expected paper/12-supp/code).")


def resolve_results_root(repo_root: Path) -> Path:
    """Use an explicit env var, committed results, or the original experiment location."""
    candidates = []
    if os.environ.get("PAPER_RESULTS_DIR"):
        candidates.append(Path(os.environ["PAPER_RESULTS_DIR"]).expanduser())
    candidates.extend([
        repo_root / "paper" / "12-supp" / "code" / "results",
        Path("/home/heydari/paper/12-supp/code/results"),
    ])
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "Results were not found. Set PAPER_RESULTS_DIR to the directory containing "
        "global_class_concepts/ and instance_perturbation/."
    )


def artifact_inventory(root: Path) -> pd.DataFrame:
    files = [p for p in root.rglob("*") if p.is_file()]
    counts = Counter((p.suffix.lower() or "[none]") for p in files)
    sizes = Counter()
    for p in files:
        sizes[p.suffix.lower() or "[none]"] += p.stat().st_size
    return pd.DataFrame([
        {"format": ext, "files": counts[ext], "size_MB": sizes[ext] / 2**20}
        for ext in sorted(counts)
    ]).sort_values("files", ascending=False, ignore_index=True)


def concept_inventory(results_root: Path, dataset: str, model: str) -> pd.DataFrame:
    base = results_root / "global_class_concepts" / dataset / model
    rows = []
    if not base.is_dir():
        return pd.DataFrame(columns=["initialization", "class", "layer_files"])
    pattern = re.compile(r"_class_(\d+)\.pth$")
    for init_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        counts = Counter()
        for path in init_dir.glob("*.pth"):
            match = pattern.search(path.name)
            if match and not path.name.startswith("class_"):
                counts[int(match.group(1))] += 1
        for class_id, count in sorted(counts.items()):
            rows.append({"initialization": init_dir.name, "class": class_id, "layer_files": count})
    return pd.DataFrame(rows)


def _torch_load(path: Path):
    import torch
    def load():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0 has no weights_only argument
            return torch.load(path, map_location="cpu")

    try:
        return load()
    except ModuleNotFoundError as exc:
        # NumPy 2 pickles reference numpy._core; NumPy 1.x exposes numpy.core.
        # The saved tensors contain ordinary arrays, so these aliases are safe.
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
        return load()


def _method_name(name: str) -> str:
    key = name.lower().replace("lrp-", "").replace("lrp_", "")
    return METHOD_LABELS.get(key, name)


def _curve_score(values, steps, insertion: bool) -> tuple[float, float, int] | None:
    """Return mean AUC/AOC, standard error, and sample count for one layer/method."""
    x = np.asarray(values, dtype=float)
    s = np.asarray(steps, dtype=float).reshape(-1)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not len(s) or s[-1] == 0:
        return None
    if insertion:
        axis = s / s[-1]
        x = x - x[0:1]
    else:
        axis = np.concatenate([[0.0], s / s[-1]])
        x = np.concatenate([np.zeros_like(x[0:1]), x], axis=0)
    length = min(len(axis), x.shape[0])
    if length < 2:
        return None
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    sample_scores = integrate(x[:length], x=axis[:length], axis=0)
    sample_scores = sample_scores[np.isfinite(sample_scores)]
    if not insertion:
        sample_scores = -sample_scores
    if not len(sample_scores):
        return None
    sem = sample_scores.std(ddof=1) / np.sqrt(len(sample_scores)) if len(sample_scores) > 1 else 0.0
    return float(sample_scores.mean()), float(sem), int(len(sample_scores))


def load_perturbation_summary(
    results_root: Path, dataset: str, model: str, run: str, class_tag: str | None = None
) -> pd.DataFrame:
    """Load every compatible deletion/insertion tensor and reduce it to paper metrics."""
    base = results_root / "instance_perturbation" / dataset / model / run
    data_dir = base / "data"
    search_dirs = [data_dir, base] if data_dir != base else [base]
    paths = []
    for folder in search_dirs:
        if folder.is_dir():
            paths.extend(folder.glob("instance_perturbation_*.pth"))
    rows = []
    for path in sorted(set(paths)):
        stem = path.stem
        insertion = stem.endswith("_insertion")
        core = stem.removeprefix("instance_perturbation_").removesuffix("_insertion")
        detected_tag = None
        match = re.search(r"_(c(?:\d+)(?:-\d+)*)$", core)
        if match:
            detected_tag = match.group(1)
            layer = core[: match.start()]
        else:
            layer = core
        if class_tag is not None and detected_tag not in (None, class_tag):
            continue
        try:
            payload = _torch_load(path)
        except Exception as exc:
            rows.append({
                "layer": layer, "mode": "load_error", "method": type(exc).__name__,
                "error": f"{type(exc).__name__}: {exc}", "source": path,
            })
            continue
        if not isinstance(payload, dict) or "steps" not in payload:
            continue
        for method, values in payload.items():
            if method == "steps":
                continue
            score = _curve_score(values, payload["steps"], insertion)
            if score:
                mean, sem, n = score
                rows.append({
                    "layer": layer, "mode": "Insertion AUC" if insertion else "Deletion AOC",
                    "method": _method_name(str(method)), "score": mean, "sem": sem,
                    "samples": n, "class_tag": detected_tag or "all", "source": path,
                })
    frame = pd.DataFrame(rows)
    frame.attrs["searched"] = [str(p) for p in search_dirs]
    frame.attrs["candidate_files"] = len(set(paths))
    frame.attrs["load_errors"] = int(sum(row.get("mode") == "load_error" for row in rows))
    errors = [row.get("error") for row in rows if row.get("error")]
    frame.attrs["first_load_error"] = errors[0] if errors else ""
    return frame


def perturbation_diagnostics(
    results_root: Path, dataset: str, model: str, run: str, summary: pd.DataFrame
) -> pd.DataFrame:
    """A compact check shown before plotting, especially useful on a new machine."""
    base = results_root / "instance_perturbation" / dataset / model
    selected = base / run
    available = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    candidates = list(selected.glob("instance_perturbation_*.pth"))
    candidates += list((selected / "data").glob("instance_perturbation_*.pth"))
    usable = 0 if summary.empty or "score" not in summary else int(summary["source"].nunique())
    return pd.DataFrame([{
        "selected_directory": str(selected),
        "directory_exists": selected.is_dir(),
        "available_runs": ", ".join(available) or "none",
        "candidate_pth_files": len(set(candidates)),
        "successfully_decoded_files": usable,
        "load_errors": summary.attrs.get("load_errors", 0),
        "first_load_error": summary.attrs.get("first_load_error", ""),
    }])


def score_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "score" not in summary:
        return pd.DataFrame()
    return (summary.groupby(["mode", "method"], as_index=False)
            .agg(mean_score=("score", "mean"), layers=("layer", "nunique"), samples=("samples", "max"))
            .sort_values(["mode", "mean_score"], ascending=[True, False], ignore_index=True))


def plot_layer_scores(summary: pd.DataFrame, title: str):
    modes = [m for m in ("Deletion AOC", "Insertion AUC") if m in set(summary.get("mode", []))]
    if not modes:
        searched = summary.attrs.get("searched", [])
        candidates = summary.attrs.get("candidate_files", 0)
        errors = summary.attrs.get("load_errors", 0)
        first_error = summary.attrs.get("first_load_error", "")
        print("No compatible perturbation tensors were found for this selection.")
        print(f"Candidate .pth files: {candidates}; load errors: {errors}")
        if first_error:
            print(f"First load error: {first_error}")
        if searched:
            print("Searched:\n  " + "\n  ".join(searched))
        print("Check RESULTS_ROOT/DATASET/MODEL/RUN above, then restart the kernel and Run All.")
        return None
    fig, axes = plt.subplots(len(modes), 1, figsize=(14, 4.2 * len(modes)), squeeze=False)
    for ax, mode in zip(axes[:, 0], modes):
        part = summary[summary["mode"] == mode]
        layers = list(dict.fromkeys(part["layer"]))
        xpos = {layer: i for i, layer in enumerate(layers)}
        for method, values in part.groupby("method", sort=False):
            values = values.sort_values("layer", key=lambda col: col.map(xpos))
            x = [xpos[v] for v in values["layer"]]
            ax.errorbar(x, values["score"], yerr=values["sem"], marker=".", linewidth=1, label=method)
        ax.set(title=mode, ylabel="score", xticks=range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=7)
        ax.grid(alpha=.2)
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle(title, y=1.01, fontsize=14)
    fig.tight_layout()
    return fig


def plot_pidnet_post_merge(results_root: Path, run: str = "logits", insertion: bool = False):
    """Reproduce the paper's exact four-layer PIDNet summary figure."""
    base = results_root / "instance_perturbation" / "flood" / "pidnet" / run
    data_dir = base / "data" if (base / "data").is_dir() else base
    suffix = "_insertion.pth" if insertion else ".pth"
    payloads = {
        layer: _torch_load(data_dir / f"instance_perturbation_{layer}{suffix}")
        for layer in PIDNET_POST_MERGE_LAYERS
    }
    labels = {
        "LRP-zplus": "LRP-z+", "LRP-gamma": "LRP-gamma",
        "LRP-eps": "LRP-eps", "GradCAM": "GradCAM",
        "Gradient": "Gradient", "activation": "activation", "random": "random",
    }
    integrate = getattr(np, "trapezoid", None) or np.trapz
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    xpos = np.arange(len(PIDNET_POST_MERGE_LAYERS))
    for index, method in enumerate(PIDNET_METHODS):
        per_layer = []
        for layer in PIDNET_POST_MERGE_LAYERS:
            payload = payloads[layer]
            steps = np.asarray(payload["steps"], dtype=float)
            values = np.asarray(payload[method], dtype=float)
            if insertion:
                axis = steps / steps[-1]
                values = values - values[0:1]
            else:
                axis = np.concatenate([[0.0], steps / steps[-1]])
                values = np.concatenate([np.zeros_like(values[0:1]), values], axis=0)
            scores = integrate(values, x=axis, axis=0)
            if not insertion:
                scores = -scores
            scores = scores[np.isfinite(scores)]
            scores = scores[scores != 0]
            per_layer.append(scores)
        means = np.asarray([values.mean() for values in per_layer])
        errors = np.asarray([
            values.std(ddof=0) / np.sqrt(len(values)) for values in per_layer
        ])
        line, = ax.plot(
            xpos, means, ".-", linewidth=1.8, markersize=7,
            label=f"{labels[method]} ({means.mean():.3f})", zorder=10 + index,
        )
        ax.fill_between(
            xpos, means - errors, means + errors,
            color=line.get_color(), alpha=0.18, zorder=index,
        )
    metric = "AUC insertion" if insertion else "AOC deletion"
    ax.set_title(f"PIDNet flood (with background) post-merge ({run}) - {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Post-merge layers")
    ax.set_xticks(xpos)
    ax.set_xticklabels(PIDNET_POST_MERGE_LAYERS, rotation=20)
    ax.set_xlim(-0.15, len(PIDNET_POST_MERGE_LAYERS) - 0.85)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    return fig


def final_figures(root: Path, keywords=(), limit: int = 24) -> list[Path]:
    """Choose summary figures first, then other rendered artifacts."""
    if not root.is_dir():
        return []
    keys = tuple(k.lower() for k in keywords)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}]
    if keys:
        paths = [p for p in paths if any(k in str(p).lower() for k in keys)]
    def priority(path):
        name = path.name.lower()
        return (0 if any(k in name for k in ("summary", "concept_perturbation", "complexity")) else 1, str(path))
    return sorted(paths, key=priority)[:limit]


def display_gallery(paths: list[Path], columns: int = 3):
    from IPython.display import Image, Markdown, display
    if not paths:
        display(Markdown("*No rendered figures found for this selection.*"))
        return
    for path in paths:
        display(Markdown(f"**{path.name}**  \n`{path}`"))
        if path.suffix.lower() == ".pdf":
            display(Markdown(f"[Open PDF]({path.as_uri()})"))
        else:
            display(Image(filename=str(path), width=900 // max(1, columns - 1)))


def display_pdf_plots(paths: list[Path], width: int = 950, height: int = 560):
    """Embed local PDF plots inline without relying on browser file:// access."""
    from IPython.display import HTML, Markdown, display
    for path in paths:
        path = Path(path)
        display(Markdown(f"### {path.stem.replace('_', ' ').title()}"))
        if not path.is_file():
            display(Markdown(f"**Missing:** `{path}`"))
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        display(HTML(
            f'<object data="data:application/pdf;base64,{encoded}" '
            f'type="application/pdf" width="{width}" height="{height}">'
            f'<p>PDF preview is unavailable. <a download="{path.name}" '
            f'href="data:application/pdf;base64,{encoded}">Download {path.name}</a>.</p>'
            f'</object>'
        ))


def export_figure_artifacts(paths: list[Path], output_dir: Path) -> pd.DataFrame:
    """Copy selected final figures into the notebook's portable output bundle."""
    destination = output_dir / "source_figures"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, source in enumerate(paths, start=1):
        target = destination / f"{index:03d}_{source.name}"
        shutil.copy2(source, target)
        rows.append({"source": str(source), "exported_to": str(target)})
    manifest = pd.DataFrame(rows, columns=["source", "exported_to"])
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    return manifest
