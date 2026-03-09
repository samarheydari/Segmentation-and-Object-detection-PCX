import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.flood_dataset import FloodDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ignore_label padding for FloodDataset crops. "
            "Saves overlays that highlight ignored pixels and reports counts "
            "inside a central box to ensure padding stays in the border."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/General_Flood_v3",
        help="Path to the Flood dataset root (folder that contains General_Flood_v3).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to pick samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="media/ignore_visualizations/center_checks",
        help="Directory where overlay images + summary JSON will be saved.",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.5,
        help="Size of the central box (fraction of width/height, 0-1).",
    )
    parser.add_argument(
        "--no-multi-scale",
        action="store_true",
        help="Disable multi-scale augmentation when sampling.",
    )
    parser.add_argument(
        "--no-random-flip",
        action="store_true",
        help="Disable random horizontal flips when sampling.",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=None,
        help="Override ignore_label sentinel. Defaults to the dataset config (-1).",
    )
    return parser.parse_args()


def _ensure_three_channels(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {rgb.shape}")
    if rgb.shape[0] == 3:
        return rgb
    if rgb.shape[0] > 3:
        return rgb[:3]
    # single-channel -> replicate
    return rgb.expand(3, -1, -1)


def _pick_indices(n_samples: int, count: int, seed: int) -> List[int]:
    count = min(count, n_samples)
    rng = np.random.default_rng(seed)
    if count == 0:
        return []
    if count >= n_samples:
        return list(range(n_samples))
    return rng.choice(n_samples, size=count, replace=False).tolist()


def _center_box(h: int, w: int, fraction: float) -> torch.Tensor:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction == 0:
        return torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    half_h = int(round(0.5 * fraction * h))
    half_w = int(round(0.5 * fraction * w))
    cy = h // 2
    cx = w // 2
    top = max(0, cy - half_h)
    bottom = min(h, cy + half_h)
    left = max(0, cx - half_w)
    right = min(w, cx + half_w)
    return torch.tensor([left, top, right, bottom], dtype=torch.int64)


def main() -> None:
    args = _parse_args()

    dataset = FloodDataset(
        root_dir=args.root,
        split=args.split,
        multi_scale=not args.no_multi_scale,
        flip=not args.no_random_flip,
        ignore_label=args.ignore_label,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = _pick_indices(len(dataset), args.count, args.seed)
    if not indices:
        print("Dataset is empty or --count=0; no visualizations generated.")
        return

    stats = []
    center_fraction = float(np.clip(args.center_fraction, 0.0, 1.0))
    for rank, idx in enumerate(indices):
        image, label = dataset[idx]
        ignore_mask = (label == dataset.ignore_label)
        total_ignore = int(ignore_mask.sum().item())

        h, w = ignore_mask.shape
        center_box = _center_box(h, w, center_fraction)
        left, top, right, bottom = center_box.tolist()
        center_slice = ignore_mask[top:bottom, left:right]
        center_ignore = int(center_slice.sum().item())

        name = dataset.files[idx]["name"]
        overlay = dataset.reverse_augmentation(image).to(torch.uint8)
        overlay = _ensure_three_channels(overlay)
        overlay = draw_segmentation_masks(
            overlay,
            ignore_mask.unsqueeze(0).bool(),
            colors=["red"],
            alpha=0.6,
        )
        if (right - left) > 0 and (bottom - top) > 0:
            box = center_box.unsqueeze(0)
            overlay = draw_bounding_boxes(
                overlay,
                box,
                colors=["yellow"],
                width=2,
            )
        save_path = output_dir / f"{rank:02d}_{name}.png"
        save_image = overlay.permute(1, 2, 0).cpu().numpy()
        save_image = np.clip(save_image, 0, 255).astype(np.uint8)
        Image.fromarray(save_image).save(save_path)

        stats.append(
            {
                "index": idx,
                "name": name,
                "image_path": str(dataset.files[idx]["img"]),
                "mask_path": str(dataset.files[idx]["label"]),
                "total_pixels": int(h * w),
                "ignored_pixels": total_ignore,
                "center_fraction": center_fraction,
                "center_box": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
                "ignored_center_pixels": center_ignore,
                "overlay_path": str(save_path),
            }
        )
        print(
            f"[{rank+1}/{len(indices)}] {name}: ignored={total_ignore} "
            f"center_ignored={center_ignore} (box={left},{top},{right},{bottom})"
        )

    summary_path = output_dir / "ignore_stats.json"
    with summary_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {len(stats)} overlays + stats to {output_dir}")


if __name__ == "__main__":
    main()
