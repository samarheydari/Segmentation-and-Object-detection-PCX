import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import transforms


def flood_train(**kwargs):
    # Keep API parity with other dataset factories used by get_dataset().
    return FloodSegmentation(split="train", **kwargs)


def flood_test(**kwargs):
    # Prefer val for evaluation if available, otherwise fallback to test.
    # This avoids script edits and keeps one stable `dataset_name="flood"` entry.
    root = kwargs.get("root", _get_data_root())
    base = _resolve_dataset_root(root)
    split = "val" if (base / "RGB" / "val" / "JPEG").is_dir() else "test"
    return FloodSegmentation(split=split, **kwargs)


def _get_data_root():
    # Central path override via environment variable so experiments can be moved
    # across machines without changing code.
    return os.getenv("FLOOD_DATA_ROOT", "/home/heydari/FHHI-XAI/data/flood_segmentation")


def _resolve_dataset_root(root: str) -> Path:
    # Support both layouts:
    # 1) <root>/General_Flood_v3/...
    # 2) <root>/...
    root_path = Path(root)
    nested = root_path / "General_Flood_v3"
    return nested if nested.is_dir() else root_path


class FloodSegmentation(torch.utils.data.Dataset):
    def __init__(self, split="train", root=None, **kwargs):
        # Do not require script-level changes:
        # default root comes from FLOOD_DATA_ROOT env var.
        self.root = _resolve_dataset_root(root or _get_data_root())
        self.split = split

        self.image_dir = self.root / "RGB" / split / "JPEG"
        self.mask_dir = self.root / "annotations" / split / "JPEG"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Flood split '{split}' not found. Expected:\n"
                f"  images: {self.image_dir}\n"
                f"  masks: {self.mask_dir}"
            )

        # Precompute aligned (image, mask) pairs once for deterministic indexing.
        self.samples = self._scan_pairs()
        self.class_names = ["background", "flood"]
        self.reverse_normalization = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(std=[1, 1, 1], mean=[-0.485, -0.456, -0.406]),
        ])

    def _scan_pairs(self):
        exts = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
        # Sorting provides stable order, which is important because
        # instance_perturbation uses stored sample indices from concept files.
        images = [p for p in self.image_dir.iterdir() if p.suffix in exts]
        masks = [p for p in self.mask_dir.iterdir() if p.suffix in exts]
        images.sort()
        masks.sort()

        # Primary strategy: pair by identical filename stem.
        mask_map = {m.stem: m for m in masks}
        samples = []
        for img in images:
            mask = mask_map.get(img.stem)
            if mask is not None:
                samples.append((img, mask))

        # Fallback strategy if stems differ but counts match.
        if not samples and len(images) == len(masks) and len(images) > 0:
            samples = list(zip(images, masks))

        if not samples:
            raise RuntimeError(
                f"No image/mask pairs found in:\n  {self.image_dir}\n  {self.mask_dir}"
            )
        return samples

    @staticmethod
    def _mask_to_label(mask_np: np.ndarray) -> np.ndarray:
        if mask_np.ndim == 2:
            # Binary mask: >0 is flood, 0 is background.
            return (mask_np > 0).astype(np.uint8)
        # RGB mask: any non-zero pixel is flood.
        return (mask_np.sum(axis=2) > 0).astype(np.uint8)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Match common segmentation preprocessing used in this codebase.
        x = transforms.ToTensor()(img)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        # Return class-index mask expected by perturbation math:
        # targets[i] == class_id
        y = torch.from_numpy(self._mask_to_label(mask)).long()
        return x, y

    def __len__(self):
        return len(self.samples)
