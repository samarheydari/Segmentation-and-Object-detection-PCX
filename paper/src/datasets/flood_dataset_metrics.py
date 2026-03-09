import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.datasets.base_dataset import BaseDataset


class FloodDataset(BaseDataset):
    """
    Flood segmentation dataset with the same preprocessing and ignore-label
    semantics as general_flood_v3.

    - Only exact colors in `color_list` are considered valid classes.
    - All other pixels are set to `ignore_label` and will be ignored by any
      metric / analysis that uses `ignore_label` (e.g., IoU via get_confusion_matrix).

    By default it will look for the official list file
    (list/General_Flood_v3/{split}.lst) if given, otherwise it falls back
    to scanning image/annotation folders and pairing by filename.
    """

    class_names = ["background", "flood"]
    # Valid label colors; everything else becomes ignore_label
    color_list = [[0, 0, 0], [1, 1, 1]]

    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = None,
        # backward-compatible alias for some notebooks
        root_dir: Optional[str] = None,
        # optional transform argument kept for API compatibility (not used here)
        transform=None,
        num_classes: int = 2,
        multi_scale: bool = False,
        flip: bool = False,
        # if None, default to -1 to match general_flood_v3 config
        ignore_label: Optional[int] = None,
        base_size: int = 2048,
        crop_size: Tuple[int, int] = (1280, 720),
        scale_factor: int = 16,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        bd_dilate_size: int = 4,
        return_or_dims: bool = False,
        strict_pairing: bool = False,
        mask_suffix_patterns: Optional[List[str]] = None,
        list_path: Optional[str] = None,
    ):
        # Accept either `root` or `root_dir` for compatibility with examples
        if root is None and root_dir is not None:
            root = root_dir
        if root is None:
            raise ValueError("root or root_dir must be provided to FloodDataset")

        # Use the same default ignore label as the original general_flood_v3
        if ignore_label is None:
            ignore_label = -1
        self.ignore_label = ignore_label  # expose explicitly for PCX / analysis

        # Initialize BaseDataset (normalization, resizing, etc.)
        super(FloodDataset, self).__init__(
            ignore_label, base_size, crop_size, scale_factor, mean, std
        )

        # preserve the base root for list files
        self.base_root = root
        if split is None:
            split = "train"

        # allow dataset to live under root/General_Flood_v3 or directly under root
        self.dataset_root = os.path.join(root, "General_Flood_v3")
        if not os.path.isdir(self.dataset_root):
            self.dataset_root = root

        self.image_dir = os.path.join(self.dataset_root, "RGB", split, "JPEG")
        self.mask_dir = os.path.join(self.dataset_root, "annotations", split, "JPEG")

        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.bd_dilate_size = bd_dilate_size
        self.return_or_dims = return_or_dims

        # filename alignment
        self.strict_pairing = strict_pairing
        self.mask_suffix_patterns = mask_suffix_patterns or [
            r"_mask$",
            r"-mask$",
            r"_label$",
            r"-label$",
            r"_gt$",
            r"-gt$",
            r"_ann$",
            r"-ann$",
            r"Ids$",
            r"Ids_?$",
        ]
        self._mask_suffix_re = re.compile(
            "|".join(self.mask_suffix_patterns), flags=re.IGNORECASE
        )

        # Honor an explicit list file if provided; otherwise rely on directory pairing
        self.list_path = list_path

        # Prefer explicit list files (matches general_flood_v3 evaluation) for deterministic pairing
        if self.list_path is not None:
            self.files = self._files_from_list(self.list_path)
        else:
            self.files = self._scan_and_pair()

        self.class_weights = None

    # --- pairing helpers -----------------------------------------------------
    def _stem_no_ext(self, p: Path) -> str:
        return p.stem.rstrip("_")

    def _norm_mask_stem(self, s: str) -> str:
        s2 = self._mask_suffix_re.sub("", s)
        return s2.rstrip("_")

    def _scan_and_pair(self):
        img_exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
        mask_exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")

        image_files_all = [
            Path(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(img_exts)
        ]
        mask_files_all = [
            Path(self.mask_dir, f)
            for f in os.listdir(self.mask_dir)
            if f.endswith(mask_exts)
        ]
        image_files_all.sort()
        mask_files_all.sort()

        img_groups = {}
        for p in image_files_all:
            k = self._stem_no_ext(p)
            img_groups.setdefault(k, []).append(p)
        for k in list(img_groups.keys()):
            img_groups[k].sort()

        if self.strict_pairing:
            mask_groups = {}
            for p in mask_files_all:
                k = self._stem_no_ext(p)
                mask_groups.setdefault(k, []).append(p)
            for k in list(mask_groups.keys()):
                mask_groups[k].sort()
        else:
            mask_groups = {}
            for p in mask_files_all:
                k = self._norm_mask_stem(self._stem_no_ext(p))
                mask_groups.setdefault(k, []).append(p)
            for k in list(mask_groups.keys()):
                mask_groups[k].sort()

        files = []
        common_keys = [s for s in sorted(img_groups.keys()) if s in mask_groups]
        for k in common_keys:
            imgs = img_groups[k]
            masks = mask_groups[k]
            n_pairs = min(len(imgs), len(masks))
            for i in range(n_pairs):
                img_path = imgs[i]
                mask_path = masks[i]
                name = os.path.splitext(os.path.basename(mask_path))[0]
                files.append(
                    {
                        "img": img_path,
                        "label": mask_path,
                        "name": name,
                    }
                )

        # fallback: 1–1 pairing if counts match exactly
        if (
            not files
            and len(image_files_all) == len(mask_files_all)
            and len(image_files_all) > 0
        ):
            for img_path, mask_path in zip(image_files_all, mask_files_all):
                name = os.path.splitext(os.path.basename(mask_path))[0]
                files.append(
                    {
                        "img": img_path,
                        "label": mask_path,
                        "name": name,
                    }
                )

        assert (
            files
        ), f"No paired images/masks found in {self.image_dir} and {self.mask_dir}"
        return files

    def _files_from_list(self, list_path: str):
        """Reproduce the list-driven pairing used by general_flood_v3 for identical ordering."""
        list_file = Path(list_path)
        if not list_file.is_absolute():
            list_file = Path(self.base_root) / list_path
        if not list_file.exists():
            raise FileNotFoundError(f"List file not found: {list_file}")

        files = []
        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_rel, mask_rel = parts[:2]
                img_path = Path(self.dataset_root) / img_rel
                mask_path = Path(self.dataset_root) / mask_rel
                name = os.path.splitext(os.path.basename(mask_path))[0]
                files.append({"img": img_path, "label": mask_path, "name": name})

        assert files, f"No entries loaded from list file: {list_file}"
        return files

    def __len__(self):
        return len(self.files)

    # --- label <-> color with ignore semantics -------------------------------
    def color2label(self, color_map: np.ndarray) -> np.ndarray:
        """
        Map RGB mask to integer label mask.

        - Start with all pixels = ignore_label
        - For each exact color in self.color_list, set the corresponding class id.
        """
        label = np.ones(color_map.shape[:2], dtype=np.int64) * self.ignore_label
        for i, v in enumerate(self.color_list):
            v = np.array(v, dtype=np.uint8)
            mask = (color_map == v).sum(2) == 3
            label[mask] = i
        return label.astype(np.int64)

    def label2color(self, label: np.ndarray) -> np.ndarray:
        """
        Map integer label mask back to RGB.

        Ignore-label pixels are left as [0,0,0] here; this is only for visualization
        of predictions, not for training.
        """
        color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
        for i, v in enumerate(self.color_list):
            color_map[label == i] = np.array(v, dtype=np.uint8)
        return color_map

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = Image.open(item["img"]).convert("RGB")
        image = np.array(image)
        image_or = image.copy()
        size = image.shape

        color_map = Image.open(item["label"]).convert("RGB")
        color_map = np.array(color_map)
        label = self.color2label(color_map)  # all non-[0,0,0]/[1,1,1] -> ignore_label

        image, label, edge = self.gen_sample(
            image,
            label,
            self.multi_scale,
            self.flip,
            edge_pad=False,
            edge_size=self.bd_dilate_size,
            city=False,
        )
        if self.return_or_dims:
            return (
                image.copy(),
                label.copy(),
                edge.copy(),
                np.array(size),
                image_or,
                name,
            )
        else:
            return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        return self.inference(config, model, image)

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + ".png"))

    # --- convenience: build directly from cfg -------------------------------
    @classmethod
    def from_config(cls, cfg, split="val", **kwargs):
        """
        Helper to construct FloodDataset using a standard PIDNet config.

        Ensures:
        - root = cfg.DATASET.ROOT
        - ignore_label = cfg.TRAIN.IGNORE_LABEL
        - list_path = cfg.DATASET.TEST_SET / TRAIN_SET if provided
        """
        list_path = kwargs.pop("list_path", None)
        if list_path is None and hasattr(cfg.DATASET, "TEST_SET") and split == "val":
            list_path = cfg.DATASET.TEST_SET
        if list_path is None and hasattr(cfg.DATASET, "TRAIN_SET") and split == "train":
            list_path = cfg.DATASET.TRAIN_SET

        return cls(
            root=cfg.DATASET.ROOT,
            split=split,
            ignore_label=cfg.TRAIN.IGNORE_LABEL,
            list_path=list_path,
            **kwargs,
        )


# backward compat alias
Flood = FloodDataset
