import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.datasets.base_dataset import BaseDataset


class FloodDataset(BaseDataset):
    """
    Simplified Flood segmentation dataset for testing.
    - Loads images from a directory (with or without masks)
    - No split structure needed
    - Supports maskless testing
    """

    class_names = ["background", "flood"]
    color_list = [[0, 0, 0], [1, 1, 1]]

    def __init__(
        self,
        root: Optional[str] = None,
        root_dir: Optional[str] = None,
        split: Optional[str] = None,  # kept for compatibility, but ignored
        transform=None,
        num_classes: int = 2,
        multi_scale: bool = False,
        flip: bool = False,
        ignore_label: Optional[int] = None,
        base_size: int = 2048,
        crop_size: Tuple[int, int] = (1280, 720),
        scale_factor: int = 16,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        bd_dilate_size: int = 4,
        return_or_dims: bool = False,
        list_path: Optional[str] = None,
        **kwargs,
    ):
        # Accept either `root` or `root_dir` for compatibility
        if root is None and root_dir is not None:
            root = root_dir
        if root is None:
            raise ValueError("root or root_dir must be provided to FloodDataset")

        # Set ignore label
        if ignore_label is None:
            ignore_label = -1
        self.ignore_label = int(ignore_label)

        # Initialize BaseDataset
        super(FloodDataset, self).__init__(
            ignore_label, base_size, crop_size, scale_factor, mean, std
        )

        root_path = Path(root)
        self.base_root = root
        self.dataset_root = root
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.bd_dilate_size = bd_dilate_size
        self.return_or_dims = return_or_dims

        # Support a single image path directly
        if root_path.is_file():
            self.image_dir = str(root_path.parent)
            self.mask_dir = None
            self.files = [{
                "img": root_path,
                "label": None,
                "name": self._stem_no_ext(root_path),
            }]
            self.class_weights = None
            return

        # Look for images in common directory names
        possible_img_dirs = [
            os.path.join(self.dataset_root, "RGB"),
            os.path.join(self.dataset_root, "images"),
            self.dataset_root,
        ]
        self.image_dir = None
        for img_dir in possible_img_dirs:
            if os.path.isdir(img_dir):
                self.image_dir = img_dir
                break

        if self.image_dir is None:
            raise ValueError(f"No image directory found in: {possible_img_dirs}")

        # Look for masks (optional)
        possible_mask_dirs = [
            os.path.join(self.dataset_root, "annotations"),
            os.path.join(self.dataset_root, "masks"),
            os.path.join(self.dataset_root, "labels"),
        ]
        self.mask_dir = None
        for mask_dir in possible_mask_dirs:
            if os.path.isdir(mask_dir):
                self.mask_dir = mask_dir
                break

        # Load file list
        if list_path is not None:
            self.files = self._files_from_list(list_path)
        else:
            self.files = self._scan_and_pair()

        self.class_weights = None

    def _stem_no_ext(self, p: Path) -> str:
        return p.stem.rstrip("_")

    def _scan_and_pair(self):
        """Scan for images, optionally pair with masks if available."""
        img_exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")

        # List all images
        image_files_all = [
            Path(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(img_exts)
        ]
        image_files_all.sort()

        files = []

        # If mask directory exists, try to pair with masks
        used_img_stems = set()
        if self.mask_dir is not None and os.path.isdir(self.mask_dir):
            mask_files_all = [
                Path(self.mask_dir, f)
                for f in os.listdir(self.mask_dir)
                if f.endswith(img_exts)
            ]
            mask_files_all.sort()

            # Simple pairing by filename
            for img_path in image_files_all:
                img_stem = self._stem_no_ext(img_path)
                for mask_path in mask_files_all:
                    mask_stem = self._stem_no_ext(mask_path)
                    if img_stem == mask_stem:
                        files.append({
                            "img": img_path,
                            "label": mask_path,
                            "name": img_stem,
                        })
                        used_img_stems.add(img_stem)
                        break

        # Also include any unpaired images as maskless samples
        if len(image_files_all) > 0:
            for img_path in image_files_all:
                name = self._stem_no_ext(img_path)
                if used_img_stems and name in used_img_stems:
                    continue
                files.append({
                    "img": img_path,
                    "label": None,  # No mask
                    "name": name,
                })

        assert files, f"No images found in {self.image_dir}"
        return files

    def _files_from_list(self, list_path: str):
        """Load file list from a text file."""
        list_file = Path(list_path)
        if not list_file.is_absolute():
            list_file = Path(self.base_root) / list_path
        if not list_file.exists():
            raise FileNotFoundError(f"List file not found: {list_file}")

        files = []
        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                img_rel = parts[0]
                img_path = Path(self.dataset_root) / img_rel
                
                # Optional mask in second column
                label_path = None
                if len(parts) >= 2:
                    mask_rel = parts[1]
                    label_path = Path(self.dataset_root) / mask_rel
                
                name = img_path.stem
                files.append({
                    "img": img_path,
                    "label": label_path,
                    "name": name,
                })

        assert files, f"No entries loaded from list file: {list_file}"
        return files

    def __len__(self):
        return len(self.files)

    def color2label(self, color_map: np.ndarray) -> np.ndarray:
        """Map RGB colors to class labels."""
        label = np.full(color_map.shape[:2], self.ignore_label, dtype=np.int32)
        for i, v in enumerate(self.color_list):
            v = np.array(v, dtype=np.uint8)
            mask = (color_map == v).sum(2) == 3
            label[mask] = np.int32(i)
        return label

    def label2color(self, label: np.ndarray) -> np.ndarray:
        """Map class labels back to RGB colors."""
        color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
        for i, v in enumerate(self.color_list):
            color_map[label == i] = np.array(v, dtype=np.uint8)
        return color_map

    def __getitem__(self, index):
        item = self.files[index]

        # Load image
        image = Image.open(item["img"]).convert("RGB")
        image = np.array(image)

        # Load or create label
        if item["label"] is not None:
            color_map = Image.open(item["label"]).convert("RGB")
            color_map = np.array(color_map)
            label = self.color2label(color_map)
        else:
            # No mask - create dummy label with ignore_label
            label = None

        # Preprocess image and label
        if label is not None:
            image, label, _ = self.gen_sample(
                image,
                label,
                self.multi_scale,
                self.flip,
                edge_pad=False,
                edge_size=self.bd_dilate_size,
                city=False,
            )
        else:
            # Process image only
            dummy_label = np.zeros(image.shape[:2], dtype=np.int32)
            image_proc, _, _ = self.gen_sample(
                image,
                dummy_label,
                self.multi_scale,
                self.flip,
                edge_pad=False,
                edge_size=self.bd_dilate_size,
                city=False,
            )
            image = image_proc
            # Create label filled with ignore_label
            label_shape = image.shape[:2] if image.ndim == 3 else image.shape
            label = np.full(label_shape, self.ignore_label, dtype=np.int32)

        # Convert to tensors
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 3:
            img_chw = image
        elif image.ndim == 3 and image.shape[2] == 3:
            img_chw = image.transpose(2, 0, 1)
        else:
            try:
                img_chw = image.transpose(2, 0, 1)
            except Exception:
                img_chw = image

        image = torch.from_numpy(img_chw).float()
        label = np.asarray(label, dtype=np.int64)
        label = torch.from_numpy(label).long()

        return image, label

    def single_scale_inference(self, config, model, image):
        return self.inference(config, model, image)

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + ".png"))

    def reverse_normalization(self, data: torch.Tensor) -> torch.Tensor:
        """Undo normalization."""
        import torch as _torch

        if not isinstance(data, _torch.Tensor):
            data = _torch.from_numpy(np.array(data))

        x = data.float()
        means = _torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        stds = _torch.tensor(self.std, dtype=x.dtype, device=x.device).view(-1, 1, 1)

        x = x * stds + means
        x = x * 255.0
        x = x.clamp(0, 255).to(_torch.float32).cpu()
        return x

    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Convert preprocessed tensor back to displayable image."""
        import torch as _torch

        try:
            x = self.reverse_normalization(data)
            if x.ndim == 3 and x.shape[0] >= 3:
                return x[:3]
            return x
        except Exception:
            if not isinstance(data, _torch.Tensor):
                data = _torch.from_numpy(np.array(data))
            x = data.float()
            mn = x.min()
            mx = x.max()
            if mx == mn:
                x = _torch.zeros_like(x)
            else:
                x = (x - mn) / (mx - mn)
            x = (x * 255.0).clamp(0, 255).to(_torch.float32).cpu()
            if x.ndim == 3 and x.shape[0] >= 3:
                return x[:3]
            return x

    @classmethod
    def from_config(cls, cfg, split="val", **kwargs):
        """Helper to construct from config."""
        list_path = kwargs.pop("list_path", None)
        return cls(
            root=cfg.DATASET.ROOT,
            split=split,
            ignore_label=cfg.TRAIN.IGNORE_LABEL,
            list_path=list_path,
            **kwargs,
        )


# backward compat alias
Flood = FloodDataset
