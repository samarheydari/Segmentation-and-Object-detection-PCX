import os
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def person_car_train(**kwargs):
    return PersonCarDataset(split="train", **kwargs)


def person_car_test(**kwargs):
    root = kwargs.get("root", _get_data_root())
    base = _resolve_dataset_root(root)
    split = "val" if (base / "images" / "val").is_dir() else "train"
    return PersonCarDataset(split=split, **kwargs)


def _get_data_root():
    return os.getenv(
        "PERSON_CAR_DATA_ROOT",
        "/home/heydari/FHHI-XAI/data/person_car_detection_data/original_BRK",
    )


def _resolve_dataset_root(root: str) -> Path:
    root_path = Path(root)
    nested = root_path / "original_BRK"
    return nested if nested.is_dir() else root_path


def _natural_key(name: str):
    # Keep ordering behavior close to natsort without extra dependency.
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _check_img_size(img_size, stride=32):
    if isinstance(img_size, int):
        s = int(np.ceil(img_size / stride) * stride)
        return [s, s]
    return [int(np.ceil(x / stride) * stride) for x in img_size]


def _letterbox_transform(img: Image.Image, target_size=640, stride=64, half=False, auto=False):
    img_np = np.array(img)[:, :, ::-1]  # RGB -> BGR
    img_size = _check_img_size(target_size, stride=stride)
    img_letterbox = _letterbox(img_np, new_shape=img_size, stride=stride, auto=auto)[0]
    image = img_letterbox.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0
    return tensor


def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints.
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class PersonCarDataset(Dataset):
    class_names = ("person", "car")

    def __init__(self, split="train", root=None, transform=None, **kwargs):
        self.root = _resolve_dataset_root(root or _get_data_root())
        self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"PersonCar split '{split}' not found. Expected:\n"
                f"  images: {self.image_dir}\n"
                f"  labels: {self.label_dir}"
            )

        self.transform = transform or (lambda img: _letterbox_transform(img, target_size=640, stride=64, half=False, auto=False))
        self.reverse_normalization = lambda data: torch.multiply(data, 255).detach().cpu()

        exts = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
        image_files = [p.name for p in self.image_dir.iterdir() if p.suffix in exts]
        label_files = [p.name for p in self.label_dir.iterdir() if p.suffix == ".txt"]
        image_files.sort(key=_natural_key)
        label_files.sort(key=_natural_key)

        image_stems = {Path(name).stem for name in image_files}
        label_stems = {Path(name).stem for name in label_files}
        common = image_stems & label_stems
        self.image_files: List[str] = [f for f in image_files if Path(f).stem in common]
        self.label_files: List[str] = [f for f in label_files if Path(f).stem in common]

        self.image_files.sort(key=_natural_key)
        self.label_files.sort(key=_natural_key)
        if len(self.image_files) != len(self.label_files):
            raise RuntimeError("Mismatch between number of images and labels.")
        for image_file, label_file in zip(self.image_files, self.label_files):
            if Path(image_file).stem != Path(label_file).stem:
                raise RuntimeError(f"Mismatch between image and label files: {image_file} and {label_file}")

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def _line_to_obj(line: str):
        class_id, x_center, y_center, width, height = line.strip().split()
        return {
            "class_id": int(class_id),
            "x_center": float(x_center),
            "y_center": float(y_center),
            "width": float(width),
            "height": float(height),
        }

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]

        img_path = self.image_dir / image_file
        label_path = self.label_dir / label_file

        image = Image.open(img_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            objects = [self._line_to_obj(line) for line in f.readlines() if line.strip()]

        image = self.transform(image)

        labels = [obj["class_id"] for obj in objects]
        targets = torch.tensor(labels, dtype=torch.long)
        targets_transformed = targets[:, None].expand(targets.shape[0], 2)
        return image, targets_transformed
