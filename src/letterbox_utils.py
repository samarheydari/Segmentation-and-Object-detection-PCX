from PIL import Image
import cv2
import torch
import numpy as np
import math

# Import the exact same letterbox function used by inferer.py
from yolov6.data.data_augment import letterbox

import numpy as np


def rescale_boxes(boxes, letterbox_shape, original_shape):
    """
    Rescale boxes from letterbox coordinates to original image coordinates.
    Matches inferer.py rescale method exactly.
    """

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()

    boxes = np.array(boxes).copy()

    ratio = min(letterbox_shape[0] / original_shape[0],
                letterbox_shape[1] / original_shape[1])
    padding = (
        (letterbox_shape[1] - original_shape[1] * ratio) / 2,
        (letterbox_shape[0] - original_shape[0] * ratio) / 2
    )

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_shape[0])

    return boxes

def check_img_size(img_size, stride=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension.
    Exact copy from inferer.py's check_img_size method.
    """
    def make_divisible(x, divisor):
        return math.ceil(x / divisor) * divisor

    if isinstance(img_size, int):
        new_size = max(make_divisible(img_size, int(stride)), floor)
    elif isinstance(img_size, list):
        new_size = [max(make_divisible(x, int(stride)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    return new_size if isinstance(img_size, list) else [new_size, new_size]


def letterbox_transform(img, target_size=640, stride=64, half=False, auto=True):
    """Matches inferer.py process_image exactly"""
    if isinstance(img, Image.Image):
        img_np = np.array(img)[:, :, ::-1]  # PIL RGB to BGR
    else:
        img_np = img  # Assume BGR from cv2

    img_size = check_img_size(target_size, stride=stride)

    # Letterbox
    img_letterbox = letterbox(img_np, new_shape=img_size, stride=stride, auto=auto)[0]

    # Convert: HWC to CHW, BGR to RGB (exactly like inferer.py line 309)
    image = img_letterbox.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = image.half() if half else image.float()
    image /= 255.0

    return image