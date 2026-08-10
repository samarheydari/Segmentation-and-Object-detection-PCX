import torch

from .deeplabv3plus import get_deeplabv3plus
from .smp import get_smp
from .yolov5 import get_yolov5
from .yolov6 import get_yolov6, get_yolov6s6
from .pidnet import get_pidnet
# from .ssd import get_ssd
MODELS = {
    # object detectors
    "yolov5": get_yolov5,
    "yolov6": get_yolov6,
    "yolov6s6": get_yolov6s6,
#     "ssd": get_ssd,
    # segmentation models
    "unet": get_smp("unet"),
    "deeplabv3plus": get_deeplabv3plus,
    "pidnet": get_pidnet, 
}

def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name in MODELS:
        model = MODELS[model_name](**kwargs)
        return model
    else:
        print(f"Model {model_name} not available")
        exit()