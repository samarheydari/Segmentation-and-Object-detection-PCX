import torch

from .deeplabv3plus import get_deeplabv3plus
from .yolov5 import get_yolov5
from .yolov6 import get_yolov6, get_yolov6s6
from .pidnet import get_pidnet
# from .ssd import get_ssd


def _get_unet(**kwargs):
    # Lazy import avoids pulling timm/wandb unless unet is requested.
    from .smp import get_smp
    return get_smp("unet")(**kwargs)


MODELS = {
    # object detectors
    "yolov5": get_yolov5,
    "yolov6": get_yolov6,
    "yolov6s6": get_yolov6s6,
#     "ssd": get_ssd,
    # segmentation models
    "unet": _get_unet,
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
