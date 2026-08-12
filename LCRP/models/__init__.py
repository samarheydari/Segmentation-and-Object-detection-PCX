import importlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_callable(module_path: str, attr_name: str):
    # Lazy import keeps optional dependencies isolated.
    # Example: segmentation_models_pytorch is only needed when "unet" is requested.
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _get_smp_unet(**kwargs):
    get_smp = _load_callable("LCRP.models.smp", "get_smp")
    return get_smp("unet")(**kwargs)


class PIDNetPerturbationAdapter(nn.Module):
    """
    Adapter around the original PIDNet module.

    Why this exists:
    - We keep `models/pidnet.py` untouched.
    - `instance_perturbation.py` expects `model(x)` to return one tensor.
    - PIDNet training-style forward can return a list/tuple.

    What this adapter does:
    1) Picks the main segmentation logits tensor.
    2) Resizes logits to match the input spatial size.
       This keeps downstream masking math aligned with dataset masks.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)

        # PIDNet with augment=True can return [aux_p, main, aux_d].
        # For perturbation we evaluate prediction change on the main head.
        if isinstance(out, (list, tuple)):
            logits = out[1] if len(out) > 1 else out[0]
        else:
            logits = out

        # Ensure logits and target masks share the same HxW in the experiment loop.
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def _get_pidnet_for_perturbation(**kwargs):
    """
    Build untouched PIDNet and adapt its interface for perturbation experiments.

    Checkpoint behavior:
    - If caller passed `ckpt_path`, keep it.
    - Otherwise pull from `PIDNET_CKPT_PATH` environment variable.
    """
    get_pidnet = _load_callable("LCRP.models.pidnet", "get_pidnet")

    # We intentionally bypass strict internal checkpoint loading from models/pidnet.py:
    # build architecture first, then load checkpoint here with flexible key handling.
    
    ckpt_path = kwargs.pop("ckpt_path", None) or os.getenv("PIDNET_CKPT_PATH")
    base = get_pidnet(**kwargs)

    if ckpt_path:
        device = kwargs.get("device", "cpu")
        raw = torch.load(ckpt_path, map_location=device)
        state_dict = raw

        # Accept common checkpoint containers.
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state", "model", "net", "module"):
                if key in raw and isinstance(raw[key], dict):
                    state_dict = raw[key]
                    break

        # Normalize common prefixes.
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[pidnet-adapter] missing keys: {len(missing)}")
        if unexpected:
            print(f"[pidnet-adapter] unexpected keys: {len(unexpected)}")

    return PIDNetPerturbationAdapter(base)


MODELS = {
    # object detectors
    "yolov5": ("LCRP.models.yolov5", "get_yolov5"),
    "yolov6": ("LCRP.models.yolov6", "get_yolov6"),
    "yolov6s6": ("LCRP.models.yolov6", "get_yolov6s6"),
    # segmentation models
    "unet": _get_smp_unet,
    "deeplabv3plus": ("LCRP.models.deeplabv3plus", "get_deeplabv3plus"),
    # Intentionally use adapter for experiment compatibility while keeping PIDNet logic unchanged.
    "pidnet": _get_pidnet_for_perturbation,
}


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name not in MODELS:
        print(f"Model {model_name} not available")
        exit()

    target = MODELS[model_name]
    if callable(target):
        return target(**kwargs)

    module_path, attr_name = target
    fn = _load_callable(module_path, attr_name)
    return fn(**kwargs)
