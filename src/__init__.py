import importlib
import os
import sys
import types

import numpy as np


def _apply_numpy_compat():
    # NumPy 2.x removed aliases that older transitive deps still reference.
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128


def _apply_wandb_stub():
    # Some timm builds import wandb eagerly; avoid importing incompatible wandb.
    if "wandb" not in sys.modules:
        wandb_stub = types.ModuleType("wandb")
        wandb_stub.__dict__["__version__"] = "0.0"
        sys.modules["wandb"] = wandb_stub


def _apply_lcrp_models_shim():
    """
    Provide a lightweight `LCRP.models` package object without executing
    `LCRP/models/__init__.py` (which eagerly imports optional heavy deps).
    """
    if "LCRP.models" in sys.modules:
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(repo_root, "LCRP", "models")

    shim = types.ModuleType("LCRP.models")
    shim.__path__ = [models_dir]

    def get_model(model_name: str, **kwargs):
        if model_name == "pidnet":
            return importlib.import_module("LCRP.models.pidnet").get_pidnet(**kwargs)
        if model_name == "deeplabv3plus":
            return importlib.import_module("LCRP.models.deeplabv3plus").get_deeplabv3plus(**kwargs)
        if model_name == "yolov5":
            return importlib.import_module("LCRP.models.yolov5").get_yolov5(**kwargs)
        if model_name == "yolov6":
            return importlib.import_module("LCRP.models.yolov6").get_yolov6(**kwargs)
        if model_name == "yolov6s6":
            return importlib.import_module("LCRP.models.yolov6").get_yolov6s6(**kwargs)
        raise ValueError(f"Model {model_name} not available")

    shim.get_model = get_model
    sys.modules["LCRP.models"] = shim


_apply_numpy_compat()
_apply_wandb_stub()
_apply_lcrp_models_shim()

