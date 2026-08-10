#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import numpy as np
from contextlib import contextmanager
from typing import Dict, Optional

import torch

from LCRP.models import get_model
from LCRP.utils.pidnet_canonizers import PIDNetCanonizer, EpsilonPlusFlatforPIDNet, EpsilonPlusFlatMulforPIDNet
from src.datasets.flood_dataset import FloodDataset


# ============================ CONFIG ============================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "pidnet"
CKPT_PATH: Optional[str] = "/home/heydari/paper/LCRP/models/flood_model.pt"

USE_DATASET = True
DATA_ROOT = "/home/heydari/FHHI-XAI/data/flood_segmentation"
DATA_SPLIT = "train"       # try "val" if train has random aug
DATA_INDEX = 0

RAND_SHAPE = (1, 3, 256, 256)

OUTPUT_INDEX = 1           # PIDNet returns list/tuple; repo uses [1]
TARGET_CLASS = 1
PIX_Y = None               # None -> center
PIX_X = None               # None -> center

USE_DOUBLE = True
ATOL = 1e-5
RTOL = 1e-5

# Choose composite:
#   - EpsilonPlusFlatforPIDNet: your repo default (likely non-conservative)
#   - EpsilonPlusFlatMulforPIDNet: your "flat mul" variant (still may be non-conservative)
COMPOSITE = "default"  # "default" or "mul"

SAVE_RELEVANCE = True
RELEVANCE_OUT_PATH = "lrp_input_relevance.pt"
# ================================================================


# Determinism (reduce “why did it change” drama)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Disable TF32 for precision
if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def temporarily_zero_biases(model: torch.nn.Module):
    saved: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is not None and "bias" in name:
                saved[name] = p.detach().clone()
                p.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name])


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state", "model", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def build_model_like_you_do(device: str) -> torch.nn.Module:
    # Patch to avoid strict internal loads during get_model()
    _orig_torch_load = torch.load
    _orig_load_state_dict = torch.nn.Module.load_state_dict

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return _orig_torch_load(*args, **kwargs)

    def _lenient_load_state_dict(self, state_dict, strict=True):
        return _orig_load_state_dict(self, state_dict, strict=False)

    torch.load = _patched_load
    torch.nn.Module.load_state_dict = _lenient_load_state_dict
    try:
        model = get_model(model_name=MODEL_NAME, device=device, classes=2)
    finally:
        torch.load = _orig_torch_load
        torch.nn.Module.load_state_dict = _orig_load_state_dict

    if CKPT_PATH is not None:
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(CKPT_PATH)
        raw = torch.load(CKPT_PATH, map_location=device)
        sd = _extract_state_dict(raw)
        sd = _strip_prefix(sd, "model.")
        sd = _strip_prefix(sd, "module.")
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model


def load_input(device: str) -> torch.Tensor:
    if not USE_DATASET:
        return torch.randn(*RAND_SHAPE, device=device)
    ds = FloodDataset(root_dir=DATA_ROOT, split=DATA_SPLIT)
    x_np = ds[DATA_INDEX][0]
    return torch.from_numpy(x_np).unsqueeze(0).to(device)


def main():
    device = DEVICE
    model = build_model_like_you_do(device)

    if USE_DOUBLE:
        model = model.double()

    x = load_input(device)
    if USE_DOUBLE:
        x = x.double()

    # Apply PIDNetCanonizer (repo requirement)
    canonizer = PIDNetCanonizer()
    handles = canonizer.apply(model)

    try:
        # Pick composite
        composite_cls = EpsilonPlusFlatforPIDNet if COMPOSITE == "default" else EpsilonPlusFlatMulforPIDNet
        composite_name = composite_cls.__name__

        # Forward once to get output geometry + explained scalar
        with torch.no_grad():
            out0 = model(x)
            if isinstance(out0, (list, tuple)):
                out0 = out0[OUTPUT_INDEX]
            if out0.ndim != 4:
                raise ValueError(f"Expected logits [B,C,H,W], got {tuple(out0.shape)}")
            _, C, H, W = out0.shape

            y = (H // 2) if PIX_Y is None else int(PIX_Y)
            xx = (W // 2) if PIX_X is None else int(PIX_X)
            c = min(max(int(TARGET_CLASS), 0), C - 1)

        # Now do the actual sanity check run (bias=0, composite, one-pixel seed)
        model.zero_grad(set_to_none=True)
        x_req = x.detach().requires_grad_(True)

        with temporarily_zero_biases(model):
            with composite_cls().context(model) as modified_model:
                out = modified_model(x_req)
                if isinstance(out, (list, tuple)):
                    out = out[OUTPUT_INDEX]

                explained_t = out[0, c, y, xx]

                init = torch.zeros_like(out)
                init[0, c, y, xx] = 1.0  # seed ONLY that scalar
                out.backward(gradient=init)

                R_in = x_req.grad.detach()
                Rin_sum_t = R_in.sum()

        explained = float(explained_t.detach().cpu())
        Rin_sum = float(Rin_sum_t.detach().cpu())
        diff = explained - Rin_sum
        ok = bool(torch.isclose(
            torch.tensor(explained, dtype=torch.float64),
            torch.tensor(Rin_sum, dtype=torch.float64),
            atol=ATOL,
            rtol=RTOL
        ))

        print("\n=== LRP SANITY CHECK (ONE PIXEL) ===")
        print(f"Device: {device}")
        print(f"Composite: {composite_name}")
        print("Biases zeroed: True")
        print(f"Precision: {'float64' if USE_DOUBLE else 'float32'}")
        print(f"Explained position: class={c}, y={y}, x={xx}")
        print(f"Explained logit (one pixel): {explained}")
        print(f"Sum input relevance:        {Rin_sum}")
        print(f"Diff (explained - sumR):    {diff}")
        print(f"CONSERVATION:              {ok}")

        if SAVE_RELEVANCE:
            torch.save(R_in.cpu(), RELEVANCE_OUT_PATH)
            print(f"Saved input relevance tensor to: {RELEVANCE_OUT_PATH}")

    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()