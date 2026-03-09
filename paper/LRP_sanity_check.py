"""
LRP sanity check (one pixel)

What it does :
1) Uses an epsilon-rule composite (you choose which composite class to use).
2) Sets ALL model biases to zero (temporarily, then restores them).
3) Seeds relevance for ONE scalar: logit[0, target_class, y, x] (one pixel).
4) Checks conservation: sum(input_relevance) == explained logit (pre-softmax / pre-sigmoid).

How to run (inside repo root):
  python "LRP sanity check.py" --help

Examples:
  # Random model + random input (no data needed)
  python "LRP sanity check.py" --random_input --target_class 1 --y 32 --x 64

  # If you have a checkpoint and want to load it
  python "LRP sanity check.py" --ckpt /path/to/pidnet.pth --random_input --target_class 1 --y 32 --x 64

Notes:
- PIDNet outputs are usually [B, C, H, W]. y/x must be within that output resolution.
- Runs in double precision by default to reduce numerical error.
"""

import argparse
from contextlib import contextmanager
import os
import sys
from typing import Dict, Tuple, Optional

import torch


# ---------- Utilities ----------

@contextmanager
def temporarily_zero_biases(model: torch.nn.Module):
    """Zero all parameters whose name contains 'bias', and restore them afterwards."""
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


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_pidnet_model(model_name: str, ckpt_path: Optional[str], device: torch.device) -> torch.nn.Module:
    """
    Loads PIDNet via repo's get_model if available.
    Falls back to a clear error if repo modules aren't importable.
    """
    try:
        # In this branch, scripts use: from LCRP.models import get_model
        from LCRP.models import get_model  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import LCRP.models.get_model. "
            "Run this script from the repo root (L-CRP) and ensure your environment is set."
        ) from e

    model = get_model(model_name)
    model.to(device)

    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Handle common checkpoint formats
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported checkpoint format (expected dict-like state).")

        # Strip possible 'module.' prefixes
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.eval()
    return model


def load_composite(composite_name: str):
    """
    Loads the epsilon-rule composite class from utils.pidnet_canonizers.
    This is where 'eps-rule in all layers' *actually* lives in this repo.
    """
    try:
        from LCRP.utils import pidnet_canonizers  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import utils.pidnet_canonizers. "
            "Run this script from the repo root (L-CRP)."
        ) from e

    if not hasattr(pidnet_canonizers, composite_name):
        available = [n for n in dir(pidnet_canonizers) if "forPIDNet" in n or "PIDNet" in n]
        raise AttributeError(
            f"Composite '{composite_name}' not found in utils.pidnet_canonizers.\n"
            f"Available (filtered): {available}"
        )

    composite_cls = getattr(pidnet_canonizers, composite_name)
    return composite_cls


# ---------- Core sanity check ----------

def lrp_sanity_check_one_pixel(
    model: torch.nn.Module,
    composite_cls,
    inp: torch.Tensor,
    target_class: int,
    y: int,
    x: int,
    *,
    output_index: int = 1,
    sanity_check_conservation: bool = True,
    use_double: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tuple[torch.Tensor, float, float, float, bool]:
    """
    Returns:
      R_in (tensor): input relevance tensor (same shape as inp)
      explained (float): the explained one-pixel logit
      Rin_sum (float): sum of input relevance
      diff (float): explained - Rin_sum
      ok (bool): conservation check
    """
    if use_double:
        model = model.double()
        inp = inp.double()

    model.zero_grad(set_to_none=True)

    inp = inp.requires_grad_(True)
    if inp.grad is not None:
        inp.grad.zero_()

    # Forward once to validate output shape + indices
    with torch.no_grad():
        out0 = model(inp)
        out0 = out0[output_index] if isinstance(out0, (list, tuple)) else out0
        if out0.ndim != 4:
            raise ValueError(f"Expected segmentation logits [B,C,H,W], got {tuple(out0.shape)}")
        B, C, H, W = out0.shape
        if not (0 <= target_class < C):
            raise ValueError(f"target_class={target_class} out of range [0, {C-1}]")
        if not (0 <= y < H and 0 <= x < W):
            raise ValueError(f"(y,x)=({y},{x}) out of range: H={H}, W={W}")

    bias_ctx = temporarily_zero_biases(model) if sanity_check_conservation else contextmanager(lambda: (yield))()

    # Zennit composite context expects to wrap the model
    with bias_ctx:
        with composite_cls().context(model) as modified_model:
            out = modified_model(inp)
            out = out[output_index] if isinstance(out, (list, tuple)) else out

            explained_t = out[0, target_class, y, x]  # ONE PIXEL scalar

            init = torch.zeros_like(out)
            init[0, target_class, y, x] = 1.0  # seed only that pixel
            out.backward(gradient=init)

            R_in = inp.grad
            Rin_sum_t = R_in.sum()

    explained = float(explained_t.detach().cpu())
    Rin_sum = float(Rin_sum_t.detach().cpu())
    diff = float((explained_t - Rin_sum_t).detach().cpu())
    ok = bool(torch.isclose(explained_t.detach(), Rin_sum_t.detach(), atol=atol, rtol=rtol).detach().cpu())

    return R_in.detach(), explained, Rin_sum, diff, ok


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="LRP sanity check (one pixel) for PIDNet branch galip_PIDNet.")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    parser.add_argument("--model_name", default="pidnet", help="Model name used by LCRP.models.get_model()")
    parser.add_argument("--ckpt", default=None, help="/home/heydari/paper/LCRP/models/flood_model.pt")
    parser.add_argument("--composite", default="EpsilonPlusFlatforPIDNet",
                        help="Composite class name in utils.pidnet_canonizers (epsilon-rule composite).")

    parser.add_argument("--random_input", action="store_true", help="Use random input instead of dataset sample.")
    parser.add_argument("--in_h", type=int, default=256, help="Random input height (only if --random_input).")
    parser.add_argument("--in_w", type=int, default=256, help="Random input width (only if --random_input).")
    parser.add_argument("--in_c", type=int, default=3, help="Random input channels (only if --random_input).")

    parser.add_argument("--target_class", type=int, required=True, help="Class index to explain.")
    parser.add_argument("--y", type=int, required=True, help="Pixel y index in OUTPUT logits space.")
    parser.add_argument("--x", type=int, required=True, help="Pixel x index in OUTPUT logits space.")

    parser.add_argument("--output_index", type=int, default=1,
                        help="If model returns tuple/list, which element is logits. Repo scripts often use [1].")
    parser.add_argument("--no_bias_zero", action="store_true", help="Disable bias-zeroing (supervisor asked: keep enabled).")
    parser.add_argument("--float32", action="store_true", help="Use float32 (not recommended for precision).")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for conservation check.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for conservation check.")

    args = parser.parse_args()

    device = pick_device(args.device)

    composite_cls = load_composite(args.composite)
    model = load_pidnet_model(args.model_name, args.ckpt, device=device)

    if args.random_input:
        inp = torch.randn(1, args.in_c, args.in_h, args.in_w, device=device)
    else:
        # If you want dataset-based input, wire it here.
        # This script intentionally avoids requiring data because sanity check can be done on random input.
        raise RuntimeError(
            "Dataset loading is not implemented here on purpose.\n"
            "Use --random_input for the sanity check (as your supervisor said it is sufficient)."
        )

    use_double = not args.float32

    R_in, explained, Rin_sum, diff, ok = lrp_sanity_check_one_pixel(
        model=model,
        composite_cls=composite_cls,
        inp=inp,
        target_class=args.target_class,
        y=args.y,
        x=args.x,
        output_index=args.output_index,
        sanity_check_conservation=(not args.no_bias_zero),
        use_double=use_double,
        atol=args.atol,
        rtol=args.rtol,
    )

    print("\n=== LRP SANITY CHECK (ONE PIXEL) ===")
    print(f"Device: {device}")
    print(f"Composite: {args.composite}")
    print(f"Biases zeroed: {not args.no_bias_zero}")
    print(f"Precision: {'float64' if use_double else 'float32'}")
    print(f"Explained logit (one pixel): {explained}")
    print(f"Sum input relevance:        {Rin_sum}")
    print(f"Diff (explained - sumR):    {diff}")
    print(f"CONSERVATION:              {ok}")

    # Optional: save relevance to disk for debugging
    out_path = "lrp_input_relevance.pt"
    torch.save(R_in.detach().cpu(), out_path)
    print(f"Saved input relevance tensor to: {out_path}")


if __name__ == "__main__":
    main()