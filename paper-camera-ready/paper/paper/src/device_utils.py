"""Helpers for consistent torch device resolution across the project."""
from __future__ import annotations

import os
from typing import Iterable, Optional, Union

import torch

DeviceLike = Optional[Union[str, int, torch.device]]


def _parse_device(spec: DeviceLike) -> Optional[torch.device]:
    """Convert a loose device specification into a ``torch.device``.

    Accepts integers, strings (``"cuda"``, ``"cuda:1"``, ``"cpu"``) or ``torch.device``
    instances. Returns ``None`` if the specification is empty. Raises ``ValueError``
    for unsupported values or when CUDA is requested but unavailable.
    """
    if spec is None:
        return None

    if isinstance(spec, torch.device):
        return spec

    if isinstance(spec, int):
        if spec < 0:
            raise ValueError("Device index must be non-negative.")
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available.")
        if spec >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {spec} out of range (count={torch.cuda.device_count()})."
            )
        return torch.device(f"cuda:{spec}")

    spec_str = str(spec).strip()
    if spec_str == "":
        return None

    lowered = spec_str.lower()
    if lowered == "cpu":
        return torch.device("cpu")

    if lowered == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available.")
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    if lowered.isdigit():
        return _parse_device(int(lowered))

    if lowered.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available.")
        index_part = lowered.split(":", 1)[1]
        if index_part and not index_part.isdigit():
            raise ValueError(f"Invalid CUDA device specification: {spec_str}")
        if index_part:
            idx = int(index_part)
            if idx >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {idx} out of range (count={torch.cuda.device_count()})."
                )
        return torch.device(lowered)

    raise ValueError(f"Unsupported device specification: {spec_str}")


def resolve_device(
    preferred: DeviceLike = None,
    env_vars: Iterable[str] = ("EXPLANATOR_DEVICE", "TORCH_DEVICE", "CUDA_DEVICE"),
) -> torch.device:
    """Resolve the torch device to use.

    The order of preference is:
    1. Explicit ``preferred`` argument if provided.
    2. The first environment variable in ``env_vars`` that yields a valid device.
    3. The current CUDA device if CUDA is available.
    4. CPU.

    Invalid specifications are ignored until a valid option is found.
    """
    specs = [preferred]
    specs.extend(os.getenv(var) for var in env_vars)

    for spec in specs:
        try:
            device = _parse_device(spec)
        except ValueError:
            continue
        if device is not None:
            return device

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    return torch.device("cpu")


def device_to_str(device: DeviceLike) -> str:
    """Human-readable representation for logging."""
    parsed = device if isinstance(device, torch.device) else _parse_device(device)
    if parsed is None:
        return "unspecified"
    if parsed.type == "cuda" and parsed.index is not None:
        return f"cuda:{parsed.index}"
    return parsed.type
