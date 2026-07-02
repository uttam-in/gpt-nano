"""
Device selection: one function that works on Mac (M1/MPS), NVIDIA (CUDA), and CPU.

Every training / inference script in this project calls get_device() so the SAME
code runs on your Mac M1 today and your NVIDIA A6000 later — no edits needed.

Priority order:
  1. CUDA  (NVIDIA GPU)  — fastest, supports bfloat16 mixed precision
  2. MPS   (Apple GPU)   — Metal Performance Shaders, the M1's GPU
  3. CPU   — always works, slowest
"""

import torch


def get_device(force: str | None = None) -> torch.device:
    """Return the best available torch device.

    force: pass "cpu", "cuda", or "mps" to override auto-detection
           (useful for debugging: CPU errors are more readable).
    """
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Pick a training dtype for the device.

    - CUDA (A6000): bfloat16 — half the memory, ~2x speed, numerically stable.
    - MPS  (M1):    float32  — MPS mixed-precision support is still patchy,
                    so we keep full precision. A ~15M-param model fits easily.
    - CPU:          float32.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def autocast_context(device: torch.device):
    """Return a context manager for mixed-precision on CUDA, or a no-op elsewhere.

    Usage:
        with autocast_context(device):
            logits, loss = model(x, y)
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=get_dtype(device))
    # nullcontext = "do nothing" — full precision on MPS/CPU
    import contextlib

    return contextlib.nullcontext()


def describe(device: torch.device) -> str:
    """Human-readable description of what we're running on."""
    if device.type == "cuda":
        return f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return "Apple GPU (MPS / Metal)"
    return "CPU"


if __name__ == "__main__":
    d = get_device()
    print(f"Selected device : {d}")
    print(f"Description     : {describe(d)}")
    print(f"Training dtype  : {get_dtype(d)}")
