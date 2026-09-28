import os
import random
from pathlib import Path

import numpy as np
import torch


def outer_prod_broadcasting(A, B):
    """Broadcasting trick"""
    return A[..., None] * B[:, None]


def set_all_seeds(seed: int):
    """
    Set Python, NumPy, and PyTorch seeds.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _ensure_tpu_runtime_env():
    """
    Prepare env vars before the first torch_xla import.

    - uv-managed CPython: libpython lives under ~/.local/share/uv/python/...
    - newer GCE TPU metadata uses TPU_ACCELERATOR_TYPE / HOST_BOUNDS, while
      torch_xla 2.9's tpu.version() still expects ACCELERATOR_TYPE in the MDS
      dict → KeyError. Skip MDS and supply ct6e-standard-4t (v6e-4) topology.
    """
    if "LD_LIBRARY_PATH" not in os.environ or "/uv/python/" not in os.environ.get(
        "LD_LIBRARY_PATH", ""
    ):
        uv_python = Path.home() / ".local" / "share" / "uv" / "python"
        if uv_python.is_dir():
            for lib in uv_python.rglob("libpython3.12.so.1.0"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                )
                break

    # Only set defaults; allow the caller / sweep script to override.
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
    os.environ.setdefault("ACCELERATOR_TYPE", "v6e-4")
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v6e-4")
    os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
    os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
    os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")
    os.environ.setdefault("TPU_WORKER_ID", "0")
    os.environ.setdefault("WORKER_ID", "0")


def get_device():
    """Prefer XLA/TPU when torch_xla is available, else CUDA, else CPU."""
    try:
        _ensure_tpu_runtime_env()
        import torch_xla.core.xla_model as xm
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return xm.xla_device()


def mark_step_if_xla():
    """Flush the XLA graph so TPU executes accumulated ops (no-op elsewhere)."""
    try:
        _ensure_tpu_runtime_env()
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    except ImportError:
        pass


def optimizer_step(optimizer):
    """XLA-aware optimizer step for BP baselines; falls back to optimizer.step()."""
    try:
        _ensure_tpu_runtime_env()
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer)
    except ImportError:
        optimizer.step()
