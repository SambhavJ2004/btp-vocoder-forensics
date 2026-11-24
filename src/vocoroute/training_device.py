"""Explicit single-device FP32 execution for the base trainer."""

import os
from contextlib import contextmanager

import torch


def resolve_device(kind: str) -> torch.device:
    if kind == "cpu":
        return torch.device("cpu")
    if kind != "cuda":
        raise ValueError("training device must be cpu or cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA was requested, but no usable CUDA device is available")
    return torch.device("cuda", 0)


@contextmanager
def training_runtime(kind: str):
    """Select deterministic algorithms and restore process settings on exit."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if kind == "cuda" and workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    flags = None
    try:
        device = resolve_device(kind)
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            flags = (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark,
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        yield device
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if flags is not None:
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark,
            ) = flags
        if kind == "cuda" and workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def accelerator_record(device: torch.device) -> dict | None:
    if device.type == "cpu":
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "index": device.index,
        "capability": [properties.major, properties.minor],
        "memory_bytes": properties.total_memory,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
