"""Fixed-shape ONNX export and numerical validation for the complete model."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from ..model.milestone2 import build_detection_model, load_coco_warmstart
from ..model.unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
    set_context_gradient_scale,
    set_cross_attention_enabled,
    set_relevance_perception_gradient_scale,
)

OUTPUT_NAMES = (
    "unified_detection",
    "state_logits",
    "round_logits",
    "maneuver_logits",
    "ego_lane_logits",
    "traffic_candidate_indices",
    "traffic_candidate_valid",
    "arrow_candidate_indices",
    "arrow_candidate_valid",
    "relevance_logits",
    "attention_weights",
)


class FullModelExportWrapper(nn.Module):
    """Expose the stable padded-set deployment tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(image)
        if not isinstance(output, tuple) or len(output) != len(OUTPUT_NAMES):
            raise RuntimeError(
                f"full export model did not return {len(OUTPUT_NAMES)} tensors"
            )
        return output


def build_full_model(
    *,
    weights_path: str | Path = "yolo11n.pt",
    checkpoint: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    payload: Any = None
    checkpoint_path: Path | None = None
    head_config = UnifiedHeadConfig()
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = payload.get("config", {}) if isinstance(payload, dict) else {}
        if isinstance(saved_config, dict):
            head_config = UnifiedHeadConfig(**saved_config.get("architecture", {}))
    wrapper = build_detection_model()
    warmstart = load_coco_warmstart(wrapper, weights_path)
    attach_unified_relevance_head(wrapper, config=head_config)
    checkpoint_report: dict[str, Any] = {"loaded": False}
    if checkpoint_path is not None:
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        result = wrapper.model.load_state_dict(state, strict=True)
        checkpoint_report = {
            "loaded": True,
            "path": str(checkpoint_path),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    return wrapper, {"warmstart": warmstart, "checkpoint": checkpoint_report}


def prepare_export_model(
    wrapper: Any,
    *,
    device: torch.device,
    half: bool,
) -> FullModelExportWrapper:
    set_cross_attention_enabled(wrapper, True)
    set_context_gradient_scale(wrapper, 0.0)
    set_relevance_perception_gradient_scale(wrapper, 0.0)
    model = wrapper.model.to(device).eval()
    model = model.half() if half else model.float()
    head = model.model[-1]
    if not isinstance(head, UnifiedTrafficControlDetect):
        raise TypeError("complete unified attention head is required for export")
    head.export = True
    head.format = "onnx"
    return FullModelExportWrapper(model).eval()


def export_full_onnx(
    wrapper: Any,
    output: str | Path,
    *,
    input_size: tuple[int, int] = (800, 1600),
    device: str = "cuda",
    half: bool = True,
    opset: int = 17,
) -> dict[str, Any]:
    resolved = torch.device(device)
    use_half = half and resolved.type == "cuda"
    export_model = prepare_export_model(wrapper, device=resolved, half=use_half)
    dtype = torch.float16 if use_half else torch.float32
    height, width = input_size
    sample = torch.zeros((1, 3, height, width), device=resolved, dtype=dtype)
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = export_model(sample)
        torch.onnx.export(
            export_model,
            sample,
            destination,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=None,
            dynamo=False,
        )
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "opset": opset,
        "dtype": str(dtype).removeprefix("torch."),
        "input_shape": list(sample.shape),
        "output_names": list(OUTPUT_NAMES),
        "output_shapes": [list(value.shape) for value in reference],
        "nodes": len(graph.graph.node),
        "checker_ok": True,
        "p2_enabled": False,
    }


def run_onnxruntime_parity(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (320, 320),
    atol: float = 2e-3,
) -> dict[str, Any]:
    """Compare a temporary FP32 graph with ONNX Runtime CPU."""

    export_model = prepare_export_model(
        wrapper, device=torch.device("cpu"), half=False
    )
    height, width = input_size
    generator = torch.Generator().manual_seed(42)
    sample = torch.rand((1, 3, height, width), generator=generator)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        path = Path(directory) / "parity.onnx"
        with torch.inference_mode():
            reference = [value.cpu().numpy() for value in export_model(sample)]
            torch.onnx.export(
                export_model,
                sample,
                path,
                opset_version=17,
                do_constant_folding=True,
                input_names=["images"],
                output_names=list(OUTPUT_NAMES),
                dynamic_axes=None,
                dynamo=False,
            )
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {"images": sample.numpy()})

    def canonicalize_candidates(values: list[np.ndarray]) -> list[np.ndarray]:
        """Remove harmless TopK tie permutations before numerical comparison."""

        if len(values) != len(OUTPUT_NAMES):
            return values
        result = [value.copy() for value in values]
        traffic_order = np.argsort(result[5], axis=1, kind="stable")
        result[5] = np.take_along_axis(result[5], traffic_order, axis=1)
        result[6] = np.take_along_axis(result[6], traffic_order, axis=1)
        result[9] = np.take_along_axis(
            result[9], traffic_order[:, None, :], axis=2
        )
        result[10] = np.take_along_axis(
            result[10], traffic_order[:, None, :, None], axis=2
        )
        arrow_order = np.argsort(result[7], axis=1, kind="stable")
        result[7] = np.take_along_axis(result[7], arrow_order, axis=1)
        result[8] = np.take_along_axis(result[8], arrow_order, axis=1)
        arrow_attention = np.take_along_axis(
            result[10][..., :-1], arrow_order[:, None, None, :], axis=3
        )
        result[10] = np.concatenate((arrow_attention, result[10][..., -1:]), axis=3)
        return result

    reference = canonicalize_candidates(reference)
    actual = canonicalize_candidates(actual)
    comparisons = {}
    all_close = True
    for name, expected, observed in zip(OUTPUT_NAMES, reference, actual):
        if expected.dtype.kind in {"b", "i", "u"}:
            difference = np.not_equal(expected, observed)
            maximum = float(difference.max(initial=False))
            mean = float(difference.mean())
            close = bool(np.array_equal(expected, observed))
        else:
            difference = np.abs(expected - observed)
            maximum = float(difference.max(initial=0.0))
            mean = float(difference.mean())
            close = bool(np.allclose(expected, observed, atol=atol, rtol=1e-3))
        all_close &= close
        comparisons[name] = {
            "shape": list(expected.shape),
            "max_abs_error": maximum,
            "mean_abs_error": mean,
            "within_tolerance": close,
        }
    if not all_close:
        raise AssertionError(f"PyTorch/ONNX Runtime parity failed: {comparisons}")
    return {
        "input_shape": list(sample.shape),
        "provider": "CPUExecutionProvider",
        "absolute_tolerance": atol,
        "outputs": comparisons,
        "parity_ok": True,
    }


def profile_pytorch_fp16(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (800, 1600),
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP16 profiling")
    model = prepare_export_model(
        wrapper, device=torch.device("cuda"), half=True
    )
    height, width = input_size
    sample = torch.zeros((1, 3, height, width), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(sample)
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000)
    values = np.asarray(durations)
    return {
        "runtime": "PyTorch CUDA FP16",
        "input_shape": list(sample.shape),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "includes_preprocessing": False,
        "includes_nms": False,
        "tensorrt": False,
    }
