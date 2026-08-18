"""YOLO11 P3-P5 construction, COCO warm-start, and structural smoke tests.

Torch and Ultralytics are imported lazily so dataset conversion and QA remain
usable in lightweight environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model" / "tlr_yolo11n.yaml"
INPUT_SIZE = (800, 1600)  # height, width
EXPECTED_STRIDES = (8, 16, 32)

# Every supported scale preserves the official YOLO11 P3-P5 layer layout. COCO
# tensors therefore transfer without index remapping; only the 80-class output
# convolutions are shape-incompatible with the two-type target.
TARGET_TO_SOURCE_LAYER = {index: index for index in range(24)}


def configure_ultralytics(project_root: Path = PROJECT_ROOT) -> Path:
    """Keep Ultralytics settings inside the project instead of user AppData."""

    settings_dir = project_root / ".ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir.resolve()))
    return settings_dir


def _runtime() -> tuple[Any, Any, str]:
    configure_ultralytics()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "Milestone 2 requires torch and ultralytics. Use the project .venv."
        ) from exc
    return torch, YOLO, ultralytics.__version__


def build_detection_model(config: str | Path = DEFAULT_CONFIG) -> Any:
    """Build the two-type standard YOLO11 P3-P5 detection model."""

    _, YOLO, _ = _runtime()
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    wrapper = YOLO(str(config_path), task="detect", verbose=False)
    detect = wrapper.model.model[-1]
    strides = tuple(int(value) for value in detect.stride.tolist())
    if strides != EXPECTED_STRIDES:
        raise ValueError(f"unexpected detection strides: {strides}")
    if int(detect.nc) != 2 or int(detect.reg_max) != 16:
        raise ValueError(
            f"unexpected Detect configuration: nc={detect.nc}, reg_max={detect.reg_max}"
        )
    return wrapper


def _source_key_for_target(target_key: str, max_layers: int = 30) -> str | None:
    """Map a target state key to the same layer in official YOLO models."""

    fields = target_key.split(".")
    if len(fields) < 3 or fields[0] != "model":
        return None
    try:
        target_layer = int(fields[1])
    except ValueError:
        return None

    source_layer = TARGET_TO_SOURCE_LAYER.get(target_layer, target_layer)
    if source_layer is None or source_layer >= max_layers:
        return None
    return ".".join(("model", str(source_layer), *fields[2:]))


def load_coco_warmstart(target: Any, weights: str | Path) -> dict[str, Any]:
    """Transfer every shape-compatible YOLO COCO tensor into the model."""

    torch, YOLO, _ = _runtime()
    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"COCO weights do not exist: {weights_path}")
    source = YOLO(str(weights_path), task="detect", verbose=False).model
    target_model = target.model if hasattr(target, "model") else target
    source_state = source.state_dict()
    target_state = target_model.state_dict()

    max_layers = max(len(target_model.model), len(source.model))
    loaded: dict[str, Any] = {}
    loaded_from: dict[str, str] = {}
    for target_key, target_value in target_state.items():
        source_key = _source_key_for_target(target_key, max_layers=max_layers)
        if source_key is None:
            continue
        source_value = source_state.get(source_key)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        loaded[target_key] = source_value.detach().to(
            device=target_value.device, dtype=target_value.dtype
        )
        loaded_from[target_key] = source_key

    target_state.update(loaded)
    target_model.load_state_dict(target_state, strict=True)
    # YOLO.train() only forwards an already-built in-memory model to its
    # trainer when the wrapper originated from a checkpoint.  This model was
    # built from YAML, so mark the completed COCO transfer as a non-resumable
    # checkpoint source; otherwise Ultralytics silently rebuilds random
    # weights from the YAML at train start.
    trainer_reuses_in_memory_weights = False
    if hasattr(target, "ckpt"):
        target.ckpt = {"epoch": -1, "warmstart_source": str(weights_path)}
        target.ckpt_path = str(weights_path)
        trainer_reuses_in_memory_weights = True
    parameter_keys = dict(target_model.named_parameters())
    loaded_parameters = sum(
        int(target_state[key].numel()) for key in loaded if key in parameter_keys
    )
    total_parameters = sum(int(value.numel()) for value in target_model.parameters())
    region_counts = {
        "backbone": 0,
        "neck_p3_p5": 0,
        "detect_p3_p5": 0,
    }
    final_layer = len(target_model.model) - 1
    backbone_end = 10
    for idx, mod in enumerate(target_model.model):
        if "Upsample" in type(mod).__name__:
            backbone_end = idx - 1
            break
    for key in loaded:
        layer = int(key.split(".")[1])
        if layer <= backbone_end:
            region_counts["backbone"] += 1
        elif layer == final_layer:
            region_counts["detect_p3_p5"] += 1
        else:
            region_counts["neck_p3_p5"] += 1
    return {
        "weights": str(weights_path),
        "loaded_state_items": len(loaded),
        "loaded_parameters": loaded_parameters,
        "target_parameters": total_parameters,
        "loaded_fraction": loaded_parameters / total_parameters,
        "loaded_state_items_by_region": region_counts,
        "pyramid_levels": ["P3", "P4", "P5"],
        "p2_enabled": False,
        "type_output_initialized_randomly": not any(
            key.endswith(".cv3.0.2.weight")
            or key.endswith(".cv3.1.2.weight")
            or key.endswith(".cv3.2.2.weight")
            for key in loaded
        ),
        "trainer_reuses_in_memory_weights": trainer_reuses_in_memory_weights,
        "mapping_examples": dict(list(loaded_from.items())[:12]),
    }


def _prediction_shape(output: Any, torch: Any) -> list[int] | None:
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    if torch.is_tensor(prediction):
        return list(prediction.shape)
    return None


def run_forward_smoke(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = INPUT_SIZE,
    device: str = "cuda",
    half: bool = True,
) -> dict[str, Any]:
    """Run one forward and verify P3-P5 shapes at the deployment resolution."""

    torch, _, ultralytics_version = _runtime()
    height, width = input_size
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("input height and width must be positive multiples of 32")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = wrapper.model.to(resolved_device).eval()
    use_half = half and resolved_device.type == "cuda"
    model = model.half() if use_half else model.float()
    detect = model.model[-1]
    captured: list[list[int]] = []

    def capture_features(_module: Any, args: tuple[Any, ...]) -> None:
        features = args[0]
        captured.extend(list(feature.shape) for feature in features)

    handle = detect.register_forward_pre_hook(capture_features)
    dtype = torch.float16 if use_half else torch.float32
    sample = torch.zeros((1, 3, height, width), device=resolved_device, dtype=dtype)
    peak_memory = None
    try:
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
        with torch.inference_mode():
            output = model(sample)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
            peak_memory = int(torch.cuda.max_memory_allocated(resolved_device))
    finally:
        handle.remove()

    strides = tuple(int(value) for value in detect.stride.tolist())
    expected_shapes = [
        [1, captured[index][1], height // stride, width // stride]
        for index, stride in enumerate(EXPECTED_STRIDES)
    ] if len(captured) == len(EXPECTED_STRIDES) else []
    if strides != EXPECTED_STRIDES:
        raise AssertionError(f"expected strides {EXPECTED_STRIDES}, got {strides}")
    if captured != expected_shapes:
        raise AssertionError(
            f"unexpected pyramid shapes: captured={captured}, expected={expected_shapes}"
        )

    expected_locations = sum(
        (height // stride) * (width // stride) for stride in EXPECTED_STRIDES
    )
    prediction_shape = _prediction_shape(output, torch)
    if prediction_shape is not None and prediction_shape[-1] != expected_locations:
        raise AssertionError(
            f"expected {expected_locations} dense locations, got {prediction_shape}"
        )

    return {
        "schema": "TLR-YOLO-MTL Milestone 2 smoke v2 (P3-P5)",
        "config": str(DEFAULT_CONFIG.resolve()),
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
        "device": str(resolved_device),
        "device_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else "CPU"
        ),
        "dtype": str(dtype).removeprefix("torch."),
        "input_shape": [1, 3, height, width],
        "strides": list(strides),
        "feature_shapes": captured,
        "prediction_shape": prediction_shape,
        "dense_locations": expected_locations,
        "parameters": sum(int(parameter.numel()) for parameter in model.parameters()),
        "peak_memory_bytes": peak_memory,
    }


def export_detection_onnx(
    wrapper: Any,
    output: str | Path,
    *,
    input_size: tuple[int, int] = INPUT_SIZE,
    device: str = "0",
    fp16: bool = True,
    opset: int = 17,
) -> dict[str, Any]:
    """Export fixed-shape detection-only ONNX and validate its graph contract."""

    _runtime()
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError("ONNX export validation requires the onnx package") from exc

    height, width = input_size
    if height % 32 or width % 32:
        raise ValueError("ONNX input height and width must be multiples of 32")
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("ONNX output path must end in .onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics derives the export path from model.pt_path.  It only uses the
    # stem and does not require the corresponding .pt file to exist.
    wrapper.model.pt_path = str(output_path.with_suffix(".pt"))
    exported = Path(
        wrapper.export(
            format="onnx",
            imgsz=[height, width],
            batch=1,
            dynamic=False,
            simplify=False,
            opset=opset,
            device=device,
            quantize=16 if fp16 else 32,
            nms=False,
            verbose=False,
        )
    ).resolve()
    if exported != output_path:
        raise RuntimeError(f"exported unexpected path: {exported} (wanted {output_path})")

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    input_dims = [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim]
    output_dims = [
        dimension.dim_value
        for dimension in graph.graph.output[0].type.tensor_type.shape.dim
    ]
    expected_locations = sum(
        (height // stride) * (width // stride) for stride in EXPECTED_STRIDES
    )
    if input_dims != [1, 3, height, width]:
        raise AssertionError(f"unexpected ONNX input shape: {input_dims}")
    if output_dims != [1, 6, expected_locations]:
        raise AssertionError(f"unexpected ONNX output shape: {output_dims}")

    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "opset": opset,
        "fp16": fp16,
        "input_shape": input_dims,
        "output_shape": output_dims,
        "nodes": len(graph.graph.node),
        "onnx_checker_ok": True,
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
