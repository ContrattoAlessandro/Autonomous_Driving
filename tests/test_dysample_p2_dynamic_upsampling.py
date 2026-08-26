"""Unit and integration tests for Ticket E40: DySample Dynamic Upsampling in the P3 -> P2 Lateral Path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e40_dysample_dynamic_upsampling import (
    DynamicUpsamplerMetrics,
    benchmark_module_fp16,
    format_e40_markdown_report,
    run_e40_dysample_audit,
)
from tlr_yolo_mtl.model.dysample import (
    CARAFE,
    BilinearUpsample,
    DySample,
    replace_p2_upsampler,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model, load_coco_warmstart
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
)


def test_dysample_forward_and_backward():
    """Verify DySample forward shapes, gradient backward flow, and styles."""
    B, C, H, W = 2, 64, 16, 32

    # Style: 'lp' (Linear Projection)
    dysample_lp = DySample(in_channels=C, scale=2, style="lp", groups=4, dyscope=False)
    x = torch.randn(B, C, H, W, requires_grad=True)
    out_lp = dysample_lp(x)

    assert out_lp.shape == (B, C, 2 * H, 2 * W)
    loss = out_lp.sum()
    loss.backward()
    assert x.grad is not None
    assert dysample_lp.offset_conv.weight.grad is not None

    # Style: 'pl' (Pointwise-Linear)
    dysample_pl = DySample(in_channels=C, scale=2, style="pl", groups=4, dyscope=True)
    x2 = torch.randn(B, C, H, W, requires_grad=True)
    out_pl = dysample_pl(x2)
    assert out_pl.shape == (B, C, 2 * H, 2 * W)
    loss2 = out_pl.sum()
    loss2.backward()
    assert x2.grad is not None


def test_carafe_and_bilinear_forward():
    """Verify CARAFE and BilinearUpsample modules."""
    B, C, H, W = 2, 64, 16, 32

    # CARAFE
    carafe = CARAFE(in_channels=C, scale=2, k_up=5, k_enc=3)
    x = torch.randn(B, C, H, W, requires_grad=True)
    out_carafe = carafe(x)
    assert out_carafe.shape == (B, C, 2 * H, 2 * W)
    loss = out_carafe.sum()
    loss.backward()
    assert x.grad is not None

    # BilinearUpsample
    bilinear = BilinearUpsample(scale=2)
    out_bilinear = bilinear(x)
    assert out_bilinear.shape == (B, C, 2 * H, 2 * W)


def test_replace_p2_upsampler_surgery():
    """Verify programmatic replacement of lateral P3 -> P2 upsampling layer."""
    cfg_path = PROJECT_ROOT / "configs" / "model" / "tlr_yolo11n_p2.yaml"
    wrapper = build_detection_model(cfg_path)

    # Initial layer 17 is nn.Upsample
    orig_layer = wrapper.model.model[17]
    assert isinstance(orig_layer, torch.nn.Upsample)

    # Swap to DySample
    dy_mod = replace_p2_upsampler(wrapper, mode="dysample", layer_index=17, groups=4)
    assert isinstance(wrapper.model.model[17], DySample)
    assert wrapper.model.model[17] is dy_mod

    # Test forward pass with swapped upsampler
    wrapper.model.eval()
    x = torch.randn(1, 3, 192, 320)
    with torch.no_grad():
        out, _ = wrapper.model(x)
    assert out.shape[0] == 1


def test_yolo11s_p2_dysample_yaml_model():
    """Verify that tlr_yolo11s_p2_dysample.yaml builds cleanly and loads COCO warmstart."""
    cfg_path = PROJECT_ROOT / "configs" / "model" / "tlr_yolo11s_p2_dysample.yaml"
    assert cfg_path.is_file()

    wrapper = build_detection_model(cfg_path)
    detect = wrapper.model.model[-1]
    strides = tuple(int(v) for v in detect.stride.tolist())
    assert strides == (4, 8, 16, 32)
    assert isinstance(wrapper.model.model[17], DySample)

    # Attach unified relevance head
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(max_traffic_lights=32, max_arrows=32))
    head = wrapper.model.model[-1]
    assert isinstance(head, UnifiedTrafficControlDetect)

    # Forward smoke
    wrapper.model.eval()
    sample = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        decoded, raw = wrapper.model(sample)
    assert decoded.shape == (1, 6, 20400)
    assert raw["state_logits"].shape == (1, 4, 20400)
    assert raw["relevance_logits"].shape == (1, 1, 32)


def test_yolo11s_p2_carafe_yaml_model():
    """Verify that tlr_yolo11s_p2_carafe.yaml builds cleanly."""
    cfg_path = PROJECT_ROOT / "configs" / "model" / "tlr_yolo11s_p2_carafe.yaml"
    assert cfg_path.is_file()

    wrapper = build_detection_model(cfg_path)
    detect = wrapper.model.model[-1]
    strides = tuple(int(v) for v in detect.stride.tolist())
    assert strides == (4, 8, 16, 32)
    assert isinstance(wrapper.model.model[17], CARAFE)


def test_audit_e40_execution_and_acceptance_criteria(tmp_path: Path):
    """Verify that E40 audit executes cleanly and passes all success criteria."""
    telemetry = run_e40_dysample_audit(output_dir=tmp_path, device="cuda")

    assert (tmp_path / "audit_e40_telemetry.json").is_file()
    assert (tmp_path / "audit_e40_summary.md").is_file()

    criteria = telemetry["acceptance_criteria"]
    assert criteria["delta_sub8px_ap50_ge_1_5pct"] is True
    assert criteria["delta_sub4px_recall_ge_2_5pct"] is True
    assert criteria["inference_overhead_le_0_8ms"] is True
    assert criteria["pareto_superiority_over_carafe"] is True

    deltas = telemetry["deltas_dysample_vs_baseline"]
    assert deltas["delta_ap_tl_sub8px"] >= 1.50
    assert deltas["delta_sub4px_recall"] >= 2.50
    assert deltas["delta_e2e_latency_ms"] <= 0.80
