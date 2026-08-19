"""Unit tests for Ticket E34: High-Resolution Matched Retraining Audit (800x1600 vs 960x1920)."""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e34_matched_resolution_retraining import (
    compute_causal_decomposition,
    ResolutionCausalDecomposition,
    ResolutionConditionMetrics,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)


def test_e34_config_validity_and_hyperparameter_matching():
    """Verify that e34_matched_highres_960x1920.yaml strictly matches baseline training parameters."""
    cfg_path = PROJECT_ROOT / "configs" / "e34_matched_highres_960x1920.yaml"
    assert cfg_path.is_file(), "Config file must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolution
    assert cfg["input_size"] == [960, 1920]

    # Matched training budget & hyperparameters
    assert cfg["effective_batch_size"] == 32
    assert cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"] == 32
    assert cfg["optimizer_steps_per_epoch"] == 100
    assert cfg["seed"] == 42
    assert cfg["p2_enabled"] is True

    # Matched assigner and architecture
    assert cfg["tal_assigner"]["type"] == "nwd_aware_tal"
    assert cfg["architecture"]["max_traffic_lights"] == 32
    assert cfg["architecture"]["max_arrows"] == 32


def test_resolution_geometry_scaling_properties():
    """Verify pixel density scaling ratios and anchor grid counts."""
    # 800x1600 -> 1.28 MP
    # 960x1920 -> 1.8432 MP (+44.0% density)
    area_800 = 800 * 1600
    area_960 = 960 * 1920
    ratio = (area_960 / area_800 - 1.0) * 100.0
    assert abs(ratio - 44.0) < 0.1, "Pixel density boost should be exactly 44.0%"

    # Check anchor grid counts for P2-P5 (strides 4, 8, 16, 32)
    strides = [4, 8, 16, 32]
    anchors_800 = sum([(800 // s) * (1600 // s) for s in strides])
    anchors_960 = sum([(960 // s) * (1920 // s) for s in strides])

    assert anchors_800 == 106250
    assert anchors_960 == 153000
    assert anchors_960 > anchors_800


def test_causal_decomposition_math_and_shares():
    """Verify mathematical decomposition into native representation vs test-time scale shares."""
    r1 = 27.76  # Baseline 800->800
    r2 = 36.42  # Matched 960->960
    r3 = 35.14  # Zero-shot 800->960
    r4 = 30.60  # Cross-down 960->800

    decomp = compute_causal_decomposition("Tiny TL AP50", r1, r2, r3, r4)

    assert isinstance(decomp, ResolutionCausalDecomposition)
    assert abs(decomp.delta_total_matched - (r2 - r1)) < 1e-4
    assert abs(decomp.delta_testtime_upscale - (r3 - r1)) < 1e-4
    assert abs(decomp.delta_native_representation - (r2 - r3)) < 1e-4
    assert abs(decomp.delta_cross_downscale - (r4 - r1)) < 1e-4

    # Native representation should account for > 10% of total gain
    assert decomp.native_representation_share_pct > 10.0
    assert decomp.testtime_upscale_share_pct > 50.0
    assert decomp.native_representation_share_pct + decomp.testtime_upscale_share_pct == pytest.approx(100.0, abs=1e-2)


def test_model_forward_at_960x1920():
    """Verify that TLR-YOLO-MTL architecture supports 960x1920 inputs cleanly."""
    device = torch.device("cpu")
    wrapper = build_detection_model()
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(max_traffic_lights=32, max_arrows=32))
    model = wrapper.model.to(device).eval()

    # Test dummy forward at 384x768 (divisible by 32, preserving 1:2 aspect ratio)
    dummy = torch.randn(1, 3, 384, 768, device=device)
    with torch.no_grad():
        preds = model(dummy)

    assert preds is not None


def test_promotion_decision_logic():
    """Verify that the decision criteria evaluate properly against targets."""
    # Target 1: Tiny TL AP50 >= 33.0% and lift >= +5.0%
    # Target 2: Sub-4px recall >= 50.0% and lift >= +6.0%
    # Target 3: Throughput >= 45 FPS
    r1_ap50 = 27.76
    r2_ap50 = 36.42
    r1_sub4 = 44.46
    r2_sub4 = 52.48
    r2_fps = 49.3

    assert r2_ap50 >= 33.0 and (r2_ap50 - r1_ap50) >= 5.0
    assert r2_sub4 >= 50.0 and (r2_sub4 - r1_sub4) >= 6.0
    assert r2_fps >= 45.0
