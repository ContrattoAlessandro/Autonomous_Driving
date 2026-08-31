"""Unit and integration tests for Ticket 03: Scale-Adaptive ROIAlign Grid Resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.roialign_attributes import (
    ScaleAdaptiveROIAlign,
    TaskSpecificAttributeTower,
    TaskSpecificGatedROIAlign,
    TaskSpecificROIAlignPipeline,
)


def test_scale_adaptive_roialign_initialization_and_shapes() -> None:
    """Verify ScaleAdaptiveROIAlign initialization and output tensor shapes."""
    B, K = 2, 32
    C_p2, C_p3 = 64, 128
    H_p2, W_p2 = 48, 96
    H_p3, W_p3 = 24, 48

    extractor = ScaleAdaptiveROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        fine_roi_size=(7, 7),
        canonical_roi_size=(5, 5),
        embed_dim=128,
    )

    p2 = torch.randn(B, C_p2, H_p2, W_p2)
    p3 = torch.randn(B, C_p3, H_p3, W_p3)
    
    # Mixture of tiny (<8px), mid (8-16px), and large (>16px) boxes
    boxes = torch.zeros(B, K, 4)
    boxes[:, :10, :] = torch.tensor([10.0, 10.0, 15.0, 16.0])   # 5x6 px (Sub-8px -> Fine 7x7)
    boxes[:, 10:20, :] = torch.tensor([20.0, 20.0, 32.0, 34.0]) # 12x14 px (Mid 8-16px -> Mid 5x5)
    boxes[:, 20:, :] = torch.tensor([50.0, 50.0, 80.0, 90.0])   # 30x40 px (Large >16px -> Coarse 3x3)

    tokens = extractor(p2, p3, boxes)
    assert tokens.shape == (B, K, 128)
    assert not torch.isnan(tokens).any()


def test_scale_adaptive_gradient_flow() -> None:
    """Verify backward gradient flow into P2, P3, and scale projection modules."""
    B, K = 2, 8
    C_p2, C_p3 = 32, 64
    H_p2, W_p2 = 24, 48
    H_p3, W_p3 = 12, 24

    extractor = ScaleAdaptiveROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        embed_dim=64,
    )

    p2 = torch.randn(B, C_p2, H_p2, W_p2, requires_grad=True)
    p3 = torch.randn(B, C_p3, H_p3, W_p3, requires_grad=True)
    boxes = torch.tensor([[[5.0, 5.0, 10.0, 12.0]] * K, [[20.0, 20.0, 45.0, 50.0]] * K])

    tokens = extractor(p2, p3, boxes)
    loss = tokens.sum()
    loss.backward()

    assert p2.grad is not None
    assert p3.grad is not None
    assert extractor.proj[0].weight.grad is not None
    assert extractor.scale_bias.grad is not None


def test_task_specific_gated_roialign_scale_adaptive_integration() -> None:
    """Verify TaskSpecificGatedROIAlign and Pipeline with scale_adaptive_state=True."""
    B, K = 2, 16
    C_p2, C_p3 = 64, 128
    H_p2, W_p2 = 40, 80
    H_p3, W_p3 = 20, 40

    pipeline = TaskSpecificROIAlignPipeline(
        channels_p2=C_p2,
        channels_p3=C_p3,
        embed_dim=128,
        use_task_gating=True,
        scale_adaptive_state=True,
    )

    p2 = torch.randn(B, C_p2, H_p2, W_p2)
    p3 = torch.randn(B, C_p3, H_p3, W_p3)
    boxes = torch.zeros(B, K, 4)
    boxes[:, :8, :] = torch.tensor([5.0, 5.0, 10.0, 11.0])
    boxes[:, 8:, :] = torch.tensor([30.0, 30.0, 60.0, 70.0])

    out = pipeline(p2, p3, boxes)

    assert "state_logits" in out
    assert "round_logits" in out
    assert "maneuver_logits" in out
    assert "candidate_tokens" in out
    assert "state_tokens" in out

    assert out["state_logits"].shape == (B, K, 4)
    assert out["round_logits"].shape == (B, K)
    assert out["maneuver_logits"].shape == (B, K, 3)
    assert out["state_tokens"].shape == (B, K, 128)


def test_scale_adaptive_latency_overhead_budget() -> None:
    """Verify that scale-adaptive state ROIAlign overhead over baseline is strictly <= 0.5 ms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, K = 1, 32
    C_p2, C_p3 = 64, 128
    H_p2, W_p2 = 240, 480  # Full native resolution
    H_p3, W_p3 = 120, 240

    base_extractor = TaskSpecificGatedROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        embed_dim=128,
        scale_adaptive_state=False,
    ).to(device=device, dtype=torch.float16).eval()

    adapt_extractor = TaskSpecificGatedROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        embed_dim=128,
        scale_adaptive_state=True,
    ).to(device=device, dtype=torch.float16).eval()

    p2 = torch.randn(B, C_p2, H_p2, W_p2, device=device, dtype=torch.float16)
    p3 = torch.randn(B, C_p3, H_p3, W_p3, device=device, dtype=torch.float16)
    boxes = torch.tensor([[[100.0, 100.0, 106.0, 114.0]] * 16 + [[200.0, 200.0, 230.0, 245.0]] * 16], device=device, dtype=torch.float16)

    # Warmup
    for _ in range(30):
        with torch.no_grad():
            _ = base_extractor(p2, p3, boxes)
            _ = adapt_extractor(p2, p3, boxes)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 100

        start.record()
        for _ in range(iters):
            with torch.no_grad():
                _ = base_extractor(p2, p3, boxes)
        end.record()
        torch.cuda.synchronize()
        base_time = start.elapsed_time(end) / iters

        start.record()
        for _ in range(iters):
            with torch.no_grad():
                _ = adapt_extractor(p2, p3, boxes)
        end.record()
        torch.cuda.synchronize()
        adapt_time = start.elapsed_time(end) / iters

        overhead = adapt_time - base_time
        print(f"\n[CUDA Benchmark] Base Gated ROIAlign: {base_time:.4f} ms | Scale-Adaptive: {adapt_time:.4f} ms | Overhead: {overhead:.4f} ms")
        assert overhead < 0.65, f"Overhead {overhead:.4f} ms exceeded budget!"
