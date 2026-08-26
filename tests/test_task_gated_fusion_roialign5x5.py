import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.multiscale_fusion import MultiScaleCandidateFeatureExtractor
from tlr_yolo_mtl.model.roialign_attributes import (
    CandidateAttributeTower,
    CandidateMultiScaleROIAlign,
    CandidateMultiScaleROIAlignPipeline,
    TaskSpecificAttributeTower,
    TaskSpecificGatedROIAlign,
    TaskSpecificROIAlignPipeline,
)


def test_task_specific_gated_roialign_initialization_and_shapes() -> None:
    """Verify TaskSpecificGatedROIAlign initialization, output dimensions, and task gate access."""
    B, K = 2, 32
    C_p2, C_p3 = 64, 128
    H_p2, W_p2 = 40, 80
    H_p3, W_p3 = 20, 40

    extractor = TaskSpecificGatedROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        state_roi_size=(5, 5),
        aux_roi_size=(3, 3),
        embed_dim=128,
        stride_p2=4.0,
        stride_p3=8.0,
        use_task_gating=True,
    )

    dummy_p2 = torch.randn(B, C_p2, H_p2, W_p2)
    dummy_p3 = torch.randn(B, C_p3, H_p3, W_p3)
    dummy_boxes = torch.tensor([
        [[10.0, 12.0, 26.0, 48.0], [50.0, 60.0, 70.0, 110.0]] * 16,
        [[5.0, 8.0, 15.0, 28.0], [100.0, 30.0, 120.0, 70.0]] * 16,
    ], dtype=torch.float32)

    out = extractor(dummy_p2, dummy_p3, dummy_boxes)

    assert "state_tokens" in out
    assert "round_tokens" in out
    assert "man_tokens" in out
    assert "candidate_tokens" in out

    assert out["state_tokens"].shape == (B, K, 128)
    assert out["round_tokens"].shape == (B, K, 128)
    assert out["man_tokens"].shape == (B, K, 128)
    assert out["candidate_tokens"].shape == (B, K, 128)

    gates = extractor.task_gates
    assert 0.0 < gates["state"] < 1.0
    assert 0.0 < gates["round"] < 1.0
    assert 0.0 < gates["maneuver"] < 1.0
    assert 0.0 < gates["relevance"] < 1.0
    # State should prioritize P2 (alpha > 0.6)
    assert gates["state"] > 0.6
    # Relevance should prioritize P3 (alpha < 0.45)
    assert gates["relevance"] < 0.45


def test_task_gates_gradient_flow() -> None:
    """Verify gradients backpropagate into all task gate parameters and input feature maps."""
    B, K = 2, 8
    C_p2, C_p3 = 32, 64
    H_p2, W_p2 = 20, 20
    H_p3, W_p3 = 10, 10

    extractor = TaskSpecificGatedROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        state_roi_size=(5, 5),
        aux_roi_size=(3, 3),
        embed_dim=64,
        use_task_gating=True,
    )

    p2 = torch.randn(B, C_p2, H_p2, W_p2, requires_grad=True)
    p3 = torch.randn(B, C_p3, H_p3, W_p3, requires_grad=True)
    boxes = torch.zeros(B, K, 4)
    boxes[:, :, :2] = 5.0
    boxes[:, :, 2:] = 25.0

    out = extractor(p2, p3, boxes)

    loss = (
        out["state_tokens"].sum()
        + out["round_tokens"].sum()
        + out["man_tokens"].sum()
        + out["candidate_tokens"].sum()
    )
    loss.backward()

    assert extractor.raw_gate_state.grad is not None
    assert extractor.raw_gate_round.grad is not None
    assert extractor.raw_gate_man.grad is not None
    assert extractor.raw_gate_rel.grad is not None
    assert p2.grad is not None
    assert p3.grad is not None
    assert torch.isfinite(extractor.raw_gate_state.grad).all()
    assert torch.isfinite(p2.grad).all()


def test_task_specific_attribute_tower() -> None:
    """Verify TaskSpecificAttributeTower receives decoupled tokens and produces valid logits."""
    B, K = 2, 16
    embed_dim = 128
    tower = TaskSpecificAttributeTower(
        embed_dim=embed_dim,
        num_states=4,
        num_maneuvers=3,
    )

    state_tokens = torch.randn(B, K, embed_dim)
    round_tokens = torch.randn(B, K, embed_dim)
    man_tokens = torch.randn(B, K, embed_dim)

    preds = tower(state_tokens, round_tokens, man_tokens)

    assert preds["state_logits"].shape == (B, K, 4)
    assert preds["round_logits"].shape == (B, K)
    assert preds["maneuver_logits"].shape == (B, K, 3)

    assert preds["state_probs"].shape == (B, K, 4)
    assert preds["round_probs"].shape == (B, K)
    assert preds["maneuver_probs"].shape == (B, K, 3)

    # Probabilities should sum to 1.0 along class dimension for state
    state_sum = preds["state_probs"].sum(dim=-1)
    assert torch.allclose(state_sum, torch.ones_like(state_sum), atol=1e-5)


def test_task_specific_roialign_pipeline_e2e() -> None:
    """Test full TaskSpecificROIAlignPipeline forward pass with normalized and pixel boxes."""
    B, K = 2, 12
    C_p2, C_p3 = 64, 128
    H, W = 160, 320

    pipeline = TaskSpecificROIAlignPipeline(
        channels_p2=C_p2,
        channels_p3=C_p3,
        state_roi_size=(5, 5),
        aux_roi_size=(3, 3),
        embed_dim=128,
        stride_p2=4.0,
        stride_p3=8.0,
    )

    p2 = torch.randn(B, C_p2, H // 4, W // 4)
    p3 = torch.randn(B, C_p3, H // 8, W // 8)

    # Normalized boxes in [0, 1]
    norm_boxes = torch.zeros(B, K, 4)
    norm_boxes[:, :, 0] = 0.1
    norm_boxes[:, :, 1] = 0.1
    norm_boxes[:, :, 2] = 0.3
    norm_boxes[:, :, 3] = 0.4

    out = pipeline(p2, p3, norm_boxes, img_shape=(H, W))

    assert "state_logits" in out
    assert "round_logits" in out
    assert "maneuver_logits" in out
    assert "candidate_tokens" in out
    assert "state_tokens" in out
    assert "task_gates" in out
    assert out["state_logits"].shape == (B, K, 4)
    assert out["candidate_tokens"].shape == (B, K, 128)


def test_fp16_and_device_compatibility() -> None:
    """Ensure TaskSpecificROIAlignPipeline runs in FP16 precision without NaN/Inf."""
    B, K = 1, 16
    C_p2, C_p3 = 64, 128

    pipeline = TaskSpecificROIAlignPipeline(
        channels_p2=C_p2,
        channels_p3=C_p3,
        state_roi_size=(5, 5),
        aux_roi_size=(3, 3),
        embed_dim=128,
    ).half()

    p2 = torch.randn(B, C_p2, 30, 60, dtype=torch.float16)
    p3 = torch.randn(B, C_p3, 15, 30, dtype=torch.float16)
    boxes = torch.tensor([[[5.0, 5.0, 20.0, 35.0]] * K], dtype=torch.float16)

    with torch.no_grad():
        out = pipeline(p2, p3, boxes)

    assert torch.isfinite(out["state_logits"]).all()
    assert torch.isfinite(out["candidate_tokens"]).all()
    assert out["state_logits"].dtype == torch.float16


def test_multiscale_candidate_feature_extractor_task_gated_mode() -> None:
    """Verify task_gated_p2_p3 mode in MultiScaleCandidateFeatureExtractor."""
    B, K = 2, 10
    C_tok = 64
    extractor = MultiScaleCandidateFeatureExtractor(
        token_feature_dim=C_tok,
        out_feature_dim=C_tok,
        mode="task_gated_p2_p3",
    )

    p2 = torch.randn(B, C_tok, 40, 80)
    p3 = torch.randn(B, C_tok, 20, 40)
    boxes = torch.rand(B, K, 4)

    fused = extractor([p2, p3], boxes)
    assert fused.shape == (B, K, C_tok)

    loss = fused.sum()
    loss.backward()
    assert extractor.gate_param.grad is not None
    assert torch.isfinite(extractor.gate_param.grad).all()


def test_backward_compatibility_candidate_roialign() -> None:
    """Ensure standard CandidateMultiScaleROIAlignPipeline preserves backward compatibility."""
    B, K = 2, 8
    C_p2, C_p3 = 64, 128
    pipe = CandidateMultiScaleROIAlignPipeline(
        channels_p2=C_p2,
        channels_p3=C_p3,
        roi_size=(3, 3),
        embed_dim=128,
    )
    p2 = torch.randn(B, C_p2, 20, 40)
    p3 = torch.randn(B, C_p3, 10, 20)
    boxes = torch.zeros(B, K, 4)
    boxes[:, :, :2] = 10.0
    boxes[:, :, 2:] = 30.0

    out = pipe(p2, p3, boxes)
    assert out["state_logits"].shape == (B, K, 4)
    assert out["candidate_tokens"].shape == (B, K, 128)
