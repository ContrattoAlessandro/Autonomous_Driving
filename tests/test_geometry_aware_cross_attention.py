import sys
from pathlib import Path
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.geometry_attention import (
    ExplicitRelativeGeometryEncoder,
    GeometryAttentionBiasMLP,
    GeometryAwareCrossAttention,
    GeometryAwareUnifiedDetect,
    attach_geometry_aware_unified_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_explicit_relative_geometry_encoder_shapes():
    B, K_TL, K_Arrow = 2, 8, 12
    encoder = ExplicitRelativeGeometryEncoder(ego_x=0.5, p_drop=0.0)

    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_maneuver = torch.rand(B, K_TL, 3)
    arrow_maneuver = torch.rand(B, K_Arrow, 3)
    tl_valid = torch.ones(B, K_TL, dtype=torch.bool)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    phi = encoder(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_maneuver,
        arrow_maneuver=arrow_maneuver,
        tl_valid=tl_valid,
        arrow_valid=arrow_valid,
    )

    # Default 18D Vanishing Point descriptor
    assert phi.shape == (B, K_TL, K_Arrow, 18)
    assert not torch.isnan(phi).any()
    assert not torch.isinf(phi).any()

    # Legacy 14D descriptor mode check
    encoder_14d = ExplicitRelativeGeometryEncoder(ego_x=0.5, include_vanishing_point=False, p_drop=0.0)
    phi_14d = encoder_14d(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_maneuver,
        arrow_maneuver=arrow_maneuver,
        tl_valid=tl_valid,
        arrow_valid=arrow_valid,
    )
    assert phi_14d.shape == (B, K_TL, K_Arrow, 14)


def test_geometry_attention_bias_mlp_init():
    B, K_TL, K_Arrow, H = 2, 4, 6, 4
    mlp = GeometryAttentionBiasMLP(in_features=18, hidden_dim=32, heads=H)

    phi = torch.randn(B, K_TL, K_Arrow, 18)
    bias = mlp(phi)

    assert bias.shape == (B, H, K_TL, K_Arrow)
    # Neutral zero initialization check
    assert torch.allclose(bias, torch.zeros_like(bias), atol=1e-6)


def test_geometry_aware_cross_attention_forward_and_backward():
    B, K_TL, K_Arrow, D, H = 2, 8, 16, 64, 4
    attn = GeometryAwareCrossAttention(
        dimension=D, heads=H, hidden_dim=32, use_confidence_gating=True
    )

    tl_tokens = torch.randn(B, K_TL, D, requires_grad=True)
    arrow_tokens = torch.randn(B, K_Arrow, D, requires_grad=True)
    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_man = torch.rand(B, K_TL, 3)
    ar_man = torch.rand(B, K_Arrow, 3)
    ar_valid = torch.ones(B, K_Arrow, dtype=torch.bool)
    ar_valid[:, 10:] = False  # test invalid masking

    conditioned, weights, bias = attn(
        traffic_tokens=tl_tokens,
        arrow_tokens=arrow_tokens,
        traffic_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        traffic_scores=tl_scores,
        arrow_scores=arrow_scores,
        traffic_round=tl_round,
        traffic_maneuver=tl_man,
        arrow_maneuver=ar_man,
        arrow_valid=ar_valid,
    )

    assert conditioned.shape == (B, K_TL, D)
    assert weights.shape == (B, H, K_TL, K_Arrow + 1)
    assert bias.shape == (B, H, K_TL, K_Arrow + 1)

    # Weights must sum to 1.0 along key dimension
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    # Invalid arrows must receive zero attention mass
    assert torch.all(weights[:, :, :, 10:K_Arrow] < 1e-6)

    # Backward gradient flow check
    loss = conditioned.sum()
    loss.backward()
    assert tl_tokens.grad is not None
    assert arrow_tokens.grad is not None
    assert attn.geometry_mlp.network[0].weight.grad is not None


def test_geometry_aware_cross_attention_fp16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for FP16 test")

    device = torch.device("cuda")
    B, K_TL, K_Arrow, D, H = 2, 8, 16, 64, 4
    attn = GeometryAwareCrossAttention(
        dimension=D, heads=H, hidden_dim=32, use_confidence_gating=True
    ).to(device).half()

    tl_tokens = torch.randn(B, K_TL, D, device=device, dtype=torch.float16)
    arrow_tokens = torch.randn(B, K_Arrow, D, device=device, dtype=torch.float16)
    tl_boxes = torch.rand(B, K_TL, 4, device=device, dtype=torch.float16)
    arrow_boxes = torch.rand(B, K_Arrow, 4, device=device, dtype=torch.float16)
    tl_scores = torch.rand(B, K_TL, device=device, dtype=torch.float16)
    arrow_scores = torch.rand(B, K_Arrow, device=device, dtype=torch.float16)
    tl_round = torch.rand(B, K_TL, device=device, dtype=torch.float16)
    tl_man = torch.rand(B, K_TL, 3, device=device, dtype=torch.float16)
    ar_man = torch.rand(B, K_Arrow, 3, device=device, dtype=torch.float16)
    ar_valid = torch.ones(B, K_Arrow, device=device, dtype=torch.bool)

    with torch.inference_mode():
        conditioned, weights, bias = attn(
            traffic_tokens=tl_tokens,
            arrow_tokens=arrow_tokens,
            traffic_boxes=tl_boxes,
            arrow_boxes=arrow_boxes,
            traffic_scores=tl_scores,
            arrow_scores=arrow_scores,
            traffic_round=tl_round,
            traffic_maneuver=tl_man,
            arrow_maneuver=ar_man,
            arrow_valid=ar_valid,
        )

    assert conditioned.dtype == torch.float16
    assert not torch.isnan(conditioned).any()


def test_attach_geometry_aware_unified_relevance_head():
    cfg_path = PROJECT_ROOT / "configs/model/tlr_yolo11n_p2.yaml"
    if not cfg_path.is_file():
        pytest.skip("Model config not found")

    wrapper = build_detection_model(cfg_path)
    head_config = UnifiedHeadConfig(
        token_dim=64,
        attention_heads=4,
        max_traffic_lights=16,
        max_arrows=16,
    )
    unified = attach_geometry_aware_unified_relevance_head(
        wrapper,
        config=head_config,
        hidden_dim=32,
        use_confidence_gating=True,
    )

    assert isinstance(unified, GeometryAwareUnifiedDetect)
    assert isinstance(unified.cross_attention, GeometryAwareCrossAttention)

    # Smoke forward pass on dummy image batch
    x = torch.randn(1, 3, 384, 640)
    out = wrapper.model(x)

    assert isinstance(out, dict)
    assert "relevance_logits" in out
    assert "attention_weights" in out
    assert "attention_geometry_bias" in out
    assert out["relevance_logits"].shape[0] == 1
    assert out["relevance_logits"].shape[1] == 1
    assert out["relevance_logits"].shape[2] == 16
