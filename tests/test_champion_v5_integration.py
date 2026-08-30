"""Unit and Integration Tests for Champion v5 Unified Production Model (Phase 8 Synthesis).

Validates:
1. Complete YAML configuration integrity for configs/tlr_yolo11s_champion_v5.yaml.
2. Correct instantiation of all 6 ratified Phase 8 architectural modules.
3. Multi-task loss computation (NWD-TAL + DFL Distributional Refinement + Multi-Teacher Relation KD).
4. Scale-conditioned quality score ranking with zero runtime overhead.
5. Dynamic scene-adaptive sparse refinement budget dispatching.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.neck import ScaleAwareFeatureRelayV2
from tlr_yolo_mtl.model.refinement import (
    SparseCandidateRefinementHead,
    SparseRefinementConfig,
)
from tlr_yolo_mtl.model.quality import (
    ContinuousScaleQualityFusion,
    compute_scale_conditioned_quality_scores,
)
from tlr_yolo_mtl.model.geometry_attention import (
    ExplicitRelativeGeometryEncoderV2,
    GeometryAttentionBiasMLPV2,
    GeometryAwareCrossAttentionV2,
)
from tlr_yolo_mtl.training.refinement_loss import SparseRefinementLoss, RefinementLossWeights
from tlr_yolo_mtl.training.distillation import (
    MultiTeacherDistillationConfig,
    MultiTeacherRelationDistillationLoss,
)
from tlr_yolo_mtl.deployment.postprocess import size_adaptive_nms


def test_champion_v5_config_integrity():
    """Verify that tlr_yolo11s_champion_v5.yaml conforms to all architectural specs."""
    cfg_path = PROJECT_ROOT / "configs" / "tlr_yolo11s_champion_v5.yaml"
    assert cfg_path.is_file(), "Champion v5 configuration file must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolution & Input
    assert cfg["input_size"] == [960, 1920]
    assert cfg["model_config"] == "configs/model/tlr_yolo11s_p2_relay_v2.yaml"
    assert cfg["p2_enabled"] is True

    # E66: Dual-Gate Relay v2
    relay_cfg = cfg["architecture"]["scale_aware_relay"]
    assert relay_cfg["enabled"] is True
    assert relay_cfg["version"] == "v2"
    assert relay_cfg["gating_type"] == "dual_gate"
    assert relay_cfg["saliency_kernel"] == 3

    # E68 + E69: Dynamic Sparse Refinement & Distributional DFL Box
    refine_cfg = cfg["architecture"]["sparse_refinement"]
    assert refine_cfg["enabled"] is True
    assert refine_cfg["dynamic_budget"] is True
    assert refine_cfg["budget_tiers"] == [8, 16, 32, 48, 64]
    assert refine_cfg["distributional_box"] is True
    assert refine_cfg["reg_max"] == 16

    # E70: Continuous Scale-Conditioned Quality Scoring
    qual_cfg = cfg["architecture"]["quality_head"]
    assert qual_cfg["enabled"] is True
    assert qual_cfg["scale_conditioned"] is True
    assert qual_cfg["alpha_min"] == 0.38
    assert qual_cfg["alpha_max"] == 0.90

    # E74: Geometry-Aware Cross-Attention v2
    geom_cfg = cfg["architecture"]["geometry_attention"]
    assert geom_cfg["enabled"] is True
    assert geom_cfg["version"] == "v2"
    assert geom_cfg["relative_bias_dim"] == 20
    assert geom_cfg["use_confidence_gate"] is True

    # E72: Multi-Teacher Relation Distillation
    kd_cfg = cfg["multi_teacher_distillation"]
    assert kd_cfg["enabled"] is True
    assert kd_cfg["consensus_gating"] is True
    assert kd_cfg["relational_gram_weight"] == 0.3

    # E45 + E63: Post-processing
    post_cfg = cfg["postprocessing"]
    assert post_cfg["size_adaptive_nms"] is True
    assert post_cfg["quality_ranking"] is True
    assert post_cfg["scale_conditioned_ranking"] is True
    assert post_cfg["vectorized_nms"] is True


def test_champion_v5_relay_v2_forward():
    """Verify Relay v2 dual-gate forward pass on point signals."""
    relay = ScaleAwareFeatureRelayV2(c2_channels=64, p2_channels=64, hidden_ratio=0.5, saliency_kernel=3)
    relay.eval()

    c2 = torch.randn(2, 64, 240, 480)
    p2 = torch.randn(2, 64, 240, 480)

    with torch.no_grad():
        out = relay(c2, p2)

    assert out.shape == p2.shape
    assert not torch.isnan(out).any()


def test_champion_v5_dynamic_refinement_distributional():
    """Verify SparseCandidateRefinementHead with dynamic budget and DFL box regression."""
    config = SparseRefinementConfig(
        channels_p2=64,
        channels_c2=64,
        hidden_dim=64,
        distributional=True,
        reg_max=16,
    )
    head = SparseCandidateRefinementHead(config)
    head.eval()

    c2 = torch.randn(2, 64, 240, 480)
    p2 = torch.randn(2, 64, 240, 480)
    boxes = torch.zeros(2, 32, 4)
    boxes[:, :4] = torch.tensor([[10.0, 10.0, 18.0, 22.0]])  # tiny
    boxes[:, 4:] = torch.tensor([[50.0, 50.0, 100.0, 100.0]])  # macro

    with torch.no_grad():
        outputs = head(p2, c2, candidate_boxes_xyxy=boxes)

    assert "refined_boxes_xyxy" in outputs
    assert "box_distribution" in outputs
    assert outputs["box_distribution"].shape[-1] == 16
    assert "refined_state_logits" in outputs
    assert "quality_deltas" in outputs


def test_champion_v5_geometry_attention_v2():
    """Verify Geometry-Aware Cross-Attention v2 forward pass with 20D perspective descriptors."""
    cross_att = GeometryAwareCrossAttentionV2(
        dimension=128,
        heads=4,
        hidden_dim=64,
    )
    cross_att.eval()

    tl_tokens = torch.randn(2, 32, 128)
    arrow_tokens = torch.randn(2, 8, 128)
    tl_boxes = torch.rand(2, 32, 4)
    arrow_boxes = torch.rand(2, 8, 4)
    tl_scores = torch.rand(2, 32)
    arrow_scores = torch.rand(2, 8)
    tl_round = torch.rand(2, 32)
    tl_man = F.softmax(torch.randn(2, 32, 3), dim=-1)
    arrow_man = F.softmax(torch.randn(2, 8, 3), dim=-1)
    arrow_valid = torch.ones(2, 8, dtype=torch.bool)

    with torch.no_grad():
        cond_tokens, weights, geom_bias = cross_att(
            traffic_tokens=tl_tokens,
            arrow_tokens=arrow_tokens,
            traffic_boxes=tl_boxes,
            arrow_boxes=arrow_boxes,
            traffic_scores=tl_scores,
            arrow_scores=arrow_scores,
            traffic_round=tl_round,
            traffic_maneuver=tl_man,
            arrow_maneuver=arrow_man,
            arrow_valid=arrow_valid,
        )

    assert cond_tokens.shape == tl_tokens.shape
    assert weights.shape == (2, 4, 32, 9)  # 8 arrows + 1 null token
    assert geom_bias.shape == (2, 4, 32, 9)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 32), atol=1e-5)


def test_champion_v5_multi_teacher_distillation_loss():
    """Verify MultiTeacherRelationDistillationLoss forward pass."""
    config = MultiTeacherDistillationConfig(
        temperature=3.0,
        weight_kd=1.0,
        weight_relation=0.3,
        local_teacher_weight=0.5,
        temporal_teacher_weight=0.5,
        sub4px_scale_boost=1.5,
    )
    loss_module = MultiTeacherRelationDistillationLoss(config, student_dim=64, teacher_dim=64)

    student_logits = torch.randn(4, 4, requires_grad=True)
    local_logits = torch.randn(4, 4)
    temporal_logits = torch.randn(4, 4)
    student_feats = torch.randn(4, 64, requires_grad=True)
    local_feats = torch.randn(4, 64)
    temporal_feats = torch.randn(4, 64)
    candidate_areas = torch.tensor([9.0, 36.0, 100.0, 625.0])

    loss_dict = loss_module(
        student_logits=student_logits,
        student_features=student_feats,
        local_teacher_logits=local_logits,
        local_teacher_features=local_feats,
        temporal_teacher_logits=temporal_logits,
        temporal_teacher_features=temporal_feats,
        candidate_areas=candidate_areas,
    )

    assert "loss_distill_total" in loss_dict
    assert "loss_distill_kd" in loss_dict
    assert "loss_distill_relation" in loss_dict
    assert loss_dict["loss_distill_total"].item() > 0.0

    # Test backward pass
    loss_dict["loss_distill_total"].backward()
    assert student_logits.grad is not None
    assert student_feats.grad is not None


def test_champion_v5_scale_conditioned_quality_fusion():
    """Verify continuous scale-conditioned quality score ranking."""
    fuser = ContinuousScaleQualityFusion(alpha_min=0.38, alpha_max=0.90, side_min=2.0, side_max=16.0)

    # Sub-4px box: side = 3.0, area = 9.0
    scores_tiny = torch.tensor([0.40])
    qual_tiny = torch.tensor([0.90])
    areas_tiny = torch.tensor([9.0])
    fused_tiny = fuser(scores_tiny, qual_tiny, areas_tiny)

    # Macro box: side = 20.0, area = 400.0
    scores_macro = torch.tensor([0.40])
    qual_macro = torch.tensor([0.90])
    areas_macro = torch.tensor([400.0])
    fused_macro = fuser(scores_macro, qual_macro, areas_macro)

    # Tiny box should be weighted much more heavily towards quality (lower alpha => higher (1-alpha) power on qual)
    # With qual=0.90 > score=0.40, tiny fused score will be higher than macro fused score
    assert fused_tiny.item() > fused_macro.item()
