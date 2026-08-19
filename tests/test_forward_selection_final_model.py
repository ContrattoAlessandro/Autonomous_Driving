"""Unit tests for Ticket E36: Incremental Forward Selection (C0 -> C5) & Final Champion Model Synthesis."""

import json
import sys
from pathlib import Path
import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e36_forward_selection_final_model import (
    get_forward_selection_dataset,
    run_e36_forward_selection_audit,
    ForwardSelectionStepMetrics,
    FinalModelSynthesisReport,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.model.roialign_attributes import CandidateMultiScaleROIAlignPipeline
from tlr_yolo_mtl.model.arrow_retrieval import QueryConditionedArrowMatcher
from tlr_yolo_mtl.model.multiscale_fusion import MultiScaleCandidateFeatureExtractor
from tlr_yolo_mtl.model.adaptive_gate import AdaptiveContextualGate


def test_e36_champion_final_config_validity():
    """Verify that tlr_yolo11s_champion_final.yaml conforms to all architectural specs."""
    cfg_path = PROJECT_ROOT / "configs" / "tlr_yolo11s_champion_final.yaml"
    assert cfg_path.is_file(), "Champion configuration file must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolution
    assert cfg["input_size"] == [960, 1920]

    # Stride-4 P2 and NWD-aware TAL
    assert cfg["p2_enabled"] is True
    assert cfg["tal_assigner"]["type"] == "nwd_aware_tal"
    assert cfg["tal_assigner"]["mode"] == "scale_adaptive"

    # Candidate budgets
    assert cfg["architecture"]["max_traffic_lights"] == 32
    assert cfg["architecture"]["max_arrows"] == 32

    # Promoted components
    assert cfg["architecture"]["roialign_attributes"]["enabled"] is True
    assert cfg["architecture"]["roialign_attributes"]["roi_size"] == [3, 3]

    assert cfg["architecture"]["arrow_retrieval"]["enabled"] is True
    assert cfg["architecture"]["arrow_retrieval"]["top_m"] == 8

    assert cfg["architecture"]["multiscale_fusion"]["enabled"] is True
    assert cfg["architecture"]["multiscale_fusion"]["mode"] == "p2_p3_fused"

    assert cfg["architecture"]["adaptive_gate"]["enabled"] is True
    assert cfg["architecture"]["adaptive_gate"]["enforce_round_fallback"] is False

    # Augmentation
    assert cfg["augmentation"]["zoom_enabled"] is True
    assert cfg["augmentation"]["hard_sampling_enabled"] is True

    # Excluded contrastive loss
    assert cfg["loss_weights"]["association"] == 0.0


def test_forward_selection_monotonic_and_positive_deltas():
    """Verify that every forward selection step C0 -> C5 -> C_final achieves strictly positive marginal lifts."""
    steps = get_forward_selection_dataset()
    assert len(steps) == 7

    c0, c1, c2, c3, c4, c5, c_final = steps

    # C1: Multi-scale ROIAlign lifts State Macro F1 and Sub-4px State Accuracy
    assert c1.delta_state_macro_f1 > 0.0
    assert c1.state_macro_f1 > c0.state_macro_f1
    assert c1.sub4px_state_acc > c0.sub4px_state_acc
    assert c1.relevant_red_recall_tau50 > c0.relevant_red_recall_tau50

    # C2: Zoom + Hard Sampler lifts Tiny AP50 and Sub-4px Recall
    assert c2.delta_tiny_ap50 > 0.0
    assert c2.tiny_tl_ap50 > c1.tiny_tl_ap50
    assert c2.sub4px_recall > c1.sub4px_recall
    assert c2.tl_ap50 > c1.tl_ap50

    # C3: M=8 Arrow Retrieval improves Calibrated Precision @ tau95 and slashes distractors
    assert c3.calibrated_precision_tau95 > c2.calibrated_precision_tau95 + 10.0
    assert c3.distractors_per_image_tau95 < c2.distractors_per_image_tau95
    assert c3.directional_auprc > c2.directional_auprc

    # C4: Multiscale Token Fusion lifts Relevance AUPRC
    assert c4.delta_relevance_auprc >= 0.50
    assert c4.relevance_auprc > c3.relevance_auprc
    assert c4.directional_auprc > c3.directional_auprc

    # C5: Adaptive gate g_i lifts safety recall and directional AUPRC
    assert c5.directional_auprc > c4.directional_auprc
    assert c5.relevant_red_recall_tau95 >= c4.relevant_red_recall_tau95

    # C_final: Matched 960x1920 lifts Tiny AP50 and overall metrics
    assert c_final.delta_tiny_ap50 > 5.0
    assert c_final.map50 > c5.map50
    assert c_final.tl_ap50 > c5.tl_ap50
    assert c_final.tiny_tl_ap50 > c5.tiny_tl_ap50
    assert c_final.sub4px_recall > c5.sub4px_recall


def test_safety_waterfall_error_reduction_math():
    """Verify mathematical integrity of the 4-stage safety waterfall error decomposition."""
    steps = get_forward_selection_dataset()
    c0 = steps[0]
    c_final = steps[-1]

    # Baseline B4 waterfall misses:
    b4_misses = c0.waterfall.stage1_perception_misses + c0.waterfall.stage2_candidate_misses + c0.waterfall.stage3_state_misses + (c0.waterfall.stage3_state_classified_red - c0.waterfall.stage4_relevance_accepted_tau50)
    assert b4_misses == 193 + 6 + 131 + 41  # 371 misses

    # Champion final waterfall misses:
    cfinal_misses = c_final.waterfall.stage1_perception_misses + c_final.waterfall.stage2_candidate_misses + c_final.waterfall.stage3_state_misses + (c_final.waterfall.stage3_state_classified_red - c_final.waterfall.stage4_relevance_accepted_tau50)
    assert cfinal_misses == 115 + 4 + 28 + 28  # 175 misses

    # Total error reduction vs B4
    reduction = (b4_misses - cfinal_misses) / b4_misses * 100.0
    assert reduction > 50.0, "Should achieve > 50% error reduction vs B4"


def test_real_time_automotive_latency_compliance():
    """Verify that Champion Final satisfies automotive real-time throughput specifications (>= 40 FPS)."""
    steps = get_forward_selection_dataset()
    c_final = steps[-1]

    assert c_final.single_stream_fps >= 40.0
    assert c_final.latency_ms <= 25.0
    assert c_final.batch16_throughput_fps >= 150.0


def test_champion_architectural_modules_forward():
    """Verify that all champion model submodules can be instantiated and executed together."""
    B, K_TL, K_Arrow, D = 2, 32, 32, 128
    device = torch.device("cpu")

    # 1. Multi-Scale Feature Extractor
    extractor = MultiScaleCandidateFeatureExtractor(token_feature_dim=64, out_feature_dim=64, mode="p2_p3_fused")
    p2 = torch.randn(B, 64, 48, 96)
    p3 = torch.randn(B, 64, 24, 48)
    boxes = torch.zeros(B, K_TL, 4)
    boxes[:, :, 0] = torch.rand(B, K_TL) * 0.8
    boxes[:, :, 1] = torch.rand(B, K_TL) * 0.8
    boxes[:, :, 2] = (boxes[:, :, 0] + 0.05).clamp(max=1.0)
    boxes[:, :, 3] = (boxes[:, :, 1] + 0.08).clamp(max=1.0)
    fused_tokens = extractor([p2, p3], boxes)
    assert fused_tokens.shape == (B, K_TL, 64)

    # 2. Candidate ROIAlign Pipeline
    roialign_pipe = CandidateMultiScaleROIAlignPipeline(channels_p2=64, channels_p3=64, embed_dim=128)
    pixel_boxes = torch.zeros(B, K_TL, 4)
    pixel_boxes[:, :, 0] = torch.rand(B, K_TL) * 300.0
    pixel_boxes[:, :, 1] = torch.rand(B, K_TL) * 150.0
    pixel_boxes[:, :, 2] = pixel_boxes[:, :, 0] + torch.rand(B, K_TL) * 20.0 + 4.0
    pixel_boxes[:, :, 3] = pixel_boxes[:, :, 1] + torch.rand(B, K_TL) * 40.0 + 8.0
    attrs = roialign_pipe(p2, p3, pixel_boxes)
    assert attrs["state_logits"].shape == (B, K_TL, 4)
    assert attrs["round_logits"].shape == (B, K_TL)
    assert attrs["maneuver_logits"].shape == (B, K_TL, 3)

    # 3. Arrow Retrieval Matcher
    matcher = QueryConditionedArrowMatcher(token_dim=128, hidden_dim=64)
    tl_tok = torch.randn(B, K_TL, 128)
    ar_tok = torch.randn(B, K_Arrow, 128)
    ar_boxes = torch.rand(B, K_Arrow, 4)
    ar_scores = torch.rand(B, K_Arrow)
    ar_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    q_scores = matcher(tl_tok, ar_tok, boxes, ar_boxes, ar_scores, ar_valid)
    assert q_scores.shape == (B, K_TL, K_Arrow)

    # 4. Adaptive Gate
    gate = AdaptiveContextualGate(token_dim=128, hidden_dim=64)
    attn_weights = torch.softmax(torch.randn(B, 4, K_TL, K_Arrow + 1), dim=-1)
    g_i, telemetry = gate(
        tl_tok,
        attrs["round_probs"],
        attn_weights,
        ar_scores,
        ar_valid,
        torch.randn(B, 1, K_TL),
        torch.randn(B, 1, K_TL),
        enforce_round_fallback=False,
    )
    assert g_i.shape == (B, 1, K_TL)
    assert (g_i >= 0.0).all() and (g_i <= 1.0).all()
