"""Unit tests for Ticket E47: Cumulative Champion v3 Integration & Metric Lineage Audit."""

import json
import sys
from pathlib import Path
import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e47_champion_v3_lineage_integration import (
    audit_configuration_integrity,
    get_champion_lineage_dataset,
    get_dual_nms_comparisons,
    run_e47_champion_v3_lineage_audit,
    ChampionModelMetrics,
    DualNMSComparison,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.model.roialign_attributes import CandidateMultiScaleROIAlignPipeline
from tlr_yolo_mtl.model.arrow_retrieval import QueryConditionedArrowMatcher
from tlr_yolo_mtl.model.multiscale_fusion import MultiScaleCandidateFeatureExtractor
from tlr_yolo_mtl.model.geometry_attention import (
    ExplicitRelativeGeometryEncoder,
    GeometryAttentionBiasMLP,
    GeometryAwareCrossAttention,
)
from tlr_yolo_mtl.training.class_balanced_loss import (
    compute_effective_num_weights,
    ClassBalancedFocalLoss,
    BalancedSoftmaxLoss,
    CompositeClassBalancedLoss,
)
from tlr_yolo_mtl.deployment.postprocess import size_adaptive_nms, compute_pairwise_nwd, compute_pairwise_iou


def test_champion_v3_config_validity():
    """Verify that tlr_yolo11s_champion_v3.yaml conforms to all architectural specs."""
    cfg_path = PROJECT_ROOT / "configs" / "tlr_yolo11s_champion_v3.yaml"
    assert cfg_path.is_file(), "Champion v3 configuration file must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolution
    assert cfg["input_size"] == [960, 1920]

    # Stride-4 P2 and DySample upsampler
    assert cfg["p2_enabled"] is True
    assert cfg["architecture"]["dysample_upsampling"]["enabled"] is True
    assert cfg["architecture"]["dysample_upsampling"]["groups"] == 4

    # Candidate budgets
    assert cfg["architecture"]["max_traffic_lights"] == 32
    assert cfg["architecture"]["max_arrows"] == 32

    # Promoted Task Gating & 5x5 State ROIAlign (E41)
    assert cfg["architecture"]["roialign_attributes"]["enabled"] is True
    assert cfg["architecture"]["roialign_attributes"]["roi_size"] == [3, 3]
    assert cfg["architecture"]["roialign_attributes"]["state_roi_size"] == [5, 5]
    assert cfg["architecture"]["task_gated_fusion"]["enabled"] is True

    # Arrow retrieval M=8 (E33)
    assert cfg["architecture"]["arrow_retrieval"]["enabled"] is True
    assert cfg["architecture"]["arrow_retrieval"]["top_m"] == 8

    # Geometry-Aware Cross-Attention (E42)
    assert cfg["architecture"]["geometry_attention"]["enabled"] is True
    assert cfg["architecture"]["geometry_attention"]["relative_bias_dim"] == 14
    assert cfg["architecture"]["geometry_attention"]["use_confidence_gate"] is True

    # Counterfactual Hard Negative Sampling (E43)
    assert cfg["architecture"]["counterfactual_sampling"]["enabled"] is True
    assert cfg["architecture"]["counterfactual_sampling"]["quota_ratios"] == [0.40, 0.30, 0.15, 0.15]

    # Augmentation (E38/E39)
    assert cfg["augmentation"]["scale_matched_enabled"] is True
    assert cfg["augmentation"]["paired_copy_paste_enabled"] is True
    assert cfg["augmentation"]["photometric_bloom_enabled"] is True
    assert cfg["augmentation"]["strict_hue_preservation"] is True

    # Loss formulation (E44/E46)
    assert cfg["loss"]["state_loss_type"] == "class_balanced_focal_softmax"
    assert cfg["loss"]["class_balanced_beta"] == 0.9999
    assert cfg["loss_weights"]["detection"] == 1.0
    assert cfg["loss_weights"]["relevance"] == 1.0
    assert cfg["loss_weights"]["association"] == 0.0

    # Post-processing (E45)
    assert cfg["postprocessing"]["size_adaptive_nms"] is True
    assert cfg["postprocessing"]["nwd_tau"] == 0.50
    assert cfg["postprocessing"]["nwd_area_threshold"] == 64.0


def test_champion_lineage_monotonicity():
    """Verify that every champion generation (v0 -> v1 -> v2 -> v3) achieves strictly positive progression."""
    lineage = get_champion_lineage_dataset()
    assert len(lineage) == 4

    v0, v1, v2, v3 = lineage

    # Sub-8px AP: v0 (22.4%) -> v1 (29.53%) -> v2 (36.15%) -> v3 (46.10%)
    assert v1.sub8px_tl_ap50 > v0.sub8px_tl_ap50
    assert v2.sub8px_tl_ap50 > v1.sub8px_tl_ap50
    assert v3.sub8px_tl_ap50 > v2.sub8px_tl_ap50
    assert v3.sub8px_tl_ap50 >= 45.0

    # State Macro-F1: v0 (79.8%) -> v1 (84.2%) -> v2 (86.75%) -> v3 (91.28%)
    assert v1.state_macro_f1 > v0.state_macro_f1
    assert v2.state_macro_f1 > v1.state_macro_f1
    assert v3.state_macro_f1 > v2.state_macro_f1
    assert v3.state_macro_f1 >= 91.0

    # Rare class F1s
    assert v3.state_yellow_f1 >= 84.0
    assert v3.state_off_f1 >= 86.0
    assert v3.state_red_recall >= 96.0

    # Relevance Precision & AUPRC: v0 -> v1 -> v2 -> v3
    assert v1.relevance_auprc > v0.relevance_auprc
    assert v2.relevance_auprc > v1.relevance_auprc
    assert v3.relevance_auprc > v2.relevance_auprc
    assert v3.relevance_auprc >= 0.940

    assert v3.relevance_precision > v1.relevance_precision + 7.0
    assert v3.cross_lane_fp_rate < v1.cross_lane_fp_rate * 0.30  # >70% relative reduction

    # Edge inference budget: RTX 5070
    assert v3.e2e_latency_ms <= 27.2
    assert v3.single_stream_fps >= 36.8


def test_dual_nms_policy_comparisons():
    """Verify that Size-Adaptive NWD post-processing achieves Pareto dominance over fixed IoU and pure NWD."""
    comparisons = get_dual_nms_comparisons()
    assert len(comparisons) == 4

    iou70, iou45, pure_nwd, size_adaptive = comparisons

    # Size-Adaptive vs IoU 0.70: duplicate reduction
    assert size_adaptive.sub8px_dup_rate < iou70.sub8px_dup_rate * 0.25  # >75% reduction
    assert size_adaptive.sub8px_ap50 > iou70.sub8px_ap50

    # Size-Adaptive vs IoU 0.45: avoidance of adjacent-lamp error
    assert size_adaptive.adjacent_lamp_error < 2.0
    assert iou45.adjacent_lamp_error > 5.0  # Over-suppresses clustered dual lamps

    # Size-Adaptive vs Pure NWD: preservation of arrow accuracy
    assert size_adaptive.arrow_ap50 >= 94.5
    assert pure_nwd.arrow_ap50 < 93.0  # Pure NWD distorts macro arrows


def test_e47_audit_execution(tmp_path):
    """Verify that the full E47 audit script executes cleanly and passes all confirmation criteria."""
    report = run_e47_champion_v3_lineage_audit(
        config_path=PROJECT_ROOT / "configs" / "tlr_yolo11s_champion_v3.yaml",
        output_dir=tmp_path,
    )

    assert report["ticket"] == "E47"
    assert report["config_audit"]["status"] == "PASSED"
    for crit_name, crit_val in report["acceptance_criteria"].items():
        assert crit_val == "PASSED", f"Criterion {crit_name} must PASS"

    assert (tmp_path / "audit_e47_champion_v3_lineage.json").is_file()
    assert (tmp_path / "audit_e47_champion_v3_lineage.png").is_file()
