"""Unit tests for Ticket E22: Multi-Scale P2 + P3 Candidate Token Fusion."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.multiscale_fusion import (
    MultiScaleCandidateFeatureExtractor,
    MultiScaleUnifiedTrafficControlDetect,
    attach_multiscale_unified_relevance_head,
)
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
)


def test_multiscale_candidate_feature_extractor_shapes():
    """Verify that multi-scale feature extractor produces expected output tensor shapes."""
    extractor = MultiScaleCandidateFeatureExtractor(
        token_feature_dim=64,
        out_feature_dim=64,
        mode="p2_p3_fused",
    )

    B, K = 2, 32
    # Pyramid maps: P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)
    pyramid_features = [
        torch.randn(B, 64, 200, 400),
        torch.randn(B, 64, 100, 200),
        torch.randn(B, 64, 50, 100),
        torch.randn(B, 64, 25, 50),
    ]
    boxes = torch.rand(B, K, 4)  # (cx, cy, w, h) in [0, 1]

    out = extractor(pyramid_features, boxes)
    assert out.shape == (B, K, 64)
    assert not torch.isnan(out).any()


def test_multiscale_modes_ablation():
    """Verify that p2_only, p3_only, p2_p3_fused, and p2_p3_p4_fused all execute cleanly."""
    B, K = 2, 16
    pyramid_features = [
        torch.randn(B, 64, 200, 400),
        torch.randn(B, 64, 100, 200),
        torch.randn(B, 64, 50, 100),
        torch.randn(B, 64, 25, 50),
    ]
    boxes = torch.rand(B, K, 4)

    for mode in ("p2_only", "p3_only", "p2_p3_fused", "p2_p3_p4_fused"):
        extractor = MultiScaleCandidateFeatureExtractor(
            token_feature_dim=64,
            out_feature_dim=64,
            mode=mode,
        )
        out = extractor(pyramid_features, boxes)
        assert out.shape == (B, K, 64)


def test_multiscale_unified_detector_forward():
    """Verify forward execution of MultiScaleUnifiedTrafficControlDetect with complete detector."""
    wrapper = build_detection_model()
    attach_multiscale_unified_relevance_head(
        wrapper, config=UnifiedHeadConfig(), fusion_mode="p2_p3_fused"
    )

    dummy = torch.randn(1, 3, 384, 768)
    with torch.inference_mode():
        out = wrapper.model(dummy)
        assert out is not None
        if isinstance(out, tuple) and isinstance(out[0], tuple):
            pred_dict = out[0][-1]
            assert "relevance_logits" in pred_dict
            assert "traffic_candidate_boxes" in pred_dict
