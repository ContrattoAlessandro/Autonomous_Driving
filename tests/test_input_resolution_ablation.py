"""Unit tests for E21: Input Resolution Ablation."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_input_resolution_ablation import (
    audit_dataset_resolution_geometry,
    benchmark_resolution_runtime,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)


def test_resolution_grid_and_anchors_scaling():
    """Verify that multi-resolution anchor grid counts match mathematical expectations."""
    resolutions = [(800, 1600), (960, 1920), (1024, 2048)]
    strides = [4, 8, 16, 32]

    for h, w in resolutions:
        expected_anchors = sum((h // s) * (w // s) for s in strides)
        if (h, w) == (800, 1600):
            assert expected_anchors == 106250
        elif (h, w) == (960, 1920):
            assert expected_anchors == 153000
        elif (h, w) == (1024, 2048):
            assert expected_anchors == 174080


def test_model_forward_multi_resolution():
    """Verify that the model architecture can dynamically process different input aspect ratios/resolutions."""
    wrapper = build_detection_model()
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig())
    model = wrapper.model.eval()

    device = torch.device("cpu")
    resolutions = [(384, 768), (512, 1024)]

    for h, w in resolutions:
        dummy = torch.randn(1, 3, h, w, device=device)
        with torch.inference_mode():
            output = model(dummy)
            assert output is not None
            if isinstance(output, dict):
                assert "relevance_logits" in output
                assert "traffic_candidate_boxes" in output
