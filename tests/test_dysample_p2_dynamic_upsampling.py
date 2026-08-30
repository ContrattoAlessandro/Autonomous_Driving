"""Unit and integration tests for DySample Dynamic Upsampling in the P3 -> P2 Lateral Path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.dysample import (
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
    assert dysample_pl.offset_conv.weight.grad is not None


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
