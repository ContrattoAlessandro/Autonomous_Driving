from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest

from tlr_yolo_mtl.deployment.temporal_smoother import (
    TemporalSlidingWindowSmoother,
    Tracklet,
    compute_size_adaptive_similarity,
    compute_temporal_flicker_rate,
)
from tlr_yolo_mtl.deployment.postprocess import postprocess_multitask_outputs


def test_tracklet_lifecycle_and_association():
    """Test tracklet creation, hit counting, age increment, and max_age pruning."""
    smoother = TemporalSlidingWindowSmoother(
        window_size=3,
        weights=[0.20, 0.35, 0.45],
        nwd_match_threshold=0.30,
        max_age=2,
    )

    boxes = torch.tensor([[100.0, 100.0, 150.0, 200.0]])
    scores = torch.tensor([0.90])
    states = torch.tensor([[0.05, 0.05, 0.85, 0.05]])
    relevance = torch.tensor([0.80])

    # Frame 1: Creation of tracklet 0
    res1 = smoother.update(boxes, scores, states, relevance)
    assert len(smoother.tracklets) == 1
    assert smoother.tracklets[0].track_id == 0
    assert smoother.tracklets[0].hits == 1
    assert smoother.tracklets[0].age == 0
    assert res1["track_ids"][0].item() == 0

    # Frame 2: Association with tracklet 0
    boxes2 = torch.tensor([[101.0, 101.0, 151.0, 201.0]])
    res2 = smoother.update(boxes2, scores, states, relevance)
    assert len(smoother.tracklets) == 1
    assert smoother.tracklets[0].track_id == 0
    assert smoother.tracklets[0].hits == 2
    assert smoother.tracklets[0].age == 0
    assert res2["track_ids"][0].item() == 0

    # Frame 3: Missing detection (age increments)
    empty_boxes = torch.empty((0, 4))
    empty_scores = torch.empty((0,))
    empty_states = torch.empty((0, 4))
    empty_relevance = torch.empty((0,))
    smoother.update(empty_boxes, empty_scores, empty_states, empty_relevance)
    assert len(smoother.tracklets) == 1
    assert smoother.tracklets[0].age == 1

    # Frame 4: Missing detection (age becomes 2)
    smoother.update(empty_boxes, empty_scores, empty_states, empty_relevance)
    assert len(smoother.tracklets) == 1
    assert smoother.tracklets[0].age == 2

    # Frame 5: Missing detection (age becomes 3 > max_age -> pruned)
    smoother.update(empty_boxes, empty_scores, empty_states, empty_relevance)
    assert len(smoother.tracklets) == 0


def test_size_adaptive_nwd_iou_association():
    """Verify Size-Adaptive NWD similarity is applied for sub-8px boxes."""
    # Tiny box: 5x10 px = 50 px^2 (< 64 px^2 threshold)
    tiny1 = torch.tensor([[200.0, 200.0, 205.0, 210.0]])
    tiny2 = torch.tensor([[202.0, 201.0, 207.0, 211.0]])  # 2 px center shift

    sim_tiny = compute_size_adaptive_similarity(tiny1, tiny2, nwd_constant=12.0, nwd_area_threshold=64.0)
    assert sim_tiny.item() > 0.75, "NWD similarity should remain high (>0.75) for tiny box small shifts"

    # Large box: 40x80 px = 3200 px^2 (>= 64 px^2 threshold)
    large1 = torch.tensor([[200.0, 200.0, 240.0, 280.0]])
    large2 = torch.tensor([[202.0, 201.0, 242.0, 281.0]])

    sim_large = compute_size_adaptive_similarity(large1, large2, nwd_constant=12.0, nwd_area_threshold=64.0)
    # IoU for large boxes with 2px shift should be close to ~0.90
    assert sim_large.item() > 0.85


def test_asymmetric_zero_lag_red_trigger():
    """Test that transitioning to Red triggers with ZERO frame lag."""
    smoother = TemporalSlidingWindowSmoother(
        window_size=3,
        weights=[0.20, 0.35, 0.45],
        asymmetric_red=True,
        red_instant_threshold=0.40,
    )

    box = torch.tensor([[300.0, 300.0, 320.0, 350.0]])
    score = torch.tensor([0.90])
    rel = torch.tensor([0.85])

    # Frames 1 and 2: Established Green
    green_state = torch.tensor([[0.02, 0.02, 0.94, 0.02]])
    smoother.update(box, score, green_state, rel)
    smoother.update(box, score, green_state, rel)

    # Frame 3: Instantaneous Red signal appears (p_red = 0.88 >= 0.40)
    red_state = torch.tensor([[0.88, 0.04, 0.04, 0.04]])
    res3 = smoother.update(box, score, red_state, rel)

    # Must immediately output RED (index 0) with zero frame lag
    assert res3["state_indices"][0].item() == 0, "Asymmetric Red Gate must trigger state index 0 immediately"
    assert res3["state_probabilities"][0, 0].item() >= 0.80, "Red probability must reflect instant red state"


def test_damped_transition_away_from_red():
    """Test that a single noisy frame does not drop an established Red state."""
    smoother = TemporalSlidingWindowSmoother(
        window_size=3,
        weights=[0.20, 0.35, 0.45],
        asymmetric_red=True,
        red_instant_threshold=0.40,
        red_release_threshold=0.55,
    )

    box = torch.tensor([[300.0, 300.0, 320.0, 350.0]])
    score = torch.tensor([0.90])
    rel = torch.tensor([0.85])

    # Frames 1, 2: Established Red
    red_state = torch.tensor([[0.94, 0.02, 0.02, 0.02]])
    smoother.update(box, score, red_state, rel)
    smoother.update(box, score, red_state, rel)

    # Frame 3: Single-frame drop / noise (e.g. AC flicker to Off)
    noisy_off_state = torch.tensor([[0.05, 0.05, 0.05, 0.85]])
    res3 = smoother.update(box, score, noisy_off_state, rel)

    # Must maintain Red (0) due to damped transition away from Red
    assert res3["state_indices"][0].item() == 0, "Single noisy frame must not release established Red state"
    assert res3["flicker_suppressed"][0].item() is True, "Flicker suppression flag should be True"


def test_flicker_suppression_and_diagnostic_metric():
    """Test that isolated single-frame AC LED spikes (Green -> Off -> Green) are smoothed."""
    smoother = TemporalSlidingWindowSmoother(
        window_size=3,
        weights=[0.20, 0.35, 0.45],
        asymmetric_red=True,
    )

    box = torch.tensor([[400.0, 200.0, 415.0, 240.0]])
    score = torch.tensor([0.85])
    rel = torch.tensor([0.90])

    green = torch.tensor([[0.02, 0.02, 0.94, 0.02]])
    off_noise = torch.tensor([[0.02, 0.02, 0.02, 0.94]])

    # Sequence: Green, Green, Off (noise), Green
    smoother.update(box, score, green, rel)
    smoother.update(box, score, green, rel)
    res_noisy = smoother.update(box, score, off_noise, rel)
    res_recover = smoother.update(box, score, green, rel)

    # Smoothed state at noisy frame should remain Green (index 2)
    assert res_noisy["state_indices"][0].item() == 2, "Smoothed state must suppress momentary Off glitch"

    # Test flicker rate metric
    raw_seq = [2, 2, 3, 2, 2]
    smoothed_seq = [2, 2, 2, 2, 2]
    raw_rate = compute_temporal_flicker_rate(raw_seq)
    smoothed_rate = compute_temporal_flicker_rate(smoothed_seq)

    assert raw_rate == 1.0 / 3.0
    assert smoothed_rate == 0.0


def test_relevance_and_joint_score_smoothing():
    """Test causal temporal smoothing on continuous relevance and joint scores."""
    smoother = TemporalSlidingWindowSmoother(
        window_size=3,
        weights=[0.20, 0.35, 0.45],
    )

    box = torch.tensor([[100.0, 100.0, 120.0, 150.0]])
    score = torch.tensor([0.80])
    state = torch.tensor([[0.05, 0.05, 0.85, 0.05]])

    # Sequential relevance probabilities: 0.60, 0.70, 0.80
    smoother.update(box, score, state, torch.tensor([0.60]))
    smoother.update(box, score, state, torch.tensor([0.70]))
    res = smoother.update(box, score, state, torch.tensor([0.80]))

    expected_rel = 0.20 * 0.60 + 0.35 * 0.70 + 0.45 * 0.80  # 0.12 + 0.245 + 0.36 = 0.725
    assert abs(res["relevance_probabilities"][0].item() - expected_rel) < 1e-4
    assert abs(res["joint_scores"][0].item() - (0.80 * expected_rel)) < 1e-4


def test_postprocess_multitask_outputs_with_temporal_smoother():
    """Test integration of TemporalSlidingWindowSmoother inside postprocess_multitask_outputs."""
    # Synthetic 11-tensor unified output
    B, N_TL, N_Arr = 1, 32, 32
    det = torch.zeros((B, 6, 100))
    # 1 valid traffic light detection
    det[0, 0:4, 0] = torch.tensor([100.0, 100.0, 120.0, 150.0])
    det[0, 4, 0] = 0.90  # TL score
    det[0, 5, 0] = 0.00  # class 0 = TL

    states = torch.zeros((B, 4, 100))
    states[0, 2, 0] = 3.0  # Green logit

    rounds = torch.zeros((B, 1, 100))
    rounds[0, 0, 0] = 2.0  # Round

    maneuvers = torch.zeros((B, 3, 100))
    ego_lane = torch.zeros((B, 1, 100))

    tl_cands = torch.zeros((B, N_TL))
    tl_cands[0, 0] = 0
    tl_cand_valid = torch.zeros((B, N_TL))
    tl_cand_valid[0, 0] = 1

    arr_cands = torch.zeros((B, N_Arr))
    arr_cand_valid = torch.zeros((B, N_Arr))

    relevance = torch.zeros((B, 1, N_TL))
    relevance[0, 0, 0] = 2.5  # sigmoid(2.5) ~ 0.924
    attention = torch.zeros((B, 4, N_TL, N_Arr))

    outputs = (
        det, states, rounds, maneuvers, ego_lane,
        tl_cands, tl_cand_valid, arr_cands, arr_cand_valid,
        relevance, attention,
    )

    smoother = TemporalSlidingWindowSmoother(window_size=3)

    res = postprocess_multitask_outputs(
        outputs,
        traffic_confidence=0.25,
        temporal_smoother=smoother,
    )

    assert "traffic_lights" in res
    assert "track_ids" in res["traffic_lights"]
    assert "flicker_suppressed" in res["traffic_lights"]
    assert res["traffic_lights"]["track_ids"].shape == (1, 1)
    assert res["traffic_lights"]["track_ids"][0, 0].item() == 0


def test_reset_functionality():
    """Test smoother reset."""
    smoother = TemporalSlidingWindowSmoother(window_size=3)
    box = torch.tensor([[100.0, 100.0, 120.0, 150.0]])
    score = torch.tensor([0.80])
    state = torch.tensor([[0.05, 0.05, 0.85, 0.05]])
    rel = torch.tensor([0.80])

    smoother.update(box, score, state, rel)
    assert len(smoother.tracklets) == 1
    assert smoother.next_track_id == 1

    smoother.reset()
    assert len(smoother.tracklets) == 0
    assert smoother.next_track_id == 0
