"""Unit and integration tests for Decoupled NWD Constants (Ticket 07).

Validates class-decoupled Gaussian Normalized Wasserstein Distance:
    - Traffic Lights (Class 0): C_TL = 12.0, Area_threshold = 64.0 px^2
    - Road Arrows (Class 1): C_arrow = 24.0, Area_threshold = 1024.0 px^2
in ScaleAdaptiveNWDAssigner (TAL) and size_adaptive_nwd_nms post-processing.
"""

import math
import pytest
import torch

from tlr_yolo_mtl.deployment.postprocess import (
    compute_pairwise_nwd,
    nwd_nms,
    postprocess_multitask_outputs,
    size_adaptive_nms,
    size_adaptive_nwd_nms,
)
from tlr_yolo_mtl.training.tal import (
    NWDAwareTaskAlignedAssigner,
    ScaleAdaptiveNWDAssigner,
    build_task_aligned_assigner,
    compute_nwd_similarity,
)


def test_nwd_similarity_scalar_and_tensor():
    """Verify compute_nwd_similarity handles scalar and tensor constants identically."""
    boxes1 = torch.tensor([[10.0, 10.0, 20.0, 40.0], [100.0, 200.0, 160.0, 320.0]])
    boxes2 = torch.tensor([[12.0, 12.0, 22.0, 42.0], [100.0, 220.0, 160.0, 340.0]])

    # Scalar computation
    sim_c12 = compute_nwd_similarity(boxes1, boxes2, constant=12.0)
    sim_c24 = compute_nwd_similarity(boxes1, boxes2, constant=24.0)

    # Tensor computation
    constants = torch.tensor([12.0, 24.0])
    sim_tensor = compute_nwd_similarity(boxes1, boxes2, constant=constants)

    assert sim_c12.shape == (2,)
    assert sim_c24.shape == (2,)
    assert torch.allclose(sim_tensor[0], sim_c12[0], atol=1e-6)
    assert torch.allclose(sim_tensor[1], sim_c24[1], atol=1e-6)

    # Arrow longitudinal shift: C=24 preserves higher alignment than C=12
    # Arrow box: 60x120 with 20px shift
    arrow_shift_c12 = sim_c12[1].item()
    arrow_shift_c24 = sim_c24[1].item()
    assert arrow_shift_c24 > arrow_shift_c12
    assert arrow_shift_c24 > 0.40  # Meaningful gradient retained


def test_scale_adaptive_nwd_assigner_decoupled():
    """Verify ScaleAdaptiveNWDAssigner correctly assigns decoupled constants per class."""
    assigner = ScaleAdaptiveNWDAssigner(
        topk=4,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        nwd_weight=0.5,
        nwd_constant_tl=12.0,
        nwd_constant_arrow=24.0,
        area_threshold_tl=64.0,
        area_threshold_arrow=1024.0,
        mode="scale_adaptive",
    )

    bs = 2
    n_max_boxes = 4
    na = 20  # 20 candidate anchors

    # Scores: high confidence on target classes
    pd_scores = torch.rand(bs, na, 2)
    pd_bboxes = torch.rand(bs, na, 4) * 200.0
    pd_bboxes[:, :, 2:] = pd_bboxes[:, :, :2] + torch.rand(bs, na, 2) * 50.0 + 5.0

    # GT labels: mix of Class 0 (TL) and Class 1 (Arrow)
    gt_labels = torch.tensor([
        [[0], [1], [0], [1]],
        [[1], [0], [1], [0]],
    ])  # (bs, n_max_boxes, 1)

    # GT boxes: some tiny (TL: 6x18=108px), some medium (Arrow: 40x80=3200px or 20x40=800px)
    gt_bboxes = torch.tensor([
        [[10.0, 10.0, 16.0, 28.0], [50.0, 50.0, 70.0, 90.0], [100.0, 100.0, 106.0, 118.0], [150.0, 150.0, 190.0, 230.0]],
        [[30.0, 30.0, 50.0, 70.0], [20.0, 20.0, 26.0, 38.0], [80.0, 80.0, 120.0, 160.0], [120.0, 120.0, 126.0, 138.0]],
    ])

    mask_gt = torch.ones(bs, n_max_boxes, na, dtype=torch.bool)

    align_metric, overlaps = assigner.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt
    )

    assert align_metric.shape == (bs, n_max_boxes, na)
    assert overlaps.shape == (bs, n_max_boxes, na)
    assert not torch.isnan(align_metric).any()
    assert not torch.isinf(align_metric).any()
    assert not torch.isnan(overlaps).any()
    assert (overlaps >= 0.0).all() and (overlaps <= 1.0 + 1e-5).all()


def test_build_task_aligned_assigner_aliases():
    """Verify factory builder correctly parses all aliases and decoupled arguments."""
    assigner = build_task_aligned_assigner(
        assigner_type="scale_adaptive_nwd_tal",
        num_classes=2,
        nwd_constant_tl=12.0,
        nwd_constant_arrow=24.0,
        tiny_transition_area_tl=64.0,
        tiny_transition_area_arrow=1024.0,
    )
    assert isinstance(assigner, ScaleAdaptiveNWDAssigner)
    assert assigner.nwd_constants_map[0] == 12.0
    assert assigner.nwd_constants_map[1] == 24.0
    assert assigner.area_thresholds_map[0] == 64.0
    assert assigner.area_thresholds_map[1] == 1024.0

    # Backwards compatibility alias
    assert issubclass(NWDAwareTaskAlignedAssigner, ScaleAdaptiveNWDAssigner) or NWDAwareTaskAlignedAssigner is ScaleAdaptiveNWDAssigner


def test_size_adaptive_nwd_nms_decoupled():
    """Verify size_adaptive_nwd_nms correctly applies class-decoupled NMS logic."""
    # 3 boxes: 2 overlapping arrows and 1 distinct TL
    boxes = torch.tensor([
        [100.0, 100.0, 160.0, 220.0],  # Arrow 1: 60x120, area 7200
        [105.0, 115.0, 165.0, 235.0],  # Arrow 2: overlapping Arrow 1 with longitudinal offset
        [10.0, 10.0, 16.0, 28.0],      # TL: 6x18, tiny
    ])
    scores = torch.tensor([0.90, 0.85, 0.95])
    classes = torch.tensor([1, 1, 0])

    kept = size_adaptive_nwd_nms(
        boxes,
        scores,
        classes=classes,
        nwd_constant_tl=12.0,
        nwd_constant_arrow=24.0,
        area_threshold_tl=64.0,
        area_threshold_arrow=1024.0,
        nwd_threshold=0.5,
        iou_threshold=0.5,
    )

    assert 2 in kept  # TL retained
    assert 0 in kept  # Highest score arrow retained
    assert len(kept) >= 2


def test_postprocess_multitask_outputs_unified():
    """Verify unified 11-tensor postprocessing runs with decoupled constants."""
    bs = 1
    num_anchors = 100
    K_tl = 32
    M_ar = 8

    # Create dummy 11-tensor output
    detection = torch.randn(bs, 6, num_anchors)  # [x, y, w, h, score_tl, score_ar]
    detection[:, 4:, :] = torch.sigmoid(detection[:, 4:, :])
    detection[:, 4, 10] = 0.90  # Confident TL
    detection[:, 5, 20] = 0.85  # Confident Arrow

    states = torch.randn(bs, 4, num_anchors)
    rounds = torch.randn(bs, 1, num_anchors)
    maneuvers = torch.randn(bs, 3, num_anchors)
    ego_lane = torch.randn(bs, 1, num_anchors)

    traffic_candidates = torch.arange(K_tl).unsqueeze(0)
    traffic_candidate_valid = torch.ones(bs, K_tl, dtype=torch.bool)
    arrow_candidates = torch.arange(M_ar).unsqueeze(0)
    arrow_candidate_valid = torch.ones(bs, M_ar, dtype=torch.bool)

    relevance = torch.randn(bs, 1, K_tl)
    attention = torch.randn(bs, 4, K_tl, M_ar + 1)

    outputs = (
        detection,
        states,
        rounds,
        maneuvers,
        ego_lane,
        traffic_candidates,
        traffic_candidate_valid,
        arrow_candidates,
        arrow_candidate_valid,
        relevance,
        attention,
    )

    results = postprocess_multitask_outputs(
        outputs,
        traffic_confidence=0.25,
        arrow_confidence=0.25,
        nms_type="size_adaptive",
        nwd_constant_tl=12.0,
        nwd_constant_arrow=24.0,
        nwd_area_threshold_tl=64.0,
        nwd_area_threshold_arrow=1024.0,
    )

    assert "traffic_lights" in results
    assert "road_arrows" in results
    assert "boxes_xyxy" in results["traffic_lights"]
    assert "boxes_xyxy" in results["road_arrows"]


def test_arrow_ap50_simulation_comparison():
    """Simulate TAL assignment matching score improvement on elongated arrows with C=24 vs C=12."""
    # Synthetic ground truth road arrow: 50px wide x 120px tall
    gt_arrow = torch.tensor([[100.0, 100.0, 150.0, 220.0]])

    # Candidate predicted anchors with longitudinal offsets (e.g. y-shifts of 5px, 15px, 25px)
    pred_candidates = torch.tensor([
        [100.0, 105.0, 150.0, 225.0],  # 5px shift
        [100.0, 115.0, 150.0, 235.0],  # 15px shift
        [100.0, 125.0, 150.0, 245.0],  # 25px shift
    ])

    nwd_c12 = compute_nwd_similarity(gt_arrow.expand(3, -1), pred_candidates, constant=12.0)
    nwd_c24 = compute_nwd_similarity(gt_arrow.expand(3, -1), pred_candidates, constant=24.0)

    # C=24 provides strictly higher and smoother similarity across perspective shifts
    for i in range(3):
        assert nwd_c24[i] > nwd_c12[i]

    # Ratio of improvement increases with offset distance (mitigating starvation on distant anchors)
    ratio_5px = (nwd_c24[0] / nwd_c12[0]).item()
    ratio_25px = (nwd_c24[2] / nwd_c12[2]).item()
    assert ratio_25px > ratio_5px
