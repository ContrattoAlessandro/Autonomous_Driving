"""Task-Aligned Assigner implementations for TLR-YOLO-MTL.

Provides standard and scale-adaptive NWD-aware TaskAlignedAssigner modules to eliminate
anchor candidate starvation on tiny sub-grid objects while preserving rigid IoU matching
on large objects. Supports decoupled Gaussian Wasserstein constants and transition thresholds
per semantic class (e.g. compact vertical traffic lights vs elongated road arrow markings).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
from torch import nn
from ultralytics.utils.tal import TaskAlignedAssigner, xyxy2xywh, xywh2xyxy
from ultralytics.utils.ops import xyxy2xywh, xywh2xyxy


def compute_nwd_similarity(
    boxes1_xyxy: torch.Tensor,
    boxes2_xyxy: torch.Tensor,
    constant: float | torch.Tensor = 12.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute Gaussian Wasserstein similarity NWD in (0, 1] between pairs of boxes.

    Args:
        boxes1_xyxy: Bounding boxes [..., 4] in (x1, y1, x2, y2) format.
        boxes2_xyxy: Bounding boxes [..., 4] in (x1, y1, x2, y2) format.
        constant: Normalization constant C in pixels (scalar float or tensor broadcastable to box pairs).
        eps: Small epsilon for numerical stability.

    Returns:
        Tensor with values in (0, 1] of shape matching the broadcasted box dimensions.
    """
    if isinstance(constant, (int, float)):
        if constant <= 0:
            raise ValueError("NWD normalization constant must be positive")
    elif isinstance(constant, torch.Tensor):
        if (constant <= 0).any():
            raise ValueError("NWD normalization constant tensor must contain strictly positive values")
    else:
        raise TypeError(f"constant must be a float or torch.Tensor, got {type(constant)}")

    if boxes1_xyxy.numel() == 0 or boxes2_xyxy.numel() == 0:
        return torch.zeros(boxes1_xyxy.shape[:-1], device=boxes1_xyxy.device, dtype=boxes1_xyxy.dtype)

    c1 = (boxes1_xyxy[..., :2] + boxes1_xyxy[..., 2:]) / 2.0
    c2 = (boxes2_xyxy[..., :2] + boxes2_xyxy[..., 2:]) / 2.0
    s1 = (boxes1_xyxy[..., 2:] - boxes1_xyxy[..., :2]).clamp_min(0.0)
    s2 = (boxes2_xyxy[..., 2:] - boxes2_xyxy[..., :2]).clamp_min(0.0)

    d_center = c1 - c2
    d_size = s1 - s2
    w2 = d_center.square().sum(-1) + 0.25 * d_size.square().sum(-1)
    return torch.exp(-torch.sqrt(w2.clamp_min(eps)) / constant)


class ScaleAdaptiveNWDAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner with Class-Decoupled Normalized Wasserstein Distance (NWD).

    Combines classification scores with a hybrid IoU + NWD localization overlap metric:
        t = s^alpha * (Metric_overlap)^beta

    where Metric_overlap provides continuous, non-zero gradient and assignment signal
    for sub-grid traffic lights and perspective-elongated road arrows where discrete IoU
    collapses to zero.

    Supports decoupled constants:
        - Traffic lights (class 0): C_TL = 12.0, Area_threshold = 64 px^2
        - Road arrows (class 1): C_arrow = 24.0, Area_threshold = 1024 px^2

    Attributes:
        nwd_weight: Weight lambda in [0, 1] for NWD component.
        nwd_constant: Base distance scaling constant C for NWD (default: 12.0).
        nwd_constant_tl: Distance scaling constant C for traffic lights (default: 12.0).
        nwd_constant_arrow: Distance scaling constant C for road arrows (default: 24.0).
        area_threshold: Upper area bound (px^2) below which NWD is active (default: 64.0).
        area_threshold_tl: Upper area bound (px^2) for traffic lights (default: 64.0).
        area_threshold_arrow: Upper area bound (px^2) for road arrows (default: 1024.0).
        mode: Blending mode ("scale_adaptive", "convex", "additive").
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
        nwd_weight: float = 0.5,
        nwd_constant: float = 12.0,
        nwd_constant_tl: float | None = None,
        nwd_constant_arrow: float | None = None,
        nwd_constant_by_class: Mapping[int, float] | Sequence[float] | None = None,
        area_threshold: float | None = 64.0,
        area_threshold_tl: float | None = None,
        area_threshold_arrow: float | None = None,
        area_threshold_by_class: Mapping[int, float] | Sequence[float] | None = None,
        mode: str = "scale_adaptive",
    ):
        super().__init__(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )
        self.nwd_weight = float(nwd_weight)
        self.nwd_constant = float(nwd_constant)
        self.area_threshold = float(area_threshold) if area_threshold is not None else None
        self.mode = str(mode)

        # Build class-specific NWD constants map
        self.nwd_constants_map: dict[int, float] = {}
        if nwd_constant_by_class is not None:
            if isinstance(nwd_constant_by_class, Mapping):
                self.nwd_constants_map = {int(k): float(v) for k, v in nwd_constant_by_class.items()}
            else:
                self.nwd_constants_map = {i: float(v) for i, v in enumerate(nwd_constant_by_class)}
        else:
            if nwd_constant_tl is not None or nwd_constant_arrow is not None:
                self.nwd_constants_map[0] = float(nwd_constant_tl if nwd_constant_tl is not None else nwd_constant)
                self.nwd_constants_map[1] = float(nwd_constant_arrow if nwd_constant_arrow is not None else 24.0)

        # Build class-specific area thresholds map
        self.area_thresholds_map: dict[int, float | None] = {}
        if area_threshold_by_class is not None:
            if isinstance(area_threshold_by_class, Mapping):
                self.area_thresholds_map = {int(k): (float(v) if v is not None else None) for k, v in area_threshold_by_class.items()}
            else:
                self.area_thresholds_map = {i: (float(v) if v is not None else None) for i, v in enumerate(area_threshold_by_class)}
        else:
            if area_threshold_tl is not None or area_threshold_arrow is not None:
                self.area_thresholds_map[0] = float(area_threshold_tl if area_threshold_tl is not None else (area_threshold if area_threshold is not None else 64.0))
                self.area_thresholds_map[1] = float(area_threshold_arrow if area_threshold_arrow is not None else 1024.0)

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute task alignment metric and localization overlaps with decoupled NWD enhancement.

        Args:
            pd_scores: Predicted classification scores (bs, num_total_anchors, num_classes).
            pd_bboxes: Predicted bounding boxes in pixels (bs, num_total_anchors, 4).
            gt_labels: Ground truth labels (bs, n_max_boxes, 1) or (bs, n_max_boxes).
            gt_bboxes: Ground truth boxes in pixels (bs, n_max_boxes, 4).
            mask_gt: Mask for valid candidate anchor pairs (bs, n_max_boxes, num_total_anchors).

        Returns:
            align_metric: Alignment metric tensor (bs, n_max_boxes, num_total_anchors).
            overlaps: Overlap metric tensor (bs, n_max_boxes, num_total_anchors).
        """
        bs = pd_scores.shape[0]
        n_max_boxes = gt_bboxes.shape[1] if gt_bboxes.ndim >= 2 else 0
        self.bs = bs
        self.n_max_boxes = n_max_boxes
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, n_max_boxes, na)
        overlaps = torch.zeros([bs, n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([bs, n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        batch_ind = torch.arange(bs, device=pd_scores.device)[:, None]
        gt_lbl = gt_labels.squeeze(-1).long() if gt_labels.ndim == 3 else gt_labels.long()
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_lbl][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]

        iou_overlaps = self.iou_calculation(gt_boxes, pd_boxes)

        if self.nwd_weight > 0 and pd_boxes.numel() > 0:
            gt_classes = gt_lbl[:, :, None].expand(-1, -1, na)[mask_gt]

            # Vectorized constant lookup per candidate pair
            if self.nwd_constants_map:
                c_table = torch.full(
                    (self.num_classes,),
                    self.nwd_constant,
                    dtype=gt_boxes.dtype,
                    device=gt_boxes.device,
                )
                for cls_idx, c_val in self.nwd_constants_map.items():
                    if 0 <= cls_idx < self.num_classes:
                        c_table[cls_idx] = float(c_val)
                constants = c_table[gt_classes.clamp(0, self.num_classes - 1)]
            else:
                constants = self.nwd_constant

            nwd_overlaps = compute_nwd_similarity(
                gt_boxes, pd_boxes, constant=constants, eps=self.eps
            )

            if self.mode == "scale_adaptive":
                gt_w = (gt_boxes[..., 2] - gt_boxes[..., 0]).clamp_min(0.0)
                gt_h = (gt_boxes[..., 3] - gt_boxes[..., 1]).clamp_min(0.0)
                gt_area = gt_w * gt_h

                if self.area_thresholds_map:
                    a_table = torch.full(
                        (self.num_classes,),
                        self.area_threshold if self.area_threshold is not None else 1e9,
                        dtype=gt_boxes.dtype,
                        device=gt_boxes.device,
                    )
                    for cls_idx, a_val in self.area_thresholds_map.items():
                        if 0 <= cls_idx < self.num_classes:
                            a_table[cls_idx] = float(a_val) if a_val is not None else 1e9
                    area_thresh = a_table[gt_classes.clamp(0, self.num_classes - 1)]
                    scale_weight = (1.0 - gt_area / area_thresh).clamp(0.0, 1.0)
                elif self.area_threshold is not None:
                    scale_weight = (1.0 - gt_area / self.area_threshold).clamp(0.0, 1.0)
                else:
                    scale_weight = torch.ones_like(gt_area)

                effective_nwd_weight = self.nwd_weight * scale_weight
                combined_overlaps = (1.0 - effective_nwd_weight) * iou_overlaps + effective_nwd_weight * nwd_overlaps
            elif self.mode == "additive":
                combined_overlaps = iou_overlaps + self.nwd_weight * nwd_overlaps
            elif self.mode == "convex":
                combined_overlaps = (1.0 - self.nwd_weight) * iou_overlaps + self.nwd_weight * nwd_overlaps
            else:
                combined_overlaps = (1.0 - self.nwd_weight) * iou_overlaps + self.nwd_weight * nwd_overlaps
            overlaps[mask_gt] = combined_overlaps
        else:
            overlaps[mask_gt] = iou_overlaps

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps


# Backward compatibility alias
NWDAwareTaskAlignedAssigner = ScaleAdaptiveNWDAssigner


def build_task_aligned_assigner(
    *,
    assigner_type: str = "standard",
    topk: int = 10,
    num_classes: int = 80,
    alpha: float = 0.5,
    beta: float = 6.0,
    stride: list | None = None,
    eps: float = 1e-9,
    topk2: int | None = None,
    nwd_weight: float = 0.5,
    nwd_constant: float = 12.0,
    nwd_constant_tl: float | None = None,
    nwd_constant_arrow: float | None = None,
    nwd_constant_by_class: Mapping[int, float] | Sequence[float] | None = None,
    area_threshold: float | None = 64.0,
    area_threshold_tl: float | None = None,
    area_threshold_arrow: float | None = None,
    area_threshold_by_class: Mapping[int, float] | Sequence[float] | None = None,
    mode: str = "scale_adaptive",
    **kwargs: Any,
) -> TaskAlignedAssigner:
    """Factory helper to build standard or ScaleAdaptiveNWDAssigner."""
    normalized_type = str(assigner_type).lower().strip()
    # Handle aliases
    if "lambda_nwd" in kwargs and "nwd_weight" not in kwargs:
        nwd_weight = float(kwargs.pop("lambda_nwd"))
    if "tiny_transition_area" in kwargs and "area_threshold" not in kwargs:
        area_threshold = float(kwargs.pop("tiny_transition_area"))
    if "tiny_transition_area_tl" in kwargs and nwd_constant_tl is None:
        area_threshold_tl = float(kwargs.pop("tiny_transition_area_tl"))
    if "tiny_transition_area_arrow" in kwargs and area_threshold_arrow is None:
        area_threshold_arrow = float(kwargs.pop("tiny_transition_area_arrow"))

    # Check for decoupled parameters in kwargs
    if "nwd_constant_tl" in kwargs and nwd_constant_tl is None:
        nwd_constant_tl = float(kwargs.pop("nwd_constant_tl"))
    if "nwd_constant_arrow" in kwargs and nwd_constant_arrow is None:
        nwd_constant_arrow = float(kwargs.pop("nwd_constant_arrow"))
    if "area_threshold_tl" in kwargs and area_threshold_tl is None:
        area_threshold_tl = float(kwargs.pop("area_threshold_tl"))
    if "area_threshold_arrow" in kwargs and area_threshold_arrow is None:
        area_threshold_arrow = float(kwargs.pop("area_threshold_arrow"))
    if "nwd_constant_by_class" in kwargs and nwd_constant_by_class is None:
        nwd_constant_by_class = kwargs.pop("nwd_constant_by_class")
    if "area_threshold_by_class" in kwargs and area_threshold_by_class is None:
        area_threshold_by_class = kwargs.pop("area_threshold_by_class")

    if normalized_type in (
        "nwd",
        "nwd_aware",
        "nwd_tal",
        "nwd_aware_tal",
        "scale_adaptive",
        "scale_adaptive_nwd",
        "scale_adaptive_nwd_tal",
        "scale_adaptive_nwd_assigner",
    ):
        return ScaleAdaptiveNWDAssigner(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
            nwd_weight=nwd_weight,
            nwd_constant=nwd_constant,
            nwd_constant_tl=nwd_constant_tl,
            nwd_constant_arrow=nwd_constant_arrow,
            nwd_constant_by_class=nwd_constant_by_class,
            area_threshold=area_threshold,
            area_threshold_tl=area_threshold_tl,
            area_threshold_arrow=area_threshold_arrow,
            area_threshold_by_class=area_threshold_by_class,
            mode=mode,
        )
    elif normalized_type in ("standard", "tal", "default"):
        return TaskAlignedAssigner(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )
    else:
        raise ValueError(f"Unknown assigner_type '{assigner_type}'. Expected 'standard' or 'nwd'.")


