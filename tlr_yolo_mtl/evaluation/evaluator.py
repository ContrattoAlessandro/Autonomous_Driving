"""Validation evaluation engine for TLR-YOLO-MTL multi-task models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from ..deployment.postprocess import size_adaptive_nms, xywh_to_xyxy

if TYPE_CHECKING:
    from ..training.losses import TLRMultiTaskCriterion
from .matching import greedy_iou_match
from .metrics import (
    binary_classification_metrics,
    compute_detection_and_attribute_map,
    compute_granular_scale_metrics,
    multiclass_confusion_matrix,
    multiclass_metrics,
    multilabel_metrics,
    validation_selection_score,
)


def evaluate_validation_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: TLRMultiTaskCriterion | None = None,
    *,
    device: torch.device,
    amp_enabled: bool = True,
    max_batches: int | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    max_detections: int = 300,
    granular_scale_metrics: bool = False,
) -> dict[str, Any]:
    """Run validation pass, computing mAP, attributes, relevance metrics and composite score."""

    model.eval()
    loss_totals: dict[str, float] = {
        "total": 0.0,
        "detection": 0.0,
        "state": 0.0,
        "round": 0.0,
        "maneuver": 0.0,
        "ego_lane": 0.0,
        "relevance": 0.0,
        "nwd": 0.0,
    }
    loss_batch_count = 0

    pred_boxes_list: list[np.ndarray] = []
    pred_scores_list: list[np.ndarray] = []
    pred_classes_list: list[np.ndarray] = []
    pred_states_list: list[np.ndarray] = []

    gt_boxes_list: list[np.ndarray] = []
    gt_classes_list: list[np.ndarray] = []
    gt_states_list: list[np.ndarray] = []

    all_pred_rel: list[float] = []
    all_gt_rel: list[int] = []

    all_pred_state: list[int] = []
    all_gt_state: list[int] = []

    all_pred_round: list[float] = []
    all_gt_round: list[int] = []

    all_pred_maneuver: list[Sequence[float]] = []
    all_gt_maneuver: list[Sequence[int]] = []

    total_gt_rel_red = 0
    recalled_gt_rel_red = 0

    for batch_idx, raw_batch in enumerate(val_loader, 1):
        if max_batches is not None and batch_idx > max_batches:
            break

        batch = {
            name: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in raw_batch.items()
        }

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predictions = model(batch["img"])
                if criterion is not None:
                    try:
                        losses = criterion(predictions, batch)
                        for name in loss_totals:
                            if hasattr(losses, name):
                                loss_totals[name] += float(
                                    getattr(losses, name).detach().float()
                                )
                        loss_batch_count += 1
                    except Exception:
                        pass

            if isinstance(predictions, tuple):
                decoded, raw = predictions
            elif isinstance(predictions, dict):
                decoded = predictions.get(0, predictions.get("decoded"))
                raw = predictions
            else:
                decoded = predictions
                raw = {}

            batch_size = int(batch["img"].shape[0])
            img_h = float(batch["img"].shape[-2])
            img_w = float(batch["img"].shape[-1])

            # Extract raw heads if available
            traffic_boxes = raw.get("traffic_candidate_boxes")  # [B, 32, 4]
            traffic_valid = raw.get("traffic_candidate_valid")  # [B, 32]
            traffic_scores = raw.get("traffic_candidate_scores")  # [B, 32]
            traffic_indices = raw.get("traffic_candidate_indices")  # [B, 32]
            relevance_logits = raw.get("relevance_logits")  # [B, 1, 32]
            state_logits = raw.get("state_logits")  # [B, 4, A]
            round_logits = raw.get("round_logits")  # [B, 1, A]
            maneuver_logits = raw.get("maneuver_logits")  # [B, 3, A]

            for b in range(batch_size):
                # 1. Post-process predictions for image b across both classes (0: TL, 1: Arrow)
                p_boxes: list[np.ndarray] = []
                p_scores: list[np.ndarray] = []
                p_classes: list[np.ndarray] = []
                p_states: list[np.ndarray] = []

                for c in (0, 1):
                    scores_c = decoded[b, 4 + c]
                    keep_mask = scores_c >= conf_threshold
                    if bool(keep_mask.any()):
                        c_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
                        boxes_xywh = decoded[b, :4, c_indices].transpose(0, 1)
                        boxes_xyxy_px = xywh_to_xyxy(boxes_xywh)
                        kept_nms = size_adaptive_nms(
                            boxes_xyxy_px,
                            scores_c[c_indices],
                            nwd_threshold=0.50,
                            iou_threshold=iou_threshold,
                            nwd_constant=12.0 if c == 0 else 24.0,
                            area_threshold=64.0 if c == 0 else 1024.0,
                        )[:max_detections]
                        kept_dense = c_indices[kept_nms]
                        kept_px = boxes_xyxy_px[kept_nms]
                        norm_scale = torch.tensor(
                            [img_w, img_h, img_w, img_h], device=device
                        )
                        kept_norm = (kept_px / norm_scale).clamp(0.0, 1.0)

                        p_boxes.append(kept_norm.cpu().numpy())
                        p_scores.append(scores_c[kept_dense].cpu().numpy())
                        p_classes.append(np.full(len(kept_nms), c, dtype=np.int64))

                        if c == 0 and state_logits is not None:
                            dense_states = state_logits[b, :, kept_dense]  # [4, K]
                            p_states.append(dense_states.argmax(0).cpu().numpy())
                        else:
                            p_states.append(np.full(len(kept_nms), -1, dtype=np.int64))

                if p_boxes:
                    pred_boxes_list.append(np.concatenate(p_boxes, axis=0))
                    pred_scores_list.append(np.concatenate(p_scores, axis=0))
                    pred_classes_list.append(np.concatenate(p_classes, axis=0))
                    pred_states_list.append(np.concatenate(p_states, axis=0))
                else:
                    pred_boxes_list.append(np.zeros((0, 4), dtype=float))
                    pred_scores_list.append(np.zeros(0, dtype=float))
                    pred_classes_list.append(np.zeros(0, dtype=np.int64))
                    pred_states_list.append(np.zeros(0, dtype=np.int64))

                # 2. Extract GT for image b
                b_mask = (batch["object_batch_idx"] == b)
                gt_xywh = batch["object_bboxes"][b_mask].cpu().numpy().reshape(-1, 4)
                if len(gt_xywh) > 0:
                    cx, cy, bw, bh = (
                        gt_xywh[:, 0],
                        gt_xywh[:, 1],
                        gt_xywh[:, 2],
                        gt_xywh[:, 3],
                    )
                    gt_xyxy = np.stack(
                        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1
                    ).reshape(-1, 4)
                    gt_cls = batch["object_cls"][b_mask].reshape(-1).cpu().numpy()
                    gt_st = batch["object_state"][b_mask].reshape(-1).cpu().numpy()
                    gt_rd = batch["object_round"][b_mask].reshape(-1).cpu().numpy()
                    gt_mv = batch["object_maneuver"][b_mask].reshape(-1, 3).cpu().numpy()
                    gt_rl = batch["object_relevance"][b_mask].reshape(-1).cpu().numpy()
                else:
                    gt_xyxy = np.zeros((0, 4), dtype=float)
                    gt_cls = np.zeros(0, dtype=np.int64)
                    gt_st = np.zeros(0, dtype=np.int64)
                    gt_rd = np.zeros(0, dtype=np.int64)
                    gt_mv = np.zeros((0, 3), dtype=float)
                    gt_rl = np.zeros(0, dtype=np.int64)

                gt_boxes_list.append(gt_xyxy)
                gt_classes_list.append(gt_cls)
                gt_states_list.append(gt_st)

                # 3. Match candidate Traffic Lights for task-specific metrics
                tl_mask = (gt_cls == 0)
                tl_gt_boxes = gt_xyxy[tl_mask]
                tl_gt_st = gt_st[tl_mask]
                tl_gt_rd = gt_rd[tl_mask]
                tl_gt_mv = gt_mv[tl_mask]
                tl_gt_rl = gt_rl[tl_mask]

                # Count GT Relevant Red TLs (state 0 == red, relevance 1 == relevant)
                rel_red_mask = (tl_gt_st == 0) & (tl_gt_rl == 1)
                total_gt_rel_red += int(np.sum(rel_red_mask))

                if (
                    len(tl_gt_boxes) > 0
                    and traffic_boxes is not None
                    and traffic_valid is not None
                    and traffic_scores is not None
                ):
                    c_valid = traffic_valid[b].bool().cpu().numpy()
                    if c_valid.any():
                        v_indices = np.where(c_valid)[0]
                        c_boxes_raw = traffic_boxes[b, v_indices].cpu().numpy()
                        cx, cy, bw, bh = (
                            c_boxes_raw[:, 0],
                            c_boxes_raw[:, 1],
                            c_boxes_raw[:, 2],
                            c_boxes_raw[:, 3],
                        )
                        c_boxes_xyxy = np.stack(
                            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                            axis=-1,
                        )
                        c_sc = traffic_scores[b, v_indices].cpu().numpy()
                        matches, _, _ = greedy_iou_match(
                            c_boxes_xyxy, c_sc, tl_gt_boxes, iou_threshold=0.5
                        )
                        c_dens = traffic_indices[b, v_indices].cpu().numpy()

                        for m in matches:
                            slot_idx = v_indices[m.prediction_index]
                            dens_idx = c_dens[m.prediction_index]
                            gt_idx = m.target_index

                            # Relevance
                            r_prob = None
                            if tl_gt_rl[gt_idx] >= 0 and relevance_logits is not None:
                                r_prob = float(
                                    relevance_logits[b, 0, slot_idx].sigmoid().item()
                                )
                                all_pred_rel.append(r_prob)
                                all_gt_rel.append(int(tl_gt_rl[gt_idx]))

                            # Relevant Red Recall tracking
                            if (
                                tl_gt_st[gt_idx] == 0
                                and tl_gt_rl[gt_idx] == 1
                                and r_prob is not None
                                and r_prob >= 0.5
                            ):
                                recalled_gt_rel_red += 1

                            # State
                            if 0 <= tl_gt_st[gt_idx] < 4 and state_logits is not None:
                                s_pred = int(
                                    state_logits[b, :, dens_idx].argmax(0).item()
                                )
                                all_pred_state.append(s_pred)
                                all_gt_state.append(int(tl_gt_st[gt_idx]))

                            # Round
                            if tl_gt_rd[gt_idx] >= 0 and round_logits is not None:
                                rd_prob = float(
                                    round_logits[b, 0, dens_idx].sigmoid().item()
                                )
                                all_pred_round.append(rd_prob)
                                all_gt_round.append(int(tl_gt_rd[gt_idx]))

                            # Maneuver
                            if (
                                np.all(tl_gt_mv[gt_idx] >= 0)
                                and maneuver_logits is not None
                            ):
                                mv_prob = (
                                    maneuver_logits[b, :, dens_idx]
                                    .sigmoid()
                                    .cpu()
                                    .numpy()
                                )
                                all_pred_maneuver.append(mv_prob.tolist())
                                all_gt_maneuver.append(
                                    tl_gt_mv[gt_idx].astype(int).tolist()
                                )

    # Compute aggregate metrics
    det_map = compute_detection_and_attribute_map(
        pred_boxes_list,
        pred_scores_list,
        pred_classes_list,
        gt_boxes_list,
        gt_classes_list,
        pred_states_list,
        gt_states_list,
        image_shape=(int(img_h), int(img_w)),
    )

    # Relevant Red Recall
    relevant_red_recall = (
        float(recalled_gt_rel_red / total_gt_rel_red)
        if total_gt_rel_red > 0
        else 0.0
    )

    # Relevance metrics
    if len(all_gt_rel) > 0 and len(np.unique(all_gt_rel)) > 1:
        rel_metrics = binary_classification_metrics(all_gt_rel, all_pred_rel)
        relevance_auprc = float(rel_metrics["auprc"])
        relevance_f1 = float(rel_metrics["f1"])
        relevance_prec = float(rel_metrics["precision"])
        relevance_rec = float(rel_metrics["recall"])
    elif len(all_gt_rel) > 0:
        relevance_auprc = 0.5
        relevance_f1 = 0.5
        relevance_prec = 0.5
        relevance_rec = 0.5
    else:
        relevance_auprc = 0.0
        relevance_f1 = 0.0
        relevance_prec = 0.0
        relevance_rec = 0.0

    # State metrics
    if len(all_gt_state) > 0:
        cm = multiclass_confusion_matrix(all_gt_state, all_pred_state, classes=4)
        st_metrics = multiclass_metrics(cm)
        state_acc = float(st_metrics["accuracy"])
        state_macro_f1 = float(st_metrics["macro_f1"])
    else:
        state_acc = 0.0
        state_macro_f1 = 0.0

    # Round metrics
    if len(all_gt_round) > 0 and len(np.unique(all_gt_round)) > 1:
        rd_metrics = binary_classification_metrics(all_gt_round, all_pred_round)
        round_f1 = float(rd_metrics["f1"])
    elif len(all_gt_round) > 0:
        round_f1 = 0.5
    else:
        round_f1 = 0.0

    # Maneuver metrics
    if len(all_gt_maneuver) > 0:
        mv_metrics = multilabel_metrics(all_gt_maneuver, all_pred_maneuver)
        maneuver_macro_f1 = float(mv_metrics["macro_f1"])
    else:
        maneuver_macro_f1 = 0.0

    # Composite selection score
    score_inputs = {
        "traffic_light_tiny_ap": det_map.get("ap_tl_sub8px", det_map.get("ap_small", 0.0)),
        "arrow_ap": det_map.get("ap_arrow_50", 0.0),
        "state_macro_f1": state_macro_f1 if not math.isnan(state_macro_f1) else 0.0,
        "round_f1": round_f1 if not math.isnan(round_f1) else 0.0,
        "maneuver_macro_f1": (
            maneuver_macro_f1 if not math.isnan(maneuver_macro_f1) else 0.0
        ),
        "relevance_auprc": relevance_auprc if not math.isnan(relevance_auprc) else 0.0,
    }
    selection_score = validation_selection_score(score_inputs)

    mean_losses = {
        name: (val / max(1, loss_batch_count))
        for name, val in loss_totals.items()
    }

    granular_res = None
    if granular_scale_metrics:
        granular_res = compute_granular_scale_metrics(
            pred_boxes_list,
            pred_scores_list,
            pred_classes_list,
            gt_boxes_list,
            gt_classes_list,
            target_class=0,
            image_shape=(int(img_h), int(img_w)),
            conf_threshold=conf_threshold,
        )

    out: dict[str, Any] = {
        "selection_score": selection_score,
        "mean_losses": mean_losses,
        "detection": det_map,
        "relevance": {
            "auprc": relevance_auprc,
            "f1": relevance_f1,
            "precision": relevance_prec,
            "recall": relevance_rec,
            "relevant_red_recall": relevant_red_recall,
        },
        "attributes": {
            "state_accuracy": state_acc,
            "state_macro_f1": state_macro_f1,
            "round_f1": round_f1,
            "maneuver_macro_f1": maneuver_macro_f1,
        },
        "samples_evaluated": len(pred_boxes_list),
    }
    if granular_res is not None:
        out["granular_scale"] = granular_res
    return out
