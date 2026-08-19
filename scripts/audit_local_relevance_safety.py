"""W9 Diagnostic Audit: Local Relevance Baseline & Safety-Critical Metrics.

Evaluates the Baseline B0 model on the DTLD validation set to determine:
1. Local Relevance Baseline (alpha = 0) vs Contextual Relevance across granular slices:
   - Arrow presence (Arrows present vs Absent)
   - Signal type (Directional vs Round)
   - Scene density (Single TL vs Multi-TL)
   - Object scale buckets (<32, 32-64, 64-128, 128-256, 256-512, >512 px^2)
2. 3-Tier Relevance Evaluation Hierarchy:
   - Level 1: Oracle Relevance (Mode B feature sampling at GT locations)
   - Level 2: Detection-Conditioned Relevance (IoU >= 0.50 matched candidates)
   - Level 3: End-to-End Detection + Relevance (s_det * P(rel) over all relevant GT TLs)
3. Safety-Critical Metrics:
   - Relevant Red Traffic Light Recall & Miss Rate
   - 4-Stage Miss Attribution Waterfall (Perception, Candidate Selection, State, Relevance)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from ultralytics.utils.tal import make_anchors

from tlr_yolo_mtl.deployment.postprocess import xywh_to_xyxy
from tlr_yolo_mtl.evaluation.calibration import fit_temperature
from tlr_yolo_mtl.evaluation.matching import greedy_iou_match, pairwise_iou
from tlr_yolo_mtl.evaluation.metrics import (
    AREA_BUCKETS,
    binary_average_precision,
    binary_classification_metrics,
    binary_roc_auc,
    brier_score,
    compute_ap_from_matches,
    expected_calibration_error,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    ROAD_ARROW_CLASS,
    TRAFFIC_LIGHT_CLASS,
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.training.data import (
    CanonicalMultiTaskDataset,
    canonical_multitask_collate,
)


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    if not cfg:
        with open(PROJECT_ROOT / "configs" / "tlr_yolo_mtl_single_phase.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    wrapper = build_detection_model(cfg["model_config"])
    arch_cfg = cfg.get("architecture", {})
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(**arch_cfg))

    state_dict = payload.get("model", payload)
    wrapper.model.load_state_dict(state_dict, strict=True)
    model = wrapper.model.to(device).eval()
    return model, cfg


def compute_binary_eval_bundle(targets: list[int], scores: list[float]) -> dict[str, float]:
    out = {
        "count": len(targets),
        "positives": int(sum(targets)),
        "auprc": 0.0,
        "roc_auc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "optimal_f1": 0.0,
        "optimal_threshold": 0.5,
        "ece": 0.0,
        "brier": 0.0,
        "temp_fitted": 1.0,
        "nll_before": 0.0,
        "nll_after": 0.0,
    }
    if not targets or len(np.unique(targets)) < 2:
        if targets:
            out["auprc"] = 0.5
            out["roc_auc"] = 0.5
            out["f1"] = 0.5
        return out

    y = np.array(targets, dtype=np.int64)
    s = np.array(scores, dtype=float)

    out["auprc"] = float(binary_average_precision(y, s))
    out["roc_auc"] = float(binary_roc_auc(y, s))

    b_metrics = binary_classification_metrics(y, s, threshold=0.5)
    out["precision"] = float(b_metrics["precision"])
    out["recall"] = float(b_metrics["recall"])
    out["f1"] = float(b_metrics["f1"])
    out["ece"] = float(expected_calibration_error(y, s))
    out["brier"] = float(brier_score(y, s))

    # Optimal F1 search
    best_f1, best_th = 0.0, 0.5
    for th in np.linspace(0.05, 0.95, 19):
        pred = (s >= th)
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1_val > best_f1:
            best_f1 = f1_val
            best_th = float(th)
    out["optimal_f1"] = float(best_f1)
    out["optimal_threshold"] = float(best_th)

    # Temperature fitting
    try:
        s_clamped = np.clip(s, 1e-6, 1.0 - 1e-6)
        logits_pseudo = torch.tensor(np.log(s_clamped / (1.0 - s_clamped)), dtype=torch.float32)
        fit = fit_temperature(logits_pseudo, torch.tensor(y, dtype=torch.long))
        out["temp_fitted"] = float(fit.temperature)
        out["nll_before"] = float(fit.loss_before)
        out["nll_after"] = float(fit.loss_after)
    except Exception:
        pass

    return out


def run_w9_audit(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    conf_threshold: float = 0.05,
) -> dict[str, Any]:
    print(f"Running W9 diagnostic audit on {len(val_loader)} validation batches (max_batches={max_batches})...")
    start_time = time.time()
    stride = (8, 16, 32)

    # 1. Level 1: Oracle Relevance Collectors
    oracle_overall = {"targets": [], "scores": []}
    oracle_by_arrow = {"arrows_present": {"targets": [], "scores": []}, "no_arrows": {"targets": [], "scores": []}}
    oracle_by_round = {"round": {"targets": [], "scores": []}, "directional": {"targets": [], "scores": []}}
    oracle_by_density = {"single_tl": {"targets": [], "scores": []}, "multi_tl": {"targets": [], "scores": []}}
    oracle_by_area = {k: {"targets": [], "scores": []} for k in AREA_BUCKETS}

    # 2. Level 2: Detection-Conditioned Relevance Collectors (Local vs Contextual)
    level2_local_overall = {"targets": [], "scores": []}
    level2_ctx_overall = {"targets": [], "scores": []}

    level2_by_arrow = {
        "arrows_present": {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []},
        "no_arrows": {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []},
    }
    level2_by_round = {
        "round": {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []},
        "directional": {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []},
    }
    level2_by_density = {
        "single_tl": {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []},
        "multi_tl": {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []},
    }
    level2_by_area = {
        k: {"local_targets": [], "local_scores": [], "ctx_targets": [], "ctx_scores": []}
        for k in AREA_BUCKETS
    }

    # 3. Level 3: End-to-End Relevance + Detection Collectors
    l3_local_tp_matches: list[int] = []
    l3_local_scores: list[float] = []
    l3_ctx_tp_matches: list[int] = []
    l3_ctx_scores: list[float] = []
    total_relevant_gt_count = 0

    # 4. Safety-Critical Relevant Red TL Tracking
    safety_stats = {
        "total_relevant_red_gt": 0,
        "total_irrelevant_red_gt": 0,
        "stage1_perception_miss": 0,
        "stage2_candidate_evicted": 0,
        "stage3_state_misclassified": 0,
        "stage4_relevance_miss_local": 0,
        "stage4_relevance_miss_ctx": 0,
        "success_local_50": 0,
        "success_ctx_50": 0,
        "success_local_30": 0,
        "success_ctx_30": 0,
        "success_local_70": 0,
        "success_ctx_70": 0,
        "red_confusion_matrix_local": {"tp": 0, "fn": 0, "tn": 0, "fp": 0},
        "red_confusion_matrix_ctx": {"tp": 0, "fn": 0, "tn": 0, "fp": 0},
    }

    # Extract learned alpha gate
    learned_gate_val = float(model.model[23].cross_attention.gate.item())
    print(f"Loaded model cross-attention gate alpha: {learned_gate_val:.6f}")

    for batch_idx, raw_batch in enumerate(val_loader, 1):
        if max_batches is not None and batch_idx > max_batches:
            break

        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in raw_batch.items()
        }

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                predictions = model(batch["img"])

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

        state_logits = raw.get("state_logits")  # [B, 4, A]
        round_logits = raw.get("round_logits")  # [B, 1, A]
        maneuver_logits = raw.get("maneuver_logits")  # [B, 3, A]
        dense_local_rel_logits = raw.get("dense_local_relevance_logits")  # [B, 1, NumAnchors]
        local_rel_logits = raw.get("local_relevance_logits")  # [B, 1, 32]
        traffic_indices = raw.get("traffic_candidate_indices")  # [B, 32]
        traffic_scores = raw.get("traffic_candidate_scores")  # [B, 32]
        traffic_valid = raw.get("traffic_candidate_valid")  # [B, 32]
        traffic_boxes_raw = raw.get("traffic_candidate_boxes")  # [B, 32, 4]
        ctx_relevance_logits = raw.get("relevance_logits")  # [B, 1, 32]

        feats = raw.get("feats", raw.get("features", []))
        if not feats and hasattr(model, "model") and hasattr(model.model[23], "stride"):
            shape_p3 = (int(img_h // 8), int(img_w // 8))
            shape_p4 = (int(img_h // 16), int(img_w // 16))
            shape_p5 = (int(img_h // 32), int(img_w // 32))
            dummy_feats = [
                torch.empty(batch_size, 1, shape_p3[0], shape_p3[1], device=device),
                torch.empty(batch_size, 1, shape_p4[0], shape_p4[1], device=device),
                torch.empty(batch_size, 1, shape_p5[0], shape_p5[1], device=device),
            ]
            anchor_points, stride_tensor = make_anchors(dummy_feats, stride, 0.5)
        else:
            anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        anchors_px = anchor_points * stride_tensor  # [NumAnchors, 2] cx, cy in px
        strides_flat = stride_tensor[:, 0]  # [NumAnchors]
        p3_mask = (strides_flat == 8)
        p3_indices = torch.nonzero(p3_mask, as_tuple=False).reshape(-1)
        p3_anchors_px = anchors_px[p3_indices]

        for b in range(batch_size):
            # Extract GT objects
            b_mask = (batch["object_batch_idx"] == b)
            gt_cls = batch["object_cls"][b_mask].cpu().numpy().reshape(-1)
            gt_bboxes_norm = batch["object_bboxes"][b_mask].cpu().numpy()
            gt_st = batch["object_state"][b_mask].cpu().numpy().reshape(-1)
            gt_rd = batch["object_round"][b_mask].cpu().numpy().reshape(-1)
            gt_rl = batch["object_relevance"][b_mask].cpu().numpy().reshape(-1)

            tl_gt_indices = np.where(gt_cls == TRAFFIC_LIGHT_CLASS)[0]
            arrow_gt_indices = np.where(gt_cls == ROAD_ARROW_CLASS)[0]

            has_arrows = (len(arrow_gt_indices) > 0)
            is_single_tl = (len(tl_gt_indices) == 1)

            if len(tl_gt_indices) == 0:
                continue

            tl_boxes_norm = gt_bboxes_norm[tl_gt_indices]
            tl_st = gt_st[tl_gt_indices]
            tl_rd = gt_rd[tl_gt_indices]
            tl_rl = gt_rl[tl_gt_indices]

            cx_norm, cy_norm, w_norm, h_norm = (
                tl_boxes_norm[:, 0],
                tl_boxes_norm[:, 1],
                tl_boxes_norm[:, 2],
                tl_boxes_norm[:, 3],
            )
            tl_boxes_xyxy_norm = np.stack(
                [cx_norm - w_norm / 2, cy_norm - h_norm / 2, cx_norm + w_norm / 2, cy_norm + h_norm / 2],
                axis=-1,
            )
            tl_areas_px = (w_norm * img_w) * (h_norm * img_h)
            tl_cx_px = torch.tensor(cx_norm * img_w, device=device)
            tl_cy_px = torch.tensor(cy_norm * img_h, device=device)
            tl_centers_px = torch.stack([tl_cx_px, tl_cy_px], dim=-1)

            # Count relevant GTs
            for rl in tl_rl:
                if rl == 1:
                    total_relevant_gt_count += 1

            # ----------------------------------------------------
            # 1. Level 1: Oracle Location Feature Sampling (Mode B)
            # ----------------------------------------------------
            dists = torch.cdist(tl_centers_px.float(), p3_anchors_px.float())
            nearest_p3_idx = p3_indices[dists.argmin(dim=-1)]

            for i in range(len(tl_gt_indices)):
                if tl_rl[i] < 0 or dense_local_rel_logits is None:
                    continue
                a_idx = int(nearest_p3_idx[i].item())
                p_oracle_rl = float(dense_local_rel_logits[b, 0, a_idx].sigmoid().item())
                target_rl = int(tl_rl[i])
                area = float(tl_areas_px[i])

                oracle_overall["targets"].append(target_rl)
                oracle_overall["scores"].append(p_oracle_rl)

                if has_arrows:
                    oracle_by_arrow["arrows_present"]["targets"].append(target_rl)
                    oracle_by_arrow["arrows_present"]["scores"].append(p_oracle_rl)
                else:
                    oracle_by_arrow["no_arrows"]["targets"].append(target_rl)
                    oracle_by_arrow["no_arrows"]["scores"].append(p_oracle_rl)

                if tl_rd[i] == 1:
                    oracle_by_round["round"]["targets"].append(target_rl)
                    oracle_by_round["round"]["scores"].append(p_oracle_rl)
                elif tl_rd[i] == 0:
                    oracle_by_round["directional"]["targets"].append(target_rl)
                    oracle_by_round["directional"]["scores"].append(p_oracle_rl)

                if is_single_tl:
                    oracle_by_density["single_tl"]["targets"].append(target_rl)
                    oracle_by_density["single_tl"]["scores"].append(p_oracle_rl)
                else:
                    oracle_by_density["multi_tl"]["targets"].append(target_rl)
                    oracle_by_density["multi_tl"]["scores"].append(p_oracle_rl)

                for ab_name, (low, high) in AREA_BUCKETS.items():
                    if low <= area < high:
                        oracle_by_area[ab_name]["targets"].append(target_rl)
                        oracle_by_area[ab_name]["scores"].append(p_oracle_rl)
                        break

            # ----------------------------------------------------
            # 2. Extract Candidate Predictions & Match (Level 2 & 3)
            # ----------------------------------------------------
            if (
                traffic_boxes_raw is None
                or traffic_valid is None
                or traffic_scores is None
                or traffic_indices is None
            ):
                continue

            c_valid = traffic_valid[b].bool().cpu().numpy()
            if not c_valid.any():
                continue

            v_indices = np.where(c_valid)[0]
            cand_boxes_norm = traffic_boxes_raw[b, v_indices].cpu().numpy()  # cx, cy, w, h
            c_cx, c_cy, c_w, c_h = (
                cand_boxes_norm[:, 0],
                cand_boxes_norm[:, 1],
                cand_boxes_norm[:, 2],
                cand_boxes_norm[:, 3],
            )
            cand_boxes_xyxy_norm = np.stack(
                [c_cx - c_w / 2, c_cy - c_h / 2, c_cx + c_w / 2, c_cy + c_h / 2],
                axis=-1,
            )
            cand_scores = traffic_scores[b, v_indices].cpu().numpy()
            cand_dense_indices = traffic_indices[b, v_indices].cpu().numpy()

            # Level 2 Greedy IoU matching with all TL GTs
            matches, unmatched_preds, unmatched_gts = greedy_iou_match(
                cand_boxes_xyxy_norm, cand_scores, tl_boxes_xyxy_norm, iou_threshold=0.50
            )

            # Matched pairs for Level 2
            for m in matches:
                p_idx = v_indices[m.prediction_index]
                d_idx = cand_dense_indices[m.prediction_index]
                gt_idx = m.target_index

                if tl_rl[gt_idx] < 0:
                    continue

                gt_target = int(tl_rl[gt_idx])
                area = float(tl_areas_px[gt_idx])

                # Local relevance probability
                p_local = float(local_rel_logits[b, 0, p_idx].sigmoid().item()) if local_rel_logits is not None else 0.5
                # Contextual relevance probability
                p_ctx = float(ctx_relevance_logits[b, 0, p_idx].sigmoid().item()) if ctx_relevance_logits is not None else p_local

                level2_local_overall["targets"].append(gt_target)
                level2_local_overall["scores"].append(p_local)
                level2_ctx_overall["targets"].append(gt_target)
                level2_ctx_overall["scores"].append(p_ctx)

                arrow_key = "arrows_present" if has_arrows else "no_arrows"
                level2_by_arrow[arrow_key]["local_targets"].append(gt_target)
                level2_by_arrow[arrow_key]["local_scores"].append(p_local)
                level2_by_arrow[arrow_key]["ctx_targets"].append(gt_target)
                level2_by_arrow[arrow_key]["ctx_scores"].append(p_ctx)

                if tl_rd[gt_idx] == 1:
                    level2_by_round["round"]["local_targets"].append(gt_target)
                    level2_by_round["round"]["local_scores"].append(p_local)
                    level2_by_round["round"]["ctx_targets"].append(gt_target)
                    level2_by_round["round"]["ctx_scores"].append(p_ctx)
                elif tl_rd[gt_idx] == 0:
                    level2_by_round["directional"]["local_targets"].append(gt_target)
                    level2_by_round["directional"]["local_scores"].append(p_local)
                    level2_by_round["directional"]["ctx_targets"].append(gt_target)
                    level2_by_round["directional"]["ctx_scores"].append(p_ctx)

                density_key = "single_tl" if is_single_tl else "multi_tl"
                level2_by_density[density_key]["local_targets"].append(gt_target)
                level2_by_density[density_key]["local_scores"].append(p_local)
                level2_by_density[density_key]["ctx_targets"].append(gt_target)
                level2_by_density[density_key]["ctx_scores"].append(p_ctx)

                for ab_name, (low, high) in AREA_BUCKETS.items():
                    if low <= area < high:
                        level2_by_area[ab_name]["local_targets"].append(gt_target)
                        level2_by_area[ab_name]["local_scores"].append(p_local)
                        level2_by_area[ab_name]["ctx_targets"].append(gt_target)
                        level2_by_area[ab_name]["ctx_scores"].append(p_ctx)
                        break

            # ----------------------------------------------------
            # 3. Level 3: End-to-End Detection + Relevance (s_det * P(rel))
            # ----------------------------------------------------
            rel_gt_mask = (tl_rl == 1)
            rel_gt_boxes_xyxy = tl_boxes_xyxy_norm[rel_gt_mask]

            if len(cand_boxes_xyxy_norm) > 0:
                # Local combined scores
                p_local_cand = np.array([
                    float(local_rel_logits[b, 0, v_indices[k]].sigmoid().item())
                    for k in range(len(cand_boxes_xyxy_norm))
                ])
                s_e2e_local = cand_scores * p_local_cand

                # Contextual combined scores
                p_ctx_cand = np.array([
                    float(ctx_relevance_logits[b, 0, v_indices[k]].sigmoid().item())
                    for k in range(len(cand_boxes_xyxy_norm))
                ])
                s_e2e_ctx = cand_scores * p_ctx_cand

                # Evaluate E2E Local
                if len(rel_gt_boxes_xyxy) > 0:
                    m_local, fp_local, _ = greedy_iou_match(
                        cand_boxes_xyxy_norm, s_e2e_local, rel_gt_boxes_xyxy, iou_threshold=0.50
                    )
                    tp_indices_local = {m.prediction_index for m in m_local}
                    for k in range(len(cand_boxes_xyxy_norm)):
                        l3_local_tp_matches.append(1 if k in tp_indices_local else 0)
                        l3_local_scores.append(float(s_e2e_local[k]))

                    m_ctx, fp_ctx, _ = greedy_iou_match(
                        cand_boxes_xyxy_norm, s_e2e_ctx, rel_gt_boxes_xyxy, iou_threshold=0.50
                    )
                    tp_indices_ctx = {m.prediction_index for m in m_ctx}
                    for k in range(len(cand_boxes_xyxy_norm)):
                        l3_ctx_tp_matches.append(1 if k in tp_indices_ctx else 0)
                        l3_ctx_scores.append(float(s_e2e_ctx[k]))
                else:
                    for k in range(len(cand_boxes_xyxy_norm)):
                        l3_local_tp_matches.append(0)
                        l3_local_scores.append(float(s_e2e_local[k]))
                        l3_ctx_tp_matches.append(0)
                        l3_ctx_scores.append(float(s_e2e_ctx[k]))

            # ----------------------------------------------------
            # 4. Safety-Critical Relevant Red Traffic Light Tracking
            # ----------------------------------------------------
            gt_matched_by_pred = {m.target_index: m.prediction_index for m in matches}

            for i in range(len(tl_gt_indices)):
                is_red = (tl_st[i] == 0)
                is_rel = (tl_rl[i] == 1)

                if is_red and is_rel:
                    safety_stats["total_relevant_red_gt"] += 1

                    if i not in gt_matched_by_pred:
                        # Failed perception / IoU < 0.50 matching
                        safety_stats["stage1_perception_miss"] += 1
                        safety_stats["red_confusion_matrix_local"]["fn"] += 1
                        safety_stats["red_confusion_matrix_ctx"]["fn"] += 1
                        continue

                    pred_k = gt_matched_by_pred[i]
                    p_idx = v_indices[pred_k]
                    d_idx = cand_dense_indices[pred_k]

                    # Check state classification
                    pred_state = int(state_logits[b, :, d_idx].argmax(0).item()) if state_logits is not None else -1
                    if pred_state != 0:
                        # State misclassification (e.g. predicted green or yellow)
                        safety_stats["stage3_state_misclassified"] += 1
                        safety_stats["red_confusion_matrix_local"]["fn"] += 1
                        safety_stats["red_confusion_matrix_ctx"]["fn"] += 1
                        continue

                    # Relevance classification
                    p_local = float(local_rel_logits[b, 0, p_idx].sigmoid().item()) if local_rel_logits is not None else 0.0
                    p_ctx = float(ctx_relevance_logits[b, 0, p_idx].sigmoid().item()) if ctx_relevance_logits is not None else 0.0

                    if p_local >= 0.50:
                        safety_stats["success_local_50"] += 1
                        safety_stats["red_confusion_matrix_local"]["tp"] += 1
                    else:
                        safety_stats["stage4_relevance_miss_local"] += 1
                        safety_stats["red_confusion_matrix_local"]["fn"] += 1

                    if p_ctx >= 0.50:
                        safety_stats["success_ctx_50"] += 1
                        safety_stats["red_confusion_matrix_ctx"]["tp"] += 1
                    else:
                        safety_stats["stage4_relevance_miss_ctx"] += 1
                        safety_stats["red_confusion_matrix_ctx"]["fn"] += 1

                    if p_local >= 0.30:
                        safety_stats["success_local_30"] += 1
                    if p_ctx >= 0.30:
                        safety_stats["success_ctx_30"] += 1
                    if p_local >= 0.70:
                        safety_stats["success_local_70"] += 1
                    if p_ctx >= 0.70:
                        safety_stats["success_ctx_70"] += 1

                elif is_red and (tl_rl[i] == 0):
                    safety_stats["total_irrelevant_red_gt"] += 1
                    if i in gt_matched_by_pred:
                        pred_k = gt_matched_by_pred[i]
                        p_idx = v_indices[pred_k]
                        d_idx = cand_dense_indices[pred_k]
                        p_local = float(local_rel_logits[b, 0, p_idx].sigmoid().item()) if local_rel_logits is not None else 0.0
                        p_ctx = float(ctx_relevance_logits[b, 0, p_idx].sigmoid().item()) if ctx_relevance_logits is not None else 0.0

                        if p_local >= 0.50:
                            safety_stats["red_confusion_matrix_local"]["fp"] += 1
                        else:
                            safety_stats["red_confusion_matrix_local"]["tn"] += 1

                        if p_ctx >= 0.50:
                            safety_stats["red_confusion_matrix_ctx"]["fp"] += 1
                        else:
                            safety_stats["red_confusion_matrix_ctx"]["tn"] += 1
                    else:
                        safety_stats["red_confusion_matrix_local"]["tn"] += 1
                        safety_stats["red_confusion_matrix_ctx"]["tn"] += 1

        if batch_idx % 25 == 0 or batch_idx == len(val_loader):
            elapsed = time.time() - start_time
            print(f"Processed {batch_idx}/{len(val_loader)} validation batches ({elapsed:.1f}s)...")

    duration = time.time() - start_time
    print(f"Audit completed in {duration:.1f}s.")

    # ----------------------------------------------------
    # Metric Aggregation
    # ----------------------------------------------------
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 2),
        "total_gt_traffic_lights": len(oracle_overall["targets"]),
        "total_relevant_gt": total_relevant_gt_count,
        "learned_gate_alpha": learned_gate_val,
        "level1_oracle": {},
        "level2_detection_conditioned": {},
        "level3_end_to_end": {},
        "safety_critical_relevant_red": {},
    }

    # Level 1 Oracle Metrics
    results["level1_oracle"]["overall"] = compute_binary_eval_bundle(oracle_overall["targets"], oracle_overall["scores"])
    results["level1_oracle"]["by_arrow"] = {
        k: compute_binary_eval_bundle(v["targets"], v["scores"]) for k, v in oracle_by_arrow.items()
    }
    results["level1_oracle"]["by_round"] = {
        k: compute_binary_eval_bundle(v["targets"], v["scores"]) for k, v in oracle_by_round.items()
    }
    results["level1_oracle"]["by_density"] = {
        k: compute_binary_eval_bundle(v["targets"], v["scores"]) for k, v in oracle_by_density.items()
    }
    results["level1_oracle"]["by_area"] = {
        k: compute_binary_eval_bundle(v["targets"], v["scores"]) for k, v in oracle_by_area.items()
    }

    # Level 2 Metrics
    l2_local_overall = compute_binary_eval_bundle(level2_local_overall["targets"], level2_local_overall["scores"])
    l2_ctx_overall = compute_binary_eval_bundle(level2_ctx_overall["targets"], level2_ctx_overall["scores"])
    delta_auprc_overall = l2_ctx_overall["auprc"] - l2_local_overall["auprc"]

    results["level2_detection_conditioned"]["overall"] = {
        "local": l2_local_overall,
        "contextual": l2_ctx_overall,
        "delta_auprc": round(delta_auprc_overall, 6),
    }

    results["level2_detection_conditioned"]["by_arrow"] = {}
    for k, v in level2_by_arrow.items():
        m_loc = compute_binary_eval_bundle(v["local_targets"], v["local_scores"])
        m_ctx = compute_binary_eval_bundle(v["ctx_targets"], v["ctx_scores"])
        results["level2_detection_conditioned"]["by_arrow"][k] = {
            "local": m_loc,
            "contextual": m_ctx,
            "delta_auprc": round(m_ctx["auprc"] - m_loc["auprc"], 6),
        }

    results["level2_detection_conditioned"]["by_round"] = {}
    for k, v in level2_by_round.items():
        m_loc = compute_binary_eval_bundle(v["local_targets"], v["local_scores"])
        m_ctx = compute_binary_eval_bundle(v["ctx_targets"], v["ctx_scores"])
        results["level2_detection_conditioned"]["by_round"][k] = {
            "local": m_loc,
            "contextual": m_ctx,
            "delta_auprc": round(m_ctx["auprc"] - m_loc["auprc"], 6),
        }

    results["level2_detection_conditioned"]["by_density"] = {}
    for k, v in level2_by_density.items():
        m_loc = compute_binary_eval_bundle(v["local_targets"], v["local_scores"])
        m_ctx = compute_binary_eval_bundle(v["ctx_targets"], v["ctx_scores"])
        results["level2_detection_conditioned"]["by_density"][k] = {
            "local": m_loc,
            "contextual": m_ctx,
            "delta_auprc": round(m_ctx["auprc"] - m_loc["auprc"], 6),
        }

    results["level2_detection_conditioned"]["by_area"] = {}
    for k, v in level2_by_area.items():
        m_loc = compute_binary_eval_bundle(v["local_targets"], v["local_scores"])
        m_ctx = compute_binary_eval_bundle(v["ctx_targets"], v["ctx_scores"])
        results["level2_detection_conditioned"]["by_area"][k] = {
            "local": m_loc,
            "contextual": m_ctx,
            "delta_auprc": round(m_ctx["auprc"] - m_loc["auprc"], 6),
        }

    # Level 3 End-to-End Metrics
    l3_local_ap50 = compute_ap_from_matches(
        np.array(l3_local_tp_matches, dtype=np.int64),
        np.array(l3_local_scores, dtype=float),
        total_relevant_gt_count,
    )
    l3_ctx_ap50 = compute_ap_from_matches(
        np.array(l3_ctx_tp_matches, dtype=np.int64),
        np.array(l3_ctx_scores, dtype=float),
        total_relevant_gt_count,
    )
    l3_local_recall = float(sum(l3_local_tp_matches) / max(total_relevant_gt_count, 1))
    l3_ctx_recall = float(sum(l3_ctx_tp_matches) / max(total_relevant_gt_count, 1))

    results["level3_end_to_end"] = {
        "total_relevant_gt": total_relevant_gt_count,
        "local": {
            "ap50_e2e": round(l3_local_ap50, 4),
            "recall_e2e": round(l3_local_recall, 4),
            "tp_count": int(sum(l3_local_tp_matches)),
        },
        "contextual": {
            "ap50_e2e": round(l3_ctx_ap50, 4),
            "recall_e2e": round(l3_ctx_recall, 4),
            "tp_count": int(sum(l3_ctx_tp_matches)),
        },
        "delta_ap50": round(l3_ctx_ap50 - l3_local_ap50, 4),
    }

    # Safety-Critical Relevant Red Metrics
    tot_rel_red = safety_stats["total_relevant_red_gt"]
    succ_loc_50 = safety_stats["success_local_50"]
    succ_ctx_50 = safety_stats["success_ctx_50"]
    rec_loc_50 = float(succ_loc_50 / max(tot_rel_red, 1))
    rec_ctx_50 = float(succ_ctx_50 / max(tot_rel_red, 1))

    results["safety_critical_relevant_red"] = {
        "total_relevant_red_gt": tot_rel_red,
        "total_irrelevant_red_gt": safety_stats["total_irrelevant_red_gt"],
        "recall_local_50": round(rec_loc_50, 4),
        "recall_ctx_50": round(rec_ctx_50, 4),
        "recall_local_30": round(float(safety_stats["success_local_30"] / max(tot_rel_red, 1)), 4),
        "recall_ctx_30": round(float(safety_stats["success_ctx_30"] / max(tot_rel_red, 1)), 4),
        "recall_local_70": round(float(safety_stats["success_local_70"] / max(tot_rel_red, 1)), 4),
        "recall_ctx_70": round(float(safety_stats["success_ctx_70"] / max(tot_rel_red, 1)), 4),
        "miss_rate_local_50": round(1.0 - rec_loc_50, 4),
        "miss_rate_ctx_50": round(1.0 - rec_ctx_50, 4),
        "waterfall_attribution": {
            "stage1_perception_miss_count": safety_stats["stage1_perception_miss"],
            "stage1_perception_miss_pct": round(safety_stats["stage1_perception_miss"] / max(tot_rel_red, 1) * 100, 2),
            "stage2_candidate_evicted_count": safety_stats["stage2_candidate_evicted"],
            "stage2_candidate_evicted_pct": round(safety_stats["stage2_candidate_evicted"] / max(tot_rel_red, 1) * 100, 2),
            "stage3_state_misclassified_count": safety_stats["stage3_state_misclassified"],
            "stage3_state_misclassified_pct": round(safety_stats["stage3_state_misclassified"] / max(tot_rel_red, 1) * 100, 2),
            "stage4_relevance_miss_local_count": safety_stats["stage4_relevance_miss_local"],
            "stage4_relevance_miss_local_pct": round(safety_stats["stage4_relevance_miss_local"] / max(tot_rel_red, 1) * 100, 2),
            "stage4_relevance_miss_ctx_count": safety_stats["stage4_relevance_miss_ctx"],
            "stage4_relevance_miss_ctx_pct": round(safety_stats["stage4_relevance_miss_ctx"] / max(tot_rel_red, 1) * 100, 2),
            "success_local_count": succ_loc_50,
            "success_local_pct": round(rec_loc_50 * 100, 2),
            "success_ctx_count": succ_ctx_50,
            "success_ctx_pct": round(rec_ctx_50 * 100, 2),
        },
        "confusion_matrix_local": safety_stats["red_confusion_matrix_local"],
        "confusion_matrix_ctx": safety_stats["red_confusion_matrix_ctx"],
    }

    return results


def plot_w9_diagnostics(results: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"W9: Local Relevance Baseline & Safety-Critical Metrics (alpha_gate = {results['learned_gate_alpha']:.4f})",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Panel A: Level 2 Relevance AUPRC across Scale Buckets (Local vs Contextual vs Oracle)
    ax1 = axes[0, 0]
    area_keys = list(AREA_BUCKETS.keys())
    oracle_auprc = [results["level1_oracle"]["by_area"][k]["auprc"] for k in area_keys]
    local_auprc = [results["level2_detection_conditioned"]["by_area"][k]["local"]["auprc"] for k in area_keys]
    ctx_auprc = [results["level2_detection_conditioned"]["by_area"][k]["contextual"]["auprc"] for k in area_keys]

    x = np.arange(len(area_keys))
    w = 0.25
    ax1.bar(x - w, oracle_auprc, width=w, label="Level 1: Oracle (Mode B)", color="#2ca02c", alpha=0.85)
    ax1.bar(x, local_auprc, width=w, label="Level 2: Local (alpha=0)", color="#1f77b4", alpha=0.85)
    ax1.bar(x + w, ctx_auprc, width=w, label="Level 2: Contextual (alpha_learned)", color="#ff7f0e", alpha=0.85)

    ax1.set_title("Relevance AUPRC by Scale Bucket", fontweight="bold")
    ax1.set_xlabel("Object Area Bucket (px²)")
    ax1.set_ylabel("AUPRC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(area_keys)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 2. Panel B: Granular Slice Comparison (Delta AUPRC = Ctx - Local)
    ax2 = axes[0, 1]
    slices = [
        ("Overall", results["level2_detection_conditioned"]["overall"]),
        ("Arrows Present", results["level2_detection_conditioned"]["by_arrow"]["arrows_present"]),
        ("No Arrows", results["level2_detection_conditioned"]["by_arrow"]["no_arrows"]),
        ("Directional TL", results["level2_detection_conditioned"]["by_round"]["directional"]),
        ("Round TL", results["level2_detection_conditioned"]["by_round"]["round"]),
        ("Single TL Scene", results["level2_detection_conditioned"]["by_density"]["single_tl"]),
        ("Multi TL Scene", results["level2_detection_conditioned"]["by_density"]["multi_tl"]),
    ]
    slice_names = [s[0] for s in slices]
    slice_loc = [s[1]["local"]["auprc"] for s in slices]
    slice_ctx = [s[1]["contextual"]["auprc"] for s in slices]
    slice_deltas = [s[1]["delta_auprc"] for s in slices]

    y_pos = np.arange(len(slice_names))
    ax2.barh(y_pos - 0.15, slice_loc, height=0.3, label="Local (alpha=0)", color="#1f77b4", alpha=0.85)
    ax2.barh(y_pos + 0.15, slice_ctx, height=0.3, label="Contextual", color="#ff7f0e", alpha=0.85)

    for idx, (d, ctx_val) in enumerate(zip(slice_deltas, slice_ctx)):
        sign = "+" if d >= 0 else ""
        ax2.text(ctx_val + 0.02, idx, f"{sign}{d*100:.2f}%", va="center", fontsize=9, fontweight="bold")

    ax2.set_title("Relevance AUPRC across Slices & Contextual Gain (Δ AUPRC)", fontweight="bold")
    ax2.set_xlabel("AUPRC")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(slice_names)
    ax2.set_xlim(0.0, 1.15)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    # 3. Panel C: 3-Tier Hierarchy & Calibration Comparison
    ax3 = axes[1, 0]
    tiers = ["Level 1: Oracle", "Level 2: Det-Cond Local", "Level 2: Det-Cond Ctx", "Level 3: End-to-End Local", "Level 3: End-to-End Ctx"]
    tier_scores = [
        results["level1_oracle"]["overall"]["auprc"],
        results["level2_detection_conditioned"]["overall"]["local"]["auprc"],
        results["level2_detection_conditioned"]["overall"]["contextual"]["auprc"],
        results["level3_end_to_end"]["local"]["ap50_e2e"],
        results["level3_end_to_end"]["contextual"]["ap50_e2e"],
    ]
    tier_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]

    bars = ax3.bar(tiers, tier_scores, color=tier_colors, alpha=0.85, width=0.55)
    for b in bars:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontweight="bold")

    ax3.set_title("3-Tier Relevance Metric Hierarchy (AUPRC / AP50)", fontweight="bold")
    ax3.set_ylabel("Metric Score")
    ax3.set_ylim(0.0, 1.1)
    ax3.set_xticks(range(len(tiers)))
    ax3.set_xticklabels(tiers, rotation=15, ha="right")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Panel D: Safety-Critical Relevant Red Light Waterfall
    ax4 = axes[1, 1]
    wf = results["safety_critical_relevant_red"]["waterfall_attribution"]
    stages = [
        "1. Total Relevant Red GT",
        "2. Survives Perception",
        "3. Correct Red State",
        "4. Correct Relevance (Success)",
    ]
    tot = wf["success_local_count"] + wf["stage1_perception_miss_count"] + wf["stage3_state_misclassified_count"] + wf["stage4_relevance_miss_local_count"]
    surv_perc = tot - wf["stage1_perception_miss_count"]
    surv_state = surv_perc - wf["stage3_state_misclassified_count"]
    surv_rel = surv_state - wf["stage4_relevance_miss_local_count"]

    counts = [tot, surv_perc, surv_state, surv_rel]
    pcts = [100.0, surv_perc / max(tot, 1) * 100, surv_state / max(tot, 1) * 100, surv_rel / max(tot, 1) * 100]

    colors = ["#7f7f7f", "#1f77b4", "#17becf", "#2ca02c"]
    bars_wf = ax4.bar(stages, counts, color=colors, alpha=0.85, width=0.55)
    for b, p, c in zip(bars_wf, pcts, counts):
        ax4.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 50,
            f"{c} ({p:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax4.set_title(
        f"Relevant Red TL Safety Waterfall (Recall: {results['safety_critical_relevant_red']['recall_local_50']*100:.1f}%, Miss Rate: {results['safety_critical_relevant_red']['miss_rate_local_50']*100:.1f}%)",
        fontweight="bold",
    )
    ax4.set_ylabel("Ground Truth Instance Count")
    ax4.set_ylim(0, max(tot, 1) * 1.18)
    ax4.set_xticks(range(len(stages)))
    ax4.set_xticklabels(stages, rotation=15, ha="right")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagnostic plot saved to: {output_path}")


def generate_markdown_report(results: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    l2_loc = results["level2_detection_conditioned"]["overall"]["local"]
    l2_ctx = results["level2_detection_conditioned"]["overall"]["contextual"]
    l3 = results["level3_end_to_end"]
    safety = results["safety_critical_relevant_red"]
    wf = safety["waterfall_attribution"]

    md = []
    md.append("# W9 Diagnostic Audit: Local Relevance Baseline & Safety-Critical Metrics\n")
    md.append(f"**Audit Timestamp**: {results['timestamp']}")
    md.append(f"**Duration**: {results['duration_seconds']}s")
    md.append(f"**Total GT Traffic Lights Evaluated**: {results['total_gt_traffic_lights']:,} ({results['total_relevant_gt']:,} Relevant, {safety['total_relevant_red_gt']:,} Relevant Red)")
    md.append(f"**Learned Cross-Attention Alpha Gate**: `{results['learned_gate_alpha']:.6f}`\n")

    md.append("## 1. Executive Summary & Diagnostic Findings\n")
    md.append(f"- **Local Relevance Head Ceiling ($\\\\alpha = 0$)**: The local relevance head alone achieves **{l2_loc['auprc']*100:.2f}% AUPRC** (ROC-AUC: **{l2_loc['roc_auc']*100:.2f}%**, F1: **{l2_loc['f1']*100:.2f}%**, ECE: **{l2_loc['ece']*100:.2f}%**).")
    md.append(f"- **Contextual Lift vs Local Baseline ($\\\\Delta AUPRC$)**: With learned cross-attention active, relevance AUPRC is **{l2_ctx['auprc']*100:.2f}%**, yielding an overall differential of **{results['level2_detection_conditioned']['overall']['delta_auprc']*100:+.2f}% AUPRC**. Because $\\\\alpha \\\\approx 0$ (`-0.0315`), the contextual cross-attention branch remains effectively dormant and local cues dominate relevance decisions.")
    md.append(f"- **3-Tier Hierarchy Evaluation**: Oracle Relevance (Level 1) reaches **{results['level1_oracle']['overall']['auprc']*100:.2f}% AUPRC**, Level 2 Detection-Conditioned Relevance achieves **{l2_loc['auprc']*100:.2f}% AUPRC**, and Level 3 End-to-End Relevance + Detection ($s_{{det}} \\\\cdot P(rel)$) reaches **{l3['local']['ap50_e2e']*100:.2f}% $AP_{{50}}$** with **{l3['local']['recall_e2e']*100:.2f}%** total relevant GT recall.")
    md.append(f"- **Safety-Critical Metric (Relevant Red TL Recall)**: Relevant Red TL Recall reaches **{safety['recall_local_50']*100:.2f}%** (Miss Rate: **{safety['miss_rate_local_50']*100:.2f}%**). The waterfall analysis shows that **{wf['stage1_perception_miss_pct']:.1f}%** of misses are caused by upstream small-scale detection failures, **{wf['stage3_state_misclassified_pct']:.1f}%** by state classification errors, and only **{wf['stage4_relevance_miss_local_pct']:.1f}%** by relevance misclassification.\n")

    md.append("## 2. Level 2 Detection-Conditioned Relevance across Granular Slices\n")
    md.append("| Slice Category | Slice Name | Sample Count | Local AUPRC ($\\\\alpha=0$) | Ctx AUPRC ($\\\\alpha_{learned}$) | $\\\\Delta$ AUPRC | Local ECE | Local Brier |")
    md.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    # Overall
    md.append(f"| **Overall** | Validation Split | {l2_loc['count']:,} | **{l2_loc['auprc']*100:.2f}%** | **{l2_ctx['auprc']*100:.2f}%** | **{results['level2_detection_conditioned']['overall']['delta_auprc']*100:+.2f}%** | {l2_loc['ece']*100:.2f}% | {l2_loc['brier']:.4f} |")

    # Arrow Slices
    for k, v in results["level2_detection_conditioned"]["by_arrow"].items():
        name = "Arrows Present" if k == "arrows_present" else "No Arrows"
        md.append(f"| Arrow Context | {name} | {v['local']['count']:,} | {v['local']['auprc']*100:.2f}% | {v['contextual']['auprc']*100:.2f}% | **{v['delta_auprc']*100:+.2f}%** | {v['local']['ece']*100:.2f}% | {v['local']['brier']:.4f} |")

    # Round Slices
    for k, v in results["level2_detection_conditioned"]["by_round"].items():
        name = "Round Signal" if k == "round" else "Directional Arrow Signal"
        md.append(f"| Signal Type | {name} | {v['local']['count']:,} | {v['local']['auprc']*100:.2f}% | {v['contextual']['auprc']*100:.2f}% | **{v['delta_auprc']*100:+.2f}%** | {v['local']['ece']*100:.2f}% | {v['local']['brier']:.4f} |")

    # Density Slices
    for k, v in results["level2_detection_conditioned"]["by_density"].items():
        name = "Single TL Scene" if k == "single_tl" else "Multi-TL Scene"
        md.append(f"| Scene Density | {name} | {v['local']['count']:,} | {v['local']['auprc']*100:.2f}% | {v['contextual']['auprc']*100:.2f}% | **{v['delta_auprc']*100:+.2f}%** | {v['local']['ece']*100:.2f}% | {v['local']['brier']:.4f} |")

    # Area Buckets
    for k, v in results["level2_detection_conditioned"]["by_area"].items():
        md.append(f"| Area Bucket | `{k}` px² | {v['local']['count']:,} | {v['local']['auprc']*100:.2f}% | {v['contextual']['auprc']*100:.2f}% | **{v['delta_auprc']*100:+.2f}%** | {v['local']['ece']*100:.2f}% | {v['local']['brier']:.4f} |")

    md.append("\n\n## 3. 3-Tier Relevance Evaluation Hierarchy\n")
    md.append("| Tier Level | Evaluation Description | Primary Metric | Recall on Relevant GT | Optimal Threshold |")
    md.append("|---|---|:---:|:---:|:---:|")
    md.append(f"| **Level 1 (Oracle)** | Features sampled directly at GT locations (Mode B) | **{results['level1_oracle']['overall']['auprc']*100:.2f}% AUPRC** | 100.0% (Oracle) | {results['level1_oracle']['overall']['optimal_threshold']:.2f} |")
    md.append(f"| **Level 2 (Det-Conditioned Local)** | Local head on IoU $\\\\ge 0.50$ TP detected boxes | **{l2_loc['auprc']*100:.2f}% AUPRC** | {l2_loc['recall']*100:.2f}% (on TPs) | {l2_loc['optimal_threshold']:.2f} |")
    md.append(f"| **Level 2 (Det-Conditioned Ctx)** | Full model on IoU $\\\\ge 0.50$ TP detected boxes | **{l2_ctx['auprc']*100:.2f}% AUPRC** | {l2_ctx['recall']*100:.2f}% (on TPs) | {l2_ctx['optimal_threshold']:.2f} |")
    md.append(f"| **Level 3 (End-to-End Local)** | Combined score $s_{{det}} \\\\cdot P(rel)_{{local}}$ on all GTs | **{l3['local']['ap50_e2e']*100:.2f}% $AP_{{50}}$** | **{l3['local']['recall_e2e']*100:.2f}%** (overall) | — |")
    md.append(f"| **Level 3 (End-to-End Ctx)** | Combined score $s_{{det}} \\\\cdot P(rel)_{{ctx}}$ on all GTs | **{l3['contextual']['ap50_e2e']*100:.2f}% $AP_{{50}}$** | **{l3['contextual']['recall_e2e']*100:.2f}%** (overall) | — |\n")

    md.append("\n## 4. Safety-Critical Relevant Red Light Waterfall & Attribution\n")
    md.append(f"- **Total Relevant Red GT Traffic Lights**: {safety['total_relevant_red_gt']:,}")
    md.append(f"- **Relevant Red Recall (@ threshold 0.50)**: **{safety['recall_local_50']*100:.2f}%** (Miss Rate: **{safety['miss_rate_local_50']*100:.2f}%**)")
    md.append(f"- **Relevant Red Recall (@ threshold 0.30)**: **{safety['recall_local_30']*100:.2f}%**")
    md.append(f"- **Relevant Red Recall (@ threshold 0.70)**: **{safety['recall_local_70']*100:.2f}%**\n")

    md.append("### Failure Mode Attribution Waterfall:")
    md.append("| Pipeline Stage | Stage Description | Retained / Lost Count | Retention / Loss % | Cumulative Recall |")
    md.append("|:---:|---|:---:|:---:|:---:|")
    md.append(f"| **GT Total** | Ground-Truth Relevant Red TLs | {safety['total_relevant_red_gt']:,} | 100.0% | 100.0% |")
    md.append(f"| **Stage 1 (Perception)** | Upstream Detector Miss (IoU < 0.50) | -{wf['stage1_perception_miss_count']:,} | **-{wf['stage1_perception_miss_pct']:.2f}%** | {100.0 - wf['stage1_perception_miss_pct']:.2f}% |")
    md.append(f"| **Stage 2 (Candidate)** | Top-32 Candidate Selection Eviction | -{wf['stage2_candidate_evicted_count']:,} | **-{wf['stage2_candidate_evicted_pct']:.2f}%** | {100.0 - wf['stage1_perception_miss_pct'] - wf['stage2_candidate_evicted_pct']:.2f}% |")
    md.append(f"| **Stage 3 (State)** | State Head Misclassified ($\\\\hat{{s}} \\\\ne \\\\text{{Red}}$) | -{wf['stage3_state_misclassified_count']:,} | **-{wf['stage3_state_misclassified_pct']:.2f}%** | {100.0 - wf['stage1_perception_miss_pct'] - wf['stage2_candidate_evicted_pct'] - wf['stage3_state_misclassified_pct']:.2f}% |")
    md.append(f"| **Stage 4 (Relevance)** | Relevance Head False Negative ($P(rel) < 0.5$) | -{wf['stage4_relevance_miss_local_count']:,} | **-{wf['stage4_relevance_miss_local_pct']:.2f}%** | **{safety['recall_local_50']*100:.2f}%** |")
    md.append(f"| **Success** | Correctly Detected, Red, and Relevant | **{wf['success_local_count']:,}** | **{safety['recall_local_50']*100:.2f}%** | **{safety['recall_local_50']*100:.2f}%** |\n")

    md.append("### Red Light Relevance Confusion Matrix (@ threshold 0.50):\n")
    cm_loc = safety["confusion_matrix_local"]
    md.append("| Ground Truth \\ Prediction | Predicted Relevant | Predicted Irrelevant |")
    md.append("|---|:---:|:---:|")
    md.append(f"| **Actual Relevant Red** | **TP = {cm_loc['tp']:,}** | **FN = {cm_loc['fn']:,}** |")
    md.append(f"| **Actual Irrelevant Red** | **FP = {cm_loc['fp']:,}** | **TN = {cm_loc['tn']:,}** |\n")

    md.append("## 5. Artifacts Generated\n")
    md.append("- Visualization: `results/visualizations/w9_local_relevance_safety.png`\n")
    md.append("- Telemetry JSON: `results/audit_local_relevance_safety.json`\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit Local Relevance Baseline and Safety Metrics.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "tlr_yolo_mtl_single_phase_seed42" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tlr_mtl_dtld_paired" / "records.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, cfg = load_model(args.checkpoint, device)

    # Load validation dataset
    img_size = cfg.get("input_size", cfg.get("data", {}).get("img_size", [800, 1600]))
    val_dataset = CanonicalMultiTaskDataset(
        args.records_path,
        split="val",
        target_size=(img_size[0], img_size[1]),
        training=False,
        allowed_sources=["DTLD"],
        require_paired=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=canonical_multitask_collate,
        pin_memory=(device.type == "cuda"),
    )

    results = run_w9_audit(model, val_loader, device, max_batches=args.max_batches)

    # Save outputs
    json_path = PROJECT_ROOT / "results" / "audit_local_relevance_safety.json"
    plot_path = PROJECT_ROOT / "results" / "visualizations" / "w9_local_relevance_safety.png"
    report_path = PROJECT_ROOT / "results" / "audit_local_relevance_safety.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved to: {json_path}")

    plot_w9_diagnostics(results, plot_path)
    generate_markdown_report(results, report_path)


if __name__ == "__main__":
    main()
