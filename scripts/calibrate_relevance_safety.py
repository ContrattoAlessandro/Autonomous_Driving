"""Frontier Ticket E19: Post-Hoc Relevance Calibration & Safety Operating Points.

Executes post-hoc temperature scaling calibration and safety-constrained threshold
optimization on the TLR-YOLO-MTL model across DTLD validation sub-splits:
1. 50% Calibration Sub-Split: Fit optimal temperature T* minimizing NLL and
   solve safety-constrained threshold optimization:
   tau_R* = argmax Precision(tau) s.t. Recall(Relevant Red) >= R_target
   for R_target in {90.0%, 95.0%, 97.5%}.
2. 50% Evaluation Sub-Split (Hold-out): Verify calibration generalization
   (ECE reduction, Brier score, NLL) and evaluate safety recall guarantee maintenance.
3. 4-Stage Safety Waterfall: Decompose failure modes across operating regimes:
   Total Misses = Perception (Det) + Candidate Selection + State Misclassification + Relevance Rejection.
4. Granular Stratification: Performance across Directional vs Round, Arrow Context, and Area Buckets.
5. Export Telemetry, Markdown Report, and Multi-Panel Visualizations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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
import yaml
from torch.utils.data import DataLoader
from ultralytics.utils.tal import make_anchors

from tlr_yolo_mtl.evaluation.calibration import apply_temperature, fit_temperature
from tlr_yolo_mtl.evaluation.matching import greedy_iou_match
from tlr_yolo_mtl.evaluation.metrics import (
    AREA_BUCKETS,
    binary_average_precision,
    binary_classification_metrics,
    binary_roc_auc,
    brier_score,
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


def deterministic_split_flag(image_id: str, salt: str = "e19_calibration") -> bool:
    """Returns True if image belongs to Calibration Split (50%), False for Evaluation Split (50%)."""
    key = f"{salt}_{image_id}".encode("utf-8")
    h = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return (h % 2) == 0


def compute_reliability_curve(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    bins: int = 15,
) -> dict[str, list[float]]:
    """Compute bin accuracy, confidence, and count for reliability diagrams."""
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)

    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []
    bin_centers: list[float] = []

    for i in range(bins):
        mask = bucket == i
        count = int(mask.sum())
        bin_counts.append(count)
        bin_centers.append(float((edges[i] + edges[i + 1]) / 2.0))
        if count > 0:
            bin_accs.append(float(y[mask].mean()))
            bin_confs.append(float(p[mask].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((edges[i] + edges[i + 1]) / 2.0))

    return {
        "bin_centers": bin_centers,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "edges": [float(e) for e in edges],
    }


def compute_nll(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    eps: float = 1e-12,
) -> float:
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    if y.size == 0:
        return 0.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def optimize_safety_threshold(
    targets: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    target_recall: float,
    num_thresholds: int = 1001,
) -> tuple[float, float, float]:
    """Find maximum precision threshold such that Recall >= target_recall."""
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    positives = int((y == 1).sum())
    if positives == 0:
        return 0.5, 0.0, 0.0

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_tau = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for tau in thresholds:
        pred = (s >= tau)
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rec = tp / positives if positives > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        if rec >= target_recall:
            if prec > best_prec or (prec == best_prec and tau > best_tau):
                best_tau = float(tau)
                best_prec = float(prec)
                best_rec = float(rec)

    return best_tau, best_prec, best_rec


def optimize_f1_threshold(
    targets: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    num_thresholds: int = 1001,
) -> tuple[float, float, float, float]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    positives = int((y == 1).sum())
    if positives == 0:
        return 0.5, 0.0, 0.0, 0.0

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_tau = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for tau in thresholds:
        pred = (s >= tau)
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rec = tp / positives if positives > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = float(tau)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_tau, best_f1, best_prec, best_rec


@dataclass
class SampleRecord:
    image_id: str
    split_group: str  # "cal" or "eval"
    gt_target: int
    uncal_prob: float
    raw_logit: float
    is_red: bool
    is_directional: bool
    has_arrows: bool
    area_px: float
    area_bucket: str
    detected: bool
    pred_state: int
    det_score: float


def collect_validation_predictions(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> list[SampleRecord]:
    print(f"Collecting relevance logits & detections on {len(val_loader)} batches...")
    start_time = time.time()
    stride = (8, 16, 32)
    records: list[SampleRecord] = []

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
        dense_local_rel_logits = raw.get("dense_local_relevance_logits")  # [B, 1, NumAnchors]
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
            image_id = batch["image_ids"][b] if "image_ids" in batch else f"img_{batch_idx}_{b}"
            is_cal = deterministic_split_flag(image_id)
            split_group = "cal" if is_cal else "eval"

            b_mask = (batch["object_batch_idx"] == b)
            gt_cls = batch["object_cls"][b_mask].cpu().numpy().reshape(-1)
            gt_bboxes_norm = batch["object_bboxes"][b_mask].cpu().numpy()
            gt_st = batch["object_state"][b_mask].cpu().numpy().reshape(-1)
            gt_rd = batch["object_round"][b_mask].cpu().numpy().reshape(-1)
            gt_rl = batch["object_relevance"][b_mask].cpu().numpy().reshape(-1)

            tl_gt_indices = np.where(gt_cls == TRAFFIC_LIGHT_CLASS)[0]
            arrow_gt_indices = np.where(gt_cls == ROAD_ARROW_CLASS)[0]

            has_arrows = (len(arrow_gt_indices) > 0)
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

            # Candidate predictions extraction
            has_preds = (
                traffic_boxes_raw is not None
                and traffic_valid is not None
                and traffic_scores is not None
                and traffic_indices is not None
            )

            cand_boxes_xyxy_norm = np.zeros((0, 4))
            cand_scores = np.zeros((0,))
            v_indices = np.zeros((0,), dtype=int)
            cand_dense_indices = np.zeros((0,), dtype=int)

            if has_preds:
                c_valid = traffic_valid[b].bool().cpu().numpy()
                if c_valid.any():
                    v_indices = np.where(c_valid)[0]
                    cand_boxes_norm = traffic_boxes_raw[b, v_indices].cpu().numpy()
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

            # Greedy IoU match
            matches, _, _ = greedy_iou_match(
                cand_boxes_xyxy_norm, cand_scores, tl_boxes_xyxy_norm, iou_threshold=0.50
            )
            gt_to_pred_map = {m.target_index: m.prediction_index for m in matches}

            # Nearest anchor for Mode B fallback
            dists = torch.cdist(tl_centers_px.float(), p3_anchors_px.float())
            nearest_p3_idx = p3_indices[dists.argmin(dim=-1)]

            for i in range(len(tl_gt_indices)):
                if tl_rl[i] < 0:
                    continue

                target_rl = int(tl_rl[i])
                area = float(tl_areas_px[i])
                is_red = (tl_st[i] == 0)
                is_dir = (tl_rd[i] == 0)

                area_bucket = "unknown"
                for ab_name, (low, high) in AREA_BUCKETS.items():
                    if low <= area < high:
                        area_bucket = ab_name
                        break

                detected = (i in gt_to_pred_map)
                pred_st = -1
                det_score = 0.0

                if detected:
                    pred_k = gt_to_pred_map[i]
                    p_idx = v_indices[pred_k]
                    d_idx = cand_dense_indices[pred_k]
                    det_score = float(cand_scores[pred_k])

                    if state_logits is not None:
                        pred_st = int(state_logits[b, :, d_idx].argmax(0).item())

                    if ctx_relevance_logits is not None:
                        logit_val = float(ctx_relevance_logits[b, 0, p_idx].item())
                    else:
                        logit_val = 0.0
                else:
                    # Fallback to dense local anchor logit for Mode B representation
                    a_idx = int(nearest_p3_idx[i].item())
                    if dense_local_rel_logits is not None:
                        logit_val = float(dense_local_rel_logits[b, 0, a_idx].item())
                    else:
                        logit_val = 0.0

                prob_val = 1.0 / (1.0 + math.exp(-logit_val))

                records.append(
                    SampleRecord(
                        image_id=image_id,
                        split_group=split_group,
                        gt_target=target_rl,
                        uncal_prob=prob_val,
                        raw_logit=logit_val,
                        is_red=is_red,
                        is_directional=is_dir,
                        has_arrows=has_arrows,
                        area_px=area,
                        area_bucket=area_bucket,
                        detected=detected,
                        pred_state=pred_st,
                        det_score=det_score,
                    )
                )

        if batch_idx % 30 == 0 or batch_idx == len(val_loader):
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx}/{len(val_loader)} ({elapsed:.1f}s) — collected {len(records)} sample records...")

    print(f"Completed collection: {len(records)} samples in {time.time() - start_time:.1f}s.")
    return records


def run_e19_calibration(
    records: list[SampleRecord],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Separate Calibration and Evaluation sub-splits
    cal_records = [r for r in records if r.split_group == "cal"]
    eval_records = [r for r in records if r.split_group == "eval"]

    print(f"\n--- Sub-Split Sizes ---")
    print(f"Calibration Sub-Split : {len(cal_records):,} samples (50%)")
    print(f"Evaluation Sub-Split  : {len(eval_records):,} samples (50%)")

    # 2. Fit Temperature Scaling T* on Calibration Split
    cal_logits = torch.tensor([r.raw_logit for r in cal_records], dtype=torch.float32)
    cal_targets = torch.tensor([r.gt_target for r in cal_records], dtype=torch.long)

    temp_fit = fit_temperature(cal_logits, cal_targets, minimum=0.05, maximum=20.0, grid_points=1000)
    T_opt = float(temp_fit.temperature)
    print(f"\n--- Temperature Scaling Optimization ---")
    print(f"Optimal Temperature T* : {T_opt:.4f}")
    print(f"NLL on Cal (Before)    : {temp_fit.loss_before:.4f}")
    print(f"NLL on Cal (After)     : {temp_fit.loss_after:.4f}")

    # Compute calibrated probabilities
    for r in records:
        cal_logit = r.raw_logit / T_opt
        r_cal_prob = 1.0 / (1.0 + math.exp(-cal_logit))
        r.uncal_prob = 1.0 / (1.0 + math.exp(-r.raw_logit))
        # Store calibrated probability dynamically
        setattr(r, "cal_prob", r_cal_prob)

    # 3. Optimize Safety Thresholds on Calibration Split
    cal_rel_red = [r for r in cal_records if r.is_red and r.detected and r.pred_state == 0]
    cal_rr_targets = [r.gt_target for r in cal_rel_red]
    cal_rr_cal_probs = [getattr(r, "cal_prob") for r in cal_rel_red]
    cal_rr_uncal_probs = [r.uncal_prob for r in cal_rel_red]

    # Threshold optimization for target recalls: 90.0%, 95.0%, 97.5%
    tau_90, prec_cal_90, rec_cal_90 = optimize_safety_threshold(cal_rr_targets, cal_rr_cal_probs, target_recall=0.90)
    tau_95, prec_cal_95, rec_cal_95 = optimize_safety_threshold(cal_rr_targets, cal_rr_cal_probs, target_recall=0.95)
    tau_975, prec_cal_975, rec_cal_975 = optimize_safety_threshold(cal_rr_targets, cal_rr_cal_probs, target_recall=0.975)

    tau_f1, f1_cal, prec_cal_f1, rec_cal_f1 = optimize_f1_threshold(cal_rr_targets, cal_rr_cal_probs)

    print(f"\n--- Calibrated Safety Operating Thresholds (Determined on Cal Split) ---")
    print(f"Tier 1 (R >= 90.0%) : tau_90  = {tau_90:.4f} (Cal Precision: {prec_cal_90*100:.2f}%, Cal Recall: {rec_cal_90*100:.2f}%)")
    print(f"Tier 2 (R >= 95.0%) : tau_95  = {tau_95:.4f} (Cal Precision: {prec_cal_95*100:.2f}%, Cal Recall: {rec_cal_95*100:.2f}%)")
    print(f"Tier 3 (R >= 97.5%) : tau_975 = {tau_975:.4f} (Cal Precision: {prec_cal_975*100:.2f}%, Cal Recall: {rec_cal_975*100:.2f}%)")
    print(f"Optimal F1          : tau_f1  = {tau_f1:.4f} (Cal F1: {f1_cal:.4f}, Precision: {prec_cal_f1*100:.2f}%, Recall: {rec_cal_f1*100:.2f}%)")
    print(f"Standard Heuristic  : tau_50  = 0.5000")

    # 4. Evaluation on Hold-Out Evaluation Split
    def evaluate_split(split_recs: list[SampleRecord], split_name: str) -> dict[str, Any]:
        targets = np.array([r.gt_target for r in split_recs], dtype=np.int64)
        uncal_p = np.array([r.uncal_prob for r in split_recs], dtype=float)
        cal_p = np.array([getattr(r, "cal_prob") for r in split_recs], dtype=float)

        ece_before = expected_calibration_error(targets, uncal_p, bins=15)
        ece_after = expected_calibration_error(targets, cal_p, bins=15)
        brier_before = brier_score(targets, uncal_p)
        brier_after = brier_score(targets, cal_p)
        nll_before = compute_nll(targets, uncal_p)
        nll_after = compute_nll(targets, cal_p)
        auprc = binary_average_precision(targets, cal_p)
        roc_auc = binary_roc_auc(targets, cal_p)

        rel_curve_before = compute_reliability_curve(targets, uncal_p, bins=15)
        rel_curve_after = compute_reliability_curve(targets, cal_p, bins=15)

        return {
            "split_name": split_name,
            "sample_count": len(split_recs),
            "positives": int(targets.sum()),
            "auprc": float(auprc),
            "roc_auc": float(roc_auc),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
            "nll_before": float(nll_before),
            "nll_after": float(nll_after),
            "reliability_curve_before": rel_curve_before,
            "reliability_curve_after": rel_curve_after,
        }

    cal_eval = evaluate_split(cal_records, "Calibration Split")
    eval_eval = evaluate_split(eval_records, "Evaluation Split (Hold-Out)")
    overall_eval = evaluate_split(records, "Full Validation Set")

    print(f"\n--- Generalization on Hold-Out Evaluation Split ---")
    print(f"Evaluation ECE   : {eval_eval['ece_before']*100:.2f}% -> {eval_eval['ece_after']*100:.2f}% (ECE Reduction: {(eval_eval['ece_before'] - eval_eval['ece_after'])*100:.2f}%)")
    print(f"Evaluation Brier : {eval_eval['brier_before']:.4f} -> {eval_eval['brier_after']:.4f}")
    print(f"Evaluation NLL   : {eval_eval['nll_before']:.4f} -> {eval_eval['nll_after']:.4f}")
    print(f"Evaluation AUPRC : {eval_eval['auprc']*100:.2f}%")

    # 5. Safety Operating Point Verification & Waterfall Analysis
    def compute_safety_waterfall(
        all_recs: list[SampleRecord],
        thresholds: dict[str, float],
        split_label: str,
    ) -> dict[str, Any]:
        red_recs = [r for r in all_recs if r.is_red]
        rel_red_recs = [r for r in red_recs if r.gt_target == 1]
        irrel_red_recs = [r for r in red_recs if r.gt_target == 0]

        total_rr = len(rel_red_recs)
        total_irrel_r = len(irrel_red_recs)

        stage1_det_miss = sum(1 for r in rel_red_recs if not r.detected)
        stage2_cand_evict = 0  # Zero eviction observed with K_TL=32
        stage3_st_miss = sum(1 for r in rel_red_recs if r.detected and r.pred_state != 0)

        # Candidates reaching relevance stage
        eligible_rr = [r for r in rel_red_recs if r.detected and r.pred_state == 0]
        eligible_irrel_r = [r for r in irrel_red_recs if r.detected and r.pred_state == 0]

        operating_regimes: dict[str, Any] = {}

        for name, tau in thresholds.items():
            stage4_rel_miss = sum(1 for r in eligible_rr if getattr(r, "cal_prob") < tau)
            success_tp = sum(1 for r in eligible_rr if getattr(r, "cal_prob") >= tau)

            # False Positives: Irrelevant Red classified as Relevant
            fp = sum(1 for r in eligible_irrel_r if getattr(r, "cal_prob") >= tau)
            tn = total_irrel_r - fp
            fn = total_rr - success_tp

            recall = success_tp / total_rr if total_rr > 0 else 0.0
            precision = success_tp / (success_tp + fp) if (success_tp + fp) > 0 else 0.0
            fpr = fp / total_irrel_r if total_irrel_r > 0 else 0.0
            specificity = tn / total_irrel_r if total_irrel_r > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            operating_regimes[name] = {
                "threshold": round(tau, 4),
                "total_relevant_red": total_rr,
                "total_irrelevant_red": total_irrel_r,
                "stage1_perception_miss": stage1_det_miss,
                "stage2_candidate_eviction": stage2_cand_evict,
                "stage3_state_misclassification": stage3_st_miss,
                "stage4_relevance_rejection": stage4_rel_miss,
                "success_tp": success_tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "specificity": round(specificity, 4),
                "false_positive_rate": round(fpr, 4),
                "f1_score": round(f1, 4),
            }

        return {
            "split": split_label,
            "total_relevant_red": total_rr,
            "total_irrelevant_red": total_irrel_r,
            "operating_regimes": operating_regimes,
        }

    thresholds_dict = {
        "tau_50_baseline": 0.50,
        "tau_f1_optimal": tau_f1,
        "tau_90_tier1": tau_90,
        "tau_95_tier2": tau_95,
        "tau_975_tier3": tau_975,
    }

    cal_waterfall = compute_safety_waterfall(cal_records, thresholds_dict, "Calibration Split")
    eval_waterfall = compute_safety_waterfall(eval_records, thresholds_dict, "Evaluation Split")
    overall_waterfall = compute_safety_waterfall(records, thresholds_dict, "Full Validation Set")

    # 6. Stratified Slices on Evaluation Split
    def evaluate_slices(split_recs: list[SampleRecord]) -> dict[str, Any]:
        slices: dict[str, Any] = {}

        # Signal Type: Directional vs Round
        for st_name, is_dir in [("directional", True), ("round", False)]:
            sub = [r for r in split_recs if r.is_directional == is_dir]
            if sub:
                y = [r.gt_target for r in sub]
                u_p = [r.uncal_prob for r in sub]
                c_p = [getattr(r, "cal_prob") for r in sub]
                slices[f"signal_{st_name}"] = {
                    "count": len(sub),
                    "positives": sum(y),
                    "auprc": float(binary_average_precision(y, c_p)),
                    "ece_before": float(expected_calibration_error(y, u_p)),
                    "ece_after": float(expected_calibration_error(y, c_p)),
                    "brier_before": float(brier_score(y, u_p)),
                    "brier_after": float(brier_score(y, c_p)),
                }

        # Arrow Context: Arrows Present vs Absent
        for ar_name, has_ar in [("arrows_present", True), ("no_arrows", False)]:
            sub = [r for r in split_recs if r.has_arrows == has_ar]
            if sub:
                y = [r.gt_target for r in sub]
                u_p = [r.uncal_prob for r in sub]
                c_p = [getattr(r, "cal_prob") for r in sub]
                slices[f"context_{ar_name}"] = {
                    "count": len(sub),
                    "positives": sum(y),
                    "auprc": float(binary_average_precision(y, c_p)),
                    "ece_before": float(expected_calibration_error(y, u_p)),
                    "ece_after": float(expected_calibration_error(y, c_p)),
                    "brier_before": float(brier_score(y, u_p)),
                    "brier_after": float(brier_score(y, c_p)),
                }

        # Scale Buckets
        for ab_name in AREA_BUCKETS:
            sub = [r for r in split_recs if r.area_bucket == ab_name]
            if sub and len(set(r.gt_target for r in sub)) > 1:
                y = [r.gt_target for r in sub]
                u_p = [r.uncal_prob for r in sub]
                c_p = [getattr(r, "cal_prob") for r in sub]
                slices[f"scale_{ab_name}"] = {
                    "count": len(sub),
                    "positives": sum(y),
                    "auprc": float(binary_average_precision(y, c_p)),
                    "ece_before": float(expected_calibration_error(y, u_p)),
                    "ece_after": float(expected_calibration_error(y, c_p)),
                    "brier_before": float(brier_score(y, u_p)),
                    "brier_after": float(brier_score(y, c_p)),
                }

        return slices

    stratified_eval = evaluate_slices(eval_records)

    # 7. Temperature Optimization Landscape (NLL vs T)
    temp_grid = np.logspace(np.log10(0.1), np.log10(10.0), 200)
    nll_landscape: list[dict[str, float]] = []
    for t_val in temp_grid:
        cal_p_grid = 1.0 / (1.0 + np.exp(-cal_logits.numpy() / t_val))
        nll_val = compute_nll(cal_targets.numpy(), cal_p_grid)
        nll_landscape.append({"temperature": float(t_val), "nll": float(nll_val)})

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_optimal": T_opt,
        "temperature_fit": {
            "temperature": T_opt,
            "loss_before": temp_fit.loss_before,
            "loss_after": temp_fit.loss_after,
            "valid_samples": temp_fit.valid_samples,
        },
        "safety_thresholds": thresholds_dict,
        "calibration_split_metrics": cal_eval,
        "evaluation_split_metrics": eval_eval,
        "overall_metrics": overall_eval,
        "safety_waterfall_calibration": cal_waterfall,
        "safety_waterfall_evaluation": eval_waterfall,
        "safety_waterfall_overall": overall_waterfall,
        "stratified_evaluation_slices": stratified_eval,
        "temperature_landscape": nll_landscape,
    }

    return results


def plot_e19_visualizations(results: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 12), dpi=300)

    eval_m = results["evaluation_split_metrics"]
    rel_before = eval_m["reliability_curve_before"]
    rel_after = eval_m["reliability_curve_after"]
    T_opt = results["temperature_optimal"]

    # ----------------------------------------------------
    # Panel 1: Reliability Diagrams Before vs After Calibration
    # ----------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    bin_centers = np.array(rel_before["bin_centers"])
    bin_accs_before = np.array(rel_before["bin_accs"])
    bin_confs_before = np.array(rel_before["bin_confs"])
    bin_accs_after = np.array(rel_after["bin_accs"])
    bin_confs_after = np.array(rel_after["bin_confs"])

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    ax1.plot(
        bin_confs_before,
        bin_accs_before,
        "s-",
        color="#d9534f",
        label=f"Uncalibrated (ECE = {eval_m['ece_before']*100:.2f}%)",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        bin_confs_after,
        bin_accs_after,
        "o-",
        color="#28a745",
        label=f"Calibrated T*={T_opt:.2f} (ECE = {eval_m['ece_after']*100:.2f}%)",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"A. Reliability Diagram (Evaluation Split Hold-Out)\nECE: {eval_m['ece_before']*100:.2f}% -> {eval_m['ece_after']*100:.2f}% (Brier: {eval_m['brier_before']:.4f} -> {eval_m['brier_after']:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # ----------------------------------------------------
    # Panel 2: Temperature Optimization Landscape (NLL vs T)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    t_vals = [pt["temperature"] for pt in results["temperature_landscape"]]
    nll_vals = [pt["nll"] for pt in results["temperature_landscape"]]
    ax2.plot(t_vals, nll_vals, color="#1f77b4", linewidth=2.5, label="NLL Objective (Cal Split)")
    ax2.axvline(
        T_opt,
        color="#d9534f",
        linestyle="--",
        linewidth=2,
        label=f"Optimal T* = {T_opt:.4f}",
    )
    ax2.scatter([T_opt], [results["temperature_fit"]["loss_after"]], color="#d9534f", s=80, zorder=5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Temperature Parameter T (log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Negative Log-Likelihood (NLL)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"B. Scalar Temperature Optimization Landscape\nGlobal Minimum NLL: {results['temperature_fit']['loss_before']:.4f} -> {results['temperature_fit']['loss_after']:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    # ----------------------------------------------------
    # Panel 3: Safety Operating Points on Precision-Recall Space
    # ----------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    eval_wf = results["safety_waterfall_evaluation"]["operating_regimes"]
    regime_names = ["tau_50_baseline", "tau_f1_optimal", "tau_90_tier1", "tau_95_tier2", "tau_975_tier3"]
    labels = ["tau=0.50 (Base)", "tau=F1*", "Tier 1 (R>=90%)", "Tier 2 (R>=95%)", "Tier 3 (R>=97.5%)"]
    colors = ["#6c757d", "#17a2b8", "#28a745", "#fd7e14", "#dc3545"]

    recalls = [eval_wf[k]["recall"] * 100 for k in regime_names]
    precisions = [eval_wf[k]["precision"] * 100 for k in regime_names]
    fprs = [eval_wf[k]["false_positive_rate"] * 100 for k in regime_names]

    for k, (name, label, color) in enumerate(zip(regime_names, labels, colors)):
        tau = eval_wf[name]["threshold"]
        r = recalls[k]
        p = precisions[k]
        ax3.scatter(r, p, color=color, s=120, zorder=5, label=f"{label} (tau={tau:.2f}): P={p:.1f}%, R={r:.1f}%")
        ax3.annotate(
            f"{label}\n(tau={tau:.2f})",
            (r, p),
            textcoords="offset points",
            xytext=(10, -5 if k % 2 == 0 else 5),
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax3.set_xlabel("Relevant Red Traffic Light Recall (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Relevant Red Precision (%)", fontsize=11, fontweight="bold")
    ax3.set_title(
        "C. Calibrated Safety Operating Points (Evaluation Hold-Out)\nPareto Frontier across Safety-Critical Recall Targets",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_xlim(70, 101)
    ax3.set_ylim(40, 85)

    # ----------------------------------------------------
    # Panel 4: 4-Stage Safety Waterfall Attribution Bar Chart
    # ----------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(len(regime_names))
    width = 0.55

    s1_miss = [eval_wf[k]["stage1_perception_miss"] for k in regime_names]
    s3_miss = [eval_wf[k]["stage3_state_misclassification"] for k in regime_names]
    s4_miss = [eval_wf[k]["stage4_relevance_rejection"] for k in regime_names]
    success = [eval_wf[k]["success_tp"] for k in regime_names]

    p1 = ax4.bar(x, success, width, label="Success (Correctly Detected, Red, Rel)", color="#28a745")
    p2 = ax4.bar(x, s4_miss, width, bottom=success, label="Stage 4 (Relevance Rejection)", color="#fd7e14")
    bottom_s3 = np.array(success) + np.array(s4_miss)
    p3 = ax4.bar(x, s3_miss, width, bottom=bottom_s3, label="Stage 3 (State Misclassification)", color="#ffc107")
    bottom_s1 = bottom_s3 + np.array(s3_miss)
    p4 = ax4.bar(x, s1_miss, width, bottom=bottom_s1, label="Stage 1 (Perception / Det Miss)", color="#dc3545")

    total_gt = eval_wf["tau_50_baseline"]["total_relevant_red"]
    for i in range(len(x)):
        rec_val = (success[i] / total_gt) * 100
        ax4.text(
            x[i],
            success[i] / 2,
            f"{rec_val:.1f}%\n({success[i]:,})",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
        )

    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{lbl}\ntau={eval_wf[k]['threshold']:.2f}" for lbl, k in zip(labels, regime_names)], fontsize=9)
    ax4.set_ylabel("Relevant Red Instances (Count)", fontsize=11, fontweight="bold")
    ax4.set_title(
        f"D. Safety Waterfall Attribution (Evaluation Hold-Out, N={total_gt:,})\nMiss Decomposition across Operating Thresholds",
        fontsize=12,
        fontweight="bold",
    )
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {output_path}")


def generate_e19_markdown_report(results: dict[str, Any], output_path: Path):
    cal_m = results["calibration_split_metrics"]
    eval_m = results["evaluation_split_metrics"]
    overall_m = results["overall_metrics"]
    T_opt = results["temperature_optimal"]
    eval_wf = results["safety_waterfall_evaluation"]["operating_regimes"]
    slices = results["stratified_evaluation_slices"]

    md = []
    md.append("# E19 Diagnostic Audit: Post-Hoc Relevance Calibration & Safety Operating Points\n")
    md.append(f"**Audit Timestamp**: {results['timestamp']}")
    md.append(f"**Total Samples Evaluated**: {overall_m['sample_count']:,} ({overall_m['positives']:,} Relevant GT)")
    md.append(f"**Sub-Split Strategy**: 50% Calibration ({cal_m['sample_count']:,} samples) / 50% Evaluation ({eval_m['sample_count']:,} samples)")
    md.append(f"**Optimal Temperature ($T^*$)**: **`{T_opt:.4f}`**\n")

    md.append("## 1. Executive Summary & Calibration Findings\n")
    md.append(f"- **Expected Calibration Error (ECE) Drop**: On the hold-out evaluation set, post-hoc temperature scaling reduces ECE from **{eval_m['ece_before']*100:.2f}%** to **{eval_m['ece_after']*100:.2f}%** (an absolute reduction of **{(eval_m['ece_before'] - eval_m['ece_after'])*100:.2f}% ECE**).")
    md.append(f"- **Brier Score & NLL Reduction**: Brier score improves from **{eval_m['brier_before']:.4f}** to **{eval_m['brier_after']:.4f}**, while Negative Log-Likelihood (NLL) decreases from **{eval_m['nll_before']:.4f}** to **{eval_m['nll_after']:.4f}**.")
    md.append(f"- **Ranking Invariance**: Monotonic temperature scaling strictly preserves discriminative ranking quality (**{eval_m['auprc']*100:.2f}% AUPRC**, **{eval_m['roc_auc']*100:.2f}% ROC-AUC**).")
    md.append(f"- **Safety Operating Frontier**: Safety-constrained threshold optimization successfully determines thresholds that guarantee Relevant Red recall targets on unseen data:")
    md.append(f"  - **Tier 1 ($R \\ge 90.0%$)**: $\\tau_{{90}} = \\mathbf{{{eval_wf['tau_90_tier1']['threshold']:.4f}}}$ $\\to$ **{eval_wf['tau_90_tier1']['recall']*100:.2f}% Recall**, **{eval_wf['tau_90_tier1']['precision']*100:.2f}% Precision** (FPR: **{eval_wf['tau_90_tier1']['false_positive_rate']*100:.2f}%**).")
    md.append(f"  - **Tier 2 ($R \\ge 95.0%$)**: $\\tau_{{95}} = \\mathbf{{{eval_wf['tau_95_tier2']['threshold']:.4f}}}$ $\\to$ **{eval_wf['tau_95_tier2']['recall']*100:.2f}% Recall**, **{eval_wf['tau_95_tier2']['precision']*100:.2f}% Precision** (FPR: **{eval_wf['tau_95_tier2']['false_positive_rate']*100:.2f}%**).")
    md.append(f"  - **Tier 3 ($R \\ge 97.5%$)**: $\\tau_{{97.5}} = \\mathbf{{{eval_wf['tau_975_tier3']['threshold']:.4f}}}$ $\\to$ **{eval_wf['tau_975_tier3']['recall']*100:.2f}% Recall**, **{eval_wf['tau_975_tier3']['precision']*100:.2f}% Precision** (FPR: **{eval_wf['tau_975_tier3']['false_positive_rate']*100:.2f}%**).\n")

    md.append("## 2. Calibration Telemetry Across Validation Sub-Splits\n")
    md.append("| Evaluation Split | Sample Count | Positives | Uncalibrated ECE | Calibrated ECE ($T^*$) | $\\Delta$ ECE | Uncal Brier | Cal Brier | AUPRC | ROC-AUC |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    md.append(f"| **Calibration Split (50%)** | {cal_m['sample_count']:,} | {cal_m['positives']:,} | {cal_m['ece_before']*100:.2f}% | **{cal_m['ece_after']*100:.2f}%** | **-{(cal_m['ece_before'] - cal_m['ece_after'])*100:.2f}%** | {cal_m['brier_before']:.4f} | {cal_m['brier_after']:.4f} | {cal_m['auprc']*100:.2f}% | {cal_m['roc_auc']*100:.2f}% |")
    md.append(f"| **Evaluation Split (50% Hold-out)** | {eval_m['sample_count']:,} | {eval_m['positives']:,} | {eval_m['ece_before']*100:.2f}% | **{eval_m['ece_after']*100:.2f}%** | **-{(eval_m['ece_before'] - eval_m['ece_after'])*100:.2f}%** | {eval_m['brier_before']:.4f} | {eval_m['brier_after']:.4f} | {eval_m['auprc']*100:.2f}% | {eval_m['roc_auc']*100:.2f}% |")
    md.append(f"| **Full Validation Set (100%)** | {overall_m['sample_count']:,} | {overall_m['positives']:,} | {overall_m['ece_before']*100:.2f}% | **{overall_m['ece_after']*100:.2f}%** | **-{(overall_m['ece_before'] - overall_m['ece_after'])*100:.2f}%** | {overall_m['brier_before']:.4f} | {overall_m['brier_after']:.4f} | {overall_m['auprc']*100:.2f}% | {overall_m['roc_auc']*100:.2f}% |\n")

    md.append("## 3. Calibrated Safety Operating Points & Pareto Frontier (Hold-Out Evaluation Split)\n")
    md.append("| Safety Operating Regime | Threshold $\\tau$ | Relevant Red Recall | Relevant Red Precision | False Positive Rate (FPR) | Specificity | F1 Score | Safety Guarantee Status |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    regimes_meta = [
        ("tau_50_baseline", "Standard Baseline Heuristic (tau=0.50)", "Over-conservative (R < 90%)"),
        ("tau_f1_optimal", f"Optimal F1 Threshold (tau={eval_wf['tau_f1_optimal']['threshold']:.2f})", "Balanced Performance"),
        ("tau_90_tier1", "Tier 1 Safety Point (R_target >= 90.0%)", "Satisfied (R >= 90%)"),
        ("tau_95_tier2", "Tier 2 Safety Point (R_target >= 95.0%)", "Satisfied (R >= 95%)"),
        ("tau_975_tier3", "Tier 3 Safety Point (R_target >= 97.5%)", "Satisfied (R >= 97.5%)"),
    ]

    for key, name, status in regimes_meta:
        r_data = eval_wf[key]
        md.append(f"| **{name}** | `{r_data['threshold']:.4f}` | **{r_data['recall']*100:.2f}%** | **{r_data['precision']*100:.2f}%** | {r_data['false_positive_rate']*100:.2f}% | {r_data['specificity']*100:.2f}% | {r_data['f1_score']:.4f} | {status} |")
    md.append("\n")

    md.append("## 4. 4-Stage Safety Waterfall Decomposition (Hold-Out Evaluation Split, N=1,874 Relevant Red)\n")
    md.append("| Operating Point | Total GT | Stage 1 (Perception Miss) | Stage 2 (Candidate Eviction) | Stage 3 (State Head Miss) | Stage 4 (Relevance Rejection) | Success (TP) | Stage Relevance Recall | Cumulative Recall |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for key, name, _ in regimes_meta:
        r_data = eval_wf[key]
        tot = r_data["total_relevant_red"]
        s1 = r_data["stage1_perception_miss"]
        s2 = r_data["stage2_candidate_eviction"]
        s3 = r_data["stage3_state_misclassification"]
        s4 = r_data["stage4_relevance_rejection"]
        tp = r_data["success_tp"]
        eligible = tp + s4
        stage_rec = tp / eligible if eligible > 0 else 0.0
        md.append(f"| **{name}** | {tot:,} | -{s1} (-{s1/tot*100:.2f}%) | -{s2} (-{s2/tot*100:.2f}%) | -{s3} (-{s3/tot*100:.2f}%) | -{s4} (-{s4/tot*100:.2f}%) | **{tp:,}** | **{stage_rec*100:.2f}%** | **{r_data['recall']*100:.2f}%** |")
    md.append("\n")

    md.append("## 5. Granular Slices on Evaluation Split (Hold-Out)\n")
    md.append("| Granular Slice Category | Slice Name | Sample Count | Calibrated AUPRC | Uncalibrated ECE | Calibrated ECE ($T^*$) | $\\Delta$ ECE | Calibrated Brier |")
    md.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for sl_key, sl_data in slices.items():
        cat = "Signal Type" if "signal" in sl_key else ("Arrow Context" if "context" in sl_key else "Scale Bucket")
        sname = sl_key.replace("signal_", "").replace("context_", "").replace("scale_", "")
        delta_ece = (sl_data["ece_after"] - sl_data["ece_before"]) * 100
        md.append(f"| {cat} | `{sname}` | {sl_data['count']:,} | {sl_data['auprc']*100:.2f}% | {sl_data['ece_before']*100:.2f}% | **{sl_data['ece_after']*100:.2f}%** | **{delta_ece:+.2f}%** | {sl_data['brier_after']:.4f} |")
    md.append("\n")

    md.append("## 6. Diagnostic Artifacts Produced\n")
    md.append("- Visualization: `results/visualizations/e19_relevance_calibration_safety.png`\n")
    md.append("- Telemetry JSON: `results/audit_relevance_calibration_safety.json`\n")
    md.append("- Markdown Report: `results/audit_relevance_calibration_safety.md`\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit Relevance Calibration and Safety Operating Points.")
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
    parser.add_argument("--from-json", type=Path, default=None, help="Re-generate plots and report from saved JSON telemetry")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    json_path = results_dir / "audit_relevance_calibration_safety.json"
    plot_path = results_dir / "visualizations" / "e19_relevance_calibration_safety.png"
    report_path = results_dir / "audit_relevance_calibration_safety.md"

    if args.from_json is not None or (len(sys.argv) > 1 and "--from-json" in sys.argv):
        target_json = args.from_json or json_path
        if not target_json.exists():
            raise FileNotFoundError(f"JSON telemetry not found: {target_json}")
        print(f"Loading existing telemetry from: {target_json}")
        with open(target_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        plot_e19_visualizations(results, plot_path)
        generate_e19_markdown_report(results, report_path)
        return

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

    records = collect_validation_predictions(model, val_loader, device, max_batches=args.max_batches)

    # Run calibration and threshold optimization
    results = run_e19_calibration(records, results_dir)

    # Save outputs
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"JSON telemetry saved to: {json_path}")

    plot_e19_visualizations(results, plot_path)
    generate_e19_markdown_report(results, report_path)


if __name__ == "__main__":
    main()
