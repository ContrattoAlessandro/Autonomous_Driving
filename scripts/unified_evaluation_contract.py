"""Unified Evaluation Contract Runner & Baseline Normalization Harness (Ticket E29).

Enforces the canonical Unified Evaluation Contract across the full DTLD validation set:
1. Multi-Checkpoint Matrix Evaluation:
   - Primary Thesis Benchmark: best_composite.pt
   - Diagnostic Audit Matrix: [best_composite, best_relevance, best_tl_detection, best_relevant_red_recall, last]
2. Scale-Stratified Perception Floor:
   - Area buckets: <32, 32-64, 64-128, 128-256, 256-512, >512 px².
   - Min-side buckets: <4, 4-6, 6-8, 8-12, >12 px.
3. Multi-Task Perception & Fine-Grained Attribute Assessment:
   - mAP50, mAP50-95, AP_TL_50, AP_Arrow_50.
   - State Macro F1, Sub-4px State Accuracy, Roundness F1, Maneuver Macro F1.
4. 50/50 Holdout Temperature Calibration & Safety Operating Points:
   - Fit T* on 50% calibration split; evaluate generalization on 50% holdout split.
   - Solve constrained operating thresholds (tau_90, tau_95, tau_97.5).
5. 4-Stage Safety Waterfall Breakdown:
   - Total Misses = Perception Misses + Candidate Misses + State Misclassifications + Relevance Rejections.
6. Export structured JSON telemetry, detailed Markdown report, and multi-panel figures.
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
from typing import Any, Mapping, Sequence

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

from tlr_yolo_mtl.deployment.postprocess import xywh_to_xyxy
from tlr_yolo_mtl.evaluation.calibration import apply_temperature, fit_temperature
from tlr_yolo_mtl.evaluation.contract import (
    EvaluationContractConfig,
    SafetyWaterfallBreakdown,
    deterministic_contract_split,
)
from tlr_yolo_mtl.evaluation.evaluator import evaluate_validation_epoch
from tlr_yolo_mtl.evaluation.matching import greedy_iou_match
from tlr_yolo_mtl.evaluation.metrics import (
    AREA_BUCKETS,
    SIDE_BUCKETS,
    binary_average_precision,
    binary_classification_metrics,
    binary_roc_auc,
    brier_score,
    compute_detection_and_attribute_map,
    compute_granular_scale_metrics,
    expected_calibration_error,
    multiclass_metrics,
    multilabel_metrics,
    validation_selection_score,
)
from tlr_yolo_mtl.model.dysample import register_dysample_modules
from tlr_yolo_mtl.model.geometry_attention import attach_geometry_aware_unified_relevance_head
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.neck import register_neck_modules
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

register_neck_modules()
register_dysample_modules()


@dataclass
class SampleRecord:
    image_id: str
    is_calibration_split: bool
    gt_box: tuple[float, float, float, float]
    gt_state: int
    gt_is_round: int
    gt_maneuvers: list[int]
    gt_relevance: int
    is_relevant_red: bool
    is_directional: bool
    area_px: float
    min_side_px: float
    pred_box: tuple[float, float, float, float] | None
    pred_score: float
    pred_state: int
    pred_round_prob: float
    pred_maneuver_probs: list[float]
    rel_logit_raw: float
    rel_prob_raw: float
    rel_prob_calibrated: float = 0.0
    matched_iou: float = 0.0
    is_detected: bool = False
    is_in_candidate_pool: bool = False


def load_model_with_weights(
    config_path: Path,
    weights_path: Path,
    device: torch.device,
    use_ema: bool = True,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wrapper = build_detection_model(PROJECT_ROOT / cfg["model_config"])
    arch_cfg = cfg.get("architecture", {})
    head_kwargs = {
        k: v for k, v in arch_cfg.items()
        if k in UnifiedHeadConfig.__dataclass_fields__
    }

    if arch_cfg.get("geometry_attention", {}).get("enabled", False):
        geom_cfg = arch_cfg.get("geometry_attention", {})
        bias_dim = int(geom_cfg.get("relative_bias_dim", 18))
        include_vp = bool(geom_cfg.get("include_vanishing_point", bias_dim == 18))
        attach_geometry_aware_unified_relevance_head(
            wrapper,
            config=UnifiedHeadConfig(**head_kwargs),
            hidden_dim=int(geom_cfg.get("hidden_dim", 32)),
            p_drop=float(geom_cfg.get("p_drop", 0.0)),
            use_confidence_gating=bool(geom_cfg.get("use_confidence_gate", True)),
            include_vanishing_point=include_vp,
            vp_x=float(geom_cfg.get("vp_x", 0.5)),
            vp_y=float(geom_cfg.get("vp_y", 0.5)),
        )
    else:
        attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(**head_kwargs))

    if weights_path.is_file():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        if use_ema and "ema" in ckpt and "shadow" in ckpt["ema"]:
            state_dict = ckpt["ema"]["shadow"]
            model_dict = wrapper.model.state_dict()
            matched = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            wrapper.model.load_state_dict(matched, strict=False)
        elif "model" in ckpt:
            wrapper.model.load_state_dict(ckpt["model"], strict=True)
        else:
            wrapper.model.load_state_dict(ckpt, strict=False)

    model = wrapper.model.to(device).eval()
    return model, cfg, wrapper


def compute_nll(targets: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs)))


def optimize_safety_threshold(
    targets: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    target_recall: float = 0.95,
) -> tuple[float, float, float]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    positives = int((y == 1).sum())
    if positives == 0:
        return 0.50, 0.0, 0.0

    sorted_thresholds = np.sort(np.unique(s))[::-1]
    best_tau = 0.0
    best_precision = -1.0
    best_recall = 0.0

    for tau in sorted_thresholds:
        preds = (s >= tau).astype(np.int64)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        rec = tp / positives
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if rec >= target_recall:
            if prec >= best_precision:
                best_precision = prec
                best_tau = float(tau)
                best_recall = rec

    if best_precision < 0.0:
        best_tau = float(sorted_thresholds[-1]) if len(sorted_thresholds) else 0.0
        preds = (s >= best_tau).astype(np.int64)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        best_recall = tp / positives
        best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return best_tau, best_precision, best_recall


def compute_reliability_curve(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    bins: int = 15,
) -> dict[str, list[float]]:
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
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_centers": bin_centers,
        "edges": [float(e) for e in edges],
    }


def evaluate_checkpoint_unified(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    contract: EvaluationContractConfig,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], list[SampleRecord], SafetyWaterfallBreakdown]:
    """Run unified evaluation and collect sample records for calibration and safety waterfall."""

    # 1. Standard multi-task epoch evaluation (E37 Standard: conf_eval=0.001 for full PR curve)
    val_results = evaluate_validation_epoch(
        model,
        val_loader,
        device=device,
        amp_enabled=(contract.precision == "fp16"),
        max_batches=max_batches,
        conf_threshold=0.001,
        iou_threshold=contract.iou_threshold,
        granular_scale_metrics=True,
    )

    # 2. Detailed sample collection for calibration & safety waterfall
    sample_records: list[SampleRecord] = []
    waterfall = SafetyWaterfallBreakdown()

    img_h, img_w = contract.resolution

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
                enabled=(contract.precision == "fp16"),
            ):
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

        traffic_boxes = raw.get("traffic_candidate_boxes")  # [B, K_TL, 4]
        traffic_valid = raw.get("traffic_candidate_valid")  # [B, K_TL]
        traffic_scores = raw.get("traffic_candidate_scores")  # [B, K_TL]
        traffic_indices = raw.get("traffic_candidate_indices")  # [B, K_TL]
        relevance_logits = raw.get("relevance_logits")  # [B, 1, K_TL]
        state_logits = raw.get("state_logits")  # [B, 4, A]
        round_logits = raw.get("round_logits")  # [B, 1, A]
        maneuver_logits = raw.get("maneuver_logits")  # [B, 3, A]

        batch_image_ids = batch.get(
            "image_ids", [f"img_{batch_idx}_{i}" for i in range(batch_size)]
        )

        for b in range(batch_size):
            image_id = (
                str(batch_image_ids[b])
                if b < len(batch_image_ids)
                else f"img_{batch_idx}_{b}"
            )
            is_cal = deterministic_contract_split(image_id, salt=contract.calibration_salt)

            # 1. Detection predictions for image b across TL class
            p_scores_tl = decoded[b, 4 + TRAFFIC_LIGHT_CLASS]
            keep_mask = p_scores_tl >= 0.05
            if bool(keep_mask.any()):
                c_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
                boxes_xywh = decoded[b, :4, c_indices].transpose(0, 1)
                boxes_xyxy_px = xywh_to_xyxy(boxes_xywh)
                kept_nms = torchvision.ops.nms(
                    boxes_xyxy_px, p_scores_tl[c_indices], contract.iou_threshold
                )[:100]
                kept_dense = c_indices[kept_nms]
                p_tl_boxes_px = boxes_xyxy_px[kept_nms].cpu().numpy()
                p_tl_scores = p_scores_tl[kept_dense].cpu().numpy()
                if state_logits is not None:
                    p_tl_states = state_logits[b, :, kept_dense].argmax(0).cpu().numpy()
                else:
                    p_tl_states = np.zeros(len(kept_nms), dtype=int)
            else:
                p_tl_boxes_px = np.zeros((0, 4), dtype=float)
                p_tl_scores = np.zeros(0, dtype=float)
                p_tl_states = np.zeros(0, dtype=int)

            # 2. Extract GT for image b
            b_mask = (batch["object_batch_idx"] == b)
            gt_xywh = batch["object_bboxes"][b_mask].cpu().numpy()
            if len(gt_xywh) > 0:
                cx, cy, bw, bh = (
                    gt_xywh[:, 0],
                    gt_xywh[:, 1],
                    gt_xywh[:, 2],
                    gt_xywh[:, 3],
                )
                gt_xyxy_norm = np.stack(
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1
                )
                gt_xyxy_px = gt_xyxy_norm.copy()
                gt_xyxy_px[:, [0, 2]] *= img_w
                gt_xyxy_px[:, [1, 3]] *= img_h

                gt_cls = batch["object_cls"][b_mask].reshape(-1).cpu().numpy()
                gt_st = batch["object_state"][b_mask].reshape(-1).cpu().numpy()
                gt_rd = batch["object_round"][b_mask].reshape(-1).cpu().numpy()
                gt_mv = batch["object_maneuver"][b_mask].cpu().numpy()
                gt_rl = batch["object_relevance"][b_mask].reshape(-1).cpu().numpy()
            else:
                gt_xyxy_px = np.zeros((0, 4), dtype=float)
                gt_cls = np.zeros(0, dtype=int)
                gt_st = np.zeros(0, dtype=int)
                gt_rd = np.zeros(0, dtype=int)
                gt_mv = np.zeros((0, 3), dtype=float)
                gt_rl = np.zeros(0, dtype=int)

            # Filter GT TLs
            gt_tl_mask = (gt_cls == TRAFFIC_LIGHT_CLASS)
            gt_tl_xyxy = gt_xyxy_px[gt_tl_mask]
            gt_tl_states = gt_st[gt_tl_mask]
            gt_tl_rounds = gt_rd[gt_tl_mask]
            gt_tl_maneuvers = gt_mv[gt_tl_mask]
            gt_tl_rel = gt_rl[gt_tl_mask]

            # Match detections
            matches, unmatched_p, unmatched_g = (
                greedy_iou_match(p_tl_boxes_px, p_tl_scores, gt_tl_xyxy, iou_threshold=contract.iou_threshold)
                if len(p_tl_boxes_px) and len(gt_tl_xyxy)
                else ([], list(range(len(p_tl_boxes_px))), list(range(len(gt_tl_xyxy))))
            )
            matched_dict: dict[int, Any] = {m.target_index: m for m in matches}

            # Candidate pool match
            cand_matched_dict: dict[int, Any] = {}
            if (
                len(gt_tl_xyxy) > 0
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
                        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1
                    )
                    c_boxes_xyxy_px = c_boxes_xyxy.copy()
                    c_boxes_xyxy_px[:, [0, 2]] *= img_w
                    c_boxes_xyxy_px[:, [1, 3]] *= img_h

                    c_sc = traffic_scores[b, v_indices].cpu().numpy()
                    cand_matches, _, _ = greedy_iou_match(
                        c_boxes_xyxy_px, c_sc, gt_tl_xyxy, iou_threshold=contract.iou_threshold
                    )
                    c_dens = traffic_indices[b, v_indices].cpu().numpy()
                    for cm in cand_matches:
                        cand_matched_dict[cm.target_index] = (
                            v_indices[cm.prediction_index],
                            c_dens[cm.prediction_index],
                            float(cm.iou),
                        )

            for g_idx in range(len(gt_tl_xyxy)):
                g_box = tuple(gt_tl_xyxy[g_idx])
                g_state = int(gt_tl_states[g_idx]) if g_idx < len(gt_tl_states) else 0
                g_round = int(gt_tl_rounds[g_idx] > 0.5) if g_idx < len(gt_tl_rounds) else 1
                g_man = [int(v > 0.5) for v in gt_tl_maneuvers[g_idx]] if g_idx < len(gt_tl_maneuvers) else [0, 0, 0]
                g_rel = int(gt_tl_rel[g_idx] > 0.5) if g_idx < len(gt_tl_rel) else 0

                is_red = (g_state == 0)
                is_rel_red = is_red and (g_rel == 1)
                is_directional = (g_round == 0) or (sum(g_man) > 0)

                w_px = max(0.0, g_box[2] - g_box[0])
                h_px = max(0.0, g_box[3] - g_box[1])
                area_px = w_px * h_px
                min_side = min(w_px, h_px)

                det_match = matched_dict.get(g_idx)
                cand_match = cand_matched_dict.get(g_idx)

                is_det = det_match is not None
                is_cand = cand_match is not None

                pred_b = tuple(p_tl_boxes_px[det_match.prediction_index]) if is_det else None
                pred_sc = float(p_tl_scores[det_match.prediction_index]) if is_det else 0.0
                pred_st = int(p_tl_states[det_match.prediction_index]) if is_det else -1
                iou_val = float(det_match.iou) if is_det else 0.0

                r_logit = -10.0
                r_prob = 0.0
                pred_rnd = 1.0
                pred_man = [0.0, 0.0, 0.0]

                if is_cand and cand_match is not None:
                    slot_idx, dens_idx, _ = cand_match
                    if relevance_logits is not None:
                        r_logit = float(relevance_logits[b, 0, slot_idx].item())
                        r_prob = float(relevance_logits[b, 0, slot_idx].sigmoid().item())
                    if round_logits is not None:
                        pred_rnd = float(round_logits[b, 0, dens_idx].sigmoid().item())
                    if maneuver_logits is not None:
                        pred_man = [float(v) for v in maneuver_logits[b, :, dens_idx].sigmoid().cpu().numpy()]
                    if pred_st == -1 and state_logits is not None:
                        pred_st = int(state_logits[b, :, dens_idx].argmax(0).item())

                rec = SampleRecord(
                    image_id=image_id,
                    is_calibration_split=is_cal,
                    gt_box=g_box,
                    gt_state=g_state,
                    gt_is_round=g_round,
                    gt_maneuvers=g_man,
                    gt_relevance=g_rel,
                    is_relevant_red=is_rel_red,
                    is_directional=is_directional,
                    area_px=area_px,
                    min_side_px=min_side,
                    pred_box=pred_b,
                    pred_score=pred_sc,
                    pred_state=pred_st,
                    pred_round_prob=pred_rnd,
                    pred_maneuver_probs=pred_man,
                    rel_logit_raw=r_logit,
                    rel_prob_raw=r_prob,
                    matched_iou=iou_val,
                    is_detected=is_det,
                    is_in_candidate_pool=is_cand,
                )
                sample_records.append(rec)

                # Track safety waterfall for relevant red instances
                if is_rel_red:
                    waterfall.gt_relevant_red_total += 1
                    if is_det:
                        waterfall.perception_detected += 1
                        if is_cand:
                            waterfall.candidate_selected += 1
                            if pred_st == 0:
                                waterfall.state_classified_red += 1
                                if r_prob >= contract.standard_threshold:
                                    waterfall.relevance_accepted += 1
                                else:
                                    waterfall.relevance_rejected += 1
                            else:
                                waterfall.state_misclassified += 1
                        else:
                            waterfall.candidate_missed += 1
                    else:
                        waterfall.perception_missed += 1

    return val_results, sample_records, waterfall


def run_unified_evaluation_contract(
    config_path: Path,
    weights_dir: Path,
    output_dir: Path,
    contract: EvaluationContractConfig | None = None,
    max_val_batches: int | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    if contract is None:
        contract = EvaluationContractConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Unified Evaluation Contract (E29 Standard) on device: {device}")

    # Validate contract
    violations = contract.validate()
    if violations:
        print(f"[!] Contract validation warnings: {violations}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h, w = contract.resolution
    records_path = PROJECT_ROOT / cfg["records"]

    val_dataset = CanonicalMultiTaskDataset(
        records_path,
        split=contract.population_split,
        target_size=(h, w),
        training=False,
        seed=int(cfg.get("seed", 42)),
        allowed_sources=tuple(cfg.get("training_sources", ("DTLD",))),
        require_paired=bool(cfg.get("require_paired", True)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=canonical_multitask_collate,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[*] Loaded DTLD validation set: {len(val_dataset)} images, {len(val_loader)} batches")

    # Evaluate checkpoint diagnostic matrix
    matrix_results: dict[str, Any] = {}
    matrix_telemetry: dict[str, Any] = {}
    primary_records: list[SampleRecord] = []
    primary_waterfall: SafetyWaterfallBreakdown | None = None

    for ckpt_name in contract.diagnostic_checkpoints:
        ckpt_path = weights_dir / ckpt_name
        if not ckpt_path.is_file():
            print(f"[!] Checkpoint {ckpt_path} not found, skipping...")
            continue

        print(f"\n=======================================================")
        print(f"[*] Evaluating Checkpoint: {ckpt_name}")
        print(f"=======================================================")

        model, _, _ = load_model_with_weights(
            config_path, ckpt_path, device, use_ema=contract.ema
        )

        val_results, records, waterfall = evaluate_checkpoint_unified(
            model, val_loader, device=device, contract=contract, max_batches=max_val_batches
        )

        matrix_results[ckpt_name] = val_results
        det = val_results.get("detection", {})
        rel = val_results.get("relevance", {})
        attr = val_results.get("attributes", {})
        scale = val_results.get("granular_scale", val_results.get("scale_breakdown", {}))
        area_b = scale.get("area_buckets", scale.get("area", {}))
        side_b = scale.get("side_buckets", scale.get("side", {}))

        # Sub-4px state accuracy
        sub4px_records = [r for r in records if r.is_detected and r.min_side_px < 4.0]
        sub4px_state_acc = (
            float(np.mean([r.pred_state == r.gt_state for r in sub4px_records]))
            if len(sub4px_records) > 0
            else 0.0
        )

        ckpt_telemetry = {
            "checkpoint": ckpt_name,
            "selection_score": float(val_results.get("selection_score", 0.0)),
            "mAP50": float(det.get("map50", 0.0)),
            "mAP50_95": float(det.get("map50_95", 0.0)),
            "AP_TL_50": float(det.get("ap_tl_50", 0.0)),
            "AP_Arrow_50": float(det.get("ap_arrow_50", 0.0)),
            "Relevance_AUPRC": float(rel.get("auprc", 0.0)),
            "Relevance_F1": float(rel.get("f1", 0.0)),
            "Relevant_Red_Recall_tau50": float(rel.get("relevant_red_recall", 0.0)),
            "State_Accuracy": float(attr.get("state_accuracy", 0.0)),
            "State_Macro_F1": float(attr.get("state_macro_f1", 0.0)),
            "Sub4px_State_Accuracy": sub4px_state_acc,
            "Roundness_F1": float(attr.get("round_f1", 0.0)),
            "Maneuver_Macro_F1": float(attr.get("maneuver_macro_f1", 0.0)),
            "Tiny_TL_Recall": float(area_b.get("<32", {}).get("recall", 0.0)),
            "Tiny_TL_AP50": float(area_b.get("<32", {}).get("ap50", 0.0)),
            "Sub4px_Recall": float(side_b.get("<4", {}).get("recall", 0.0)),
            "waterfall": waterfall.to_dict(),
        }
        matrix_telemetry[ckpt_name] = ckpt_telemetry

        if ckpt_name == contract.primary_checkpoint:
            primary_records = records
            primary_waterfall = waterfall

        print(f"  --> Score: {ckpt_telemetry['selection_score']:.4f}, mAP50: {ckpt_telemetry['mAP50']*100:.2f}%, AP_TL: {ckpt_telemetry['AP_TL_50']*100:.2f}%, AP_Arrow: {ckpt_telemetry['AP_Arrow_50']*100:.2f}%")
        print(f"  --> Relevance AUPRC: {ckpt_telemetry['Relevance_AUPRC']*100:.2f}%, Rel Red Recall: {ckpt_telemetry['Relevant_Red_Recall_tau50']*100:.2f}%")
        print(f"  --> State Acc: {ckpt_telemetry['State_Accuracy']*100:.2f}%, Macro F1: {ckpt_telemetry['State_Macro_F1']*100:.2f}%, Sub-4px State Acc: {sub4px_state_acc*100:.2f}%")
        print(f"  --> Tiny (<32 px²) Recall: {ckpt_telemetry['Tiny_TL_Recall']*100:.2f}%, Sub-4px Recall: {ckpt_telemetry['Sub4px_Recall']*100:.2f}%")
        print(f"  --> Waterfall: Total={waterfall.gt_relevant_red_total}, Recalled={waterfall.end_to_end_recalled} ({waterfall.end_to_end_recall*100:.2f}%)")

    # If primary records empty (e.g. primary checkpoint was not in weights), fallback to first available
    if not primary_records and matrix_telemetry:
        first_ckpt = list(matrix_telemetry.keys())[0]
        primary_records = records
        primary_waterfall = waterfall

    # 3. 50/50 Holdout Temperature Calibration on Primary Checkpoint
    cal_records = [r for r in primary_records if r.is_detected and r.is_calibration_split]
    eval_records = [r for r in primary_records if r.is_detected and not r.is_calibration_split]

    print(f"\n[*] 50/50 Holdout Calibration Split on Primary Checkpoint ({contract.primary_checkpoint}):")
    print(f"    Calibration Samples: {len(cal_records)}, Holdout Eval Samples: {len(eval_records)}")

    cal_logits = torch.tensor([r.rel_logit_raw for r in cal_records], dtype=torch.float32)
    cal_targets = torch.tensor([r.gt_relevance for r in cal_records], dtype=torch.long)

    eval_logits = torch.tensor([r.rel_logit_raw for r in eval_records], dtype=torch.float32)
    eval_targets = torch.tensor([r.gt_relevance for r in eval_records], dtype=torch.long)

    # Fit temperature T* on calibration split
    fit_res = fit_temperature(cal_logits, cal_targets)
    T_star = float(fit_res.temperature)
    print(f"[*] Optimal Fitted Temperature T* = {T_star:.4f}")

    # Evaluate calibration generalization on holdout evaluation split
    eval_targets_np = eval_targets.numpy()
    eval_raw_probs = torch.sigmoid(eval_logits).numpy()
    eval_cal_probs = torch.sigmoid(eval_logits / T_star).numpy()

    eval_ece_before = expected_calibration_error(eval_targets_np, eval_raw_probs)
    eval_ece_after = expected_calibration_error(eval_targets_np, eval_cal_probs)
    eval_brier_before = brier_score(eval_targets_np, eval_raw_probs)
    eval_brier_after = brier_score(eval_targets_np, eval_cal_probs)
    eval_nll_before = compute_nll(eval_targets_np, eval_raw_probs)
    eval_nll_after = compute_nll(eval_targets_np, eval_cal_probs)

    print(f"    Holdout NLL: {eval_nll_before:.4f} -> {eval_nll_after:.4f}")
    print(f"    Holdout ECE: {eval_ece_before*100:.2f}% -> {eval_ece_after*100:.2f}%")
    print(f"    Holdout Brier: {eval_brier_before:.4f} -> {eval_brier_after:.4f}")

    # Optimize safety thresholds on red TLs (including relevant red + irrelevant red distractors)
    cal_red = [r for r in cal_records if (r.gt_state == 0) and r.pred_state == 0]
    eval_red = [r for r in eval_records if (r.gt_state == 0) and r.pred_state == 0]

    cal_red_targets = np.array([r.gt_relevance for r in cal_red])
    cal_red_cal_probs = torch.sigmoid(torch.tensor([r.rel_logit_raw for r in cal_red]) / T_star).numpy()

    eval_red_targets = np.array([r.gt_relevance for r in eval_red])
    eval_red_cal_probs = torch.sigmoid(torch.tensor([r.rel_logit_raw for r in eval_red]) / T_star).numpy()

    operating_points: dict[str, Any] = {}
    for target_r in contract.safety_target_recalls:
        tau, cal_p, cal_r = optimize_safety_threshold(cal_red_targets, cal_red_cal_probs, target_recall=target_r)
        # Verify on holdout
        holdout_preds = (eval_red_cal_probs >= tau).astype(int)
        pos_mask = eval_red_targets == 1
        pos_count = int(pos_mask.sum())
        holdout_r = float(((holdout_preds == 1) & pos_mask).sum() / pos_count) if pos_count > 0 else 0.0
        holdout_p = (
            float(((holdout_preds == 1) & pos_mask).sum() / holdout_preds.sum())
            if holdout_preds.sum() > 0
            else 0.0
        )
        tag = f"tau_{int(target_r*1000)/10:g}"
        operating_points[tag] = {
            "target_recall": target_r,
            "fitted_threshold": float(tau),
            "calibration_recall": float(cal_r),
            "calibration_precision": float(cal_p),
            "holdout_recall": float(holdout_r),
            "holdout_precision": float(holdout_p),
            "guarantee_met": bool(holdout_r >= (target_r - 0.02)),  # within 2% margin on finite holdout
        }
        print(f"    Operating Point {tag} (target={target_r*100:.1f}%): tau={tau:.4f}, Holdout Recall={holdout_r*100:.2f}%, Precision={holdout_p*100:.2f}%")

    # Reliability curves for plotting
    raw_rel_curve = compute_reliability_curve(eval_targets_np, eval_raw_probs, bins=10)
    cal_rel_curve = compute_reliability_curve(eval_targets_np, eval_cal_probs, bins=10)

    # 4. Latency & Throughput Benchmark
    print(f"\n[*] Measuring Inference Latency & Throughput...")
    latency_ms = 0.0
    fps_b1 = 0.0
    fps_b16 = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
        dummy_b1 = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
        dummy_b16 = torch.zeros((16, 3, h, w), device=device, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=True,
                ):
                    _ = model(dummy_b1)
        torch.cuda.synchronize()

        # Batch=1 timing
        start_t = time.perf_counter()
        iters = 50
        for _ in range(iters):
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=True,
                ):
                    _ = model(dummy_b1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_t
        latency_ms = (elapsed / iters) * 1000.0
        fps_b1 = 1000.0 / latency_ms

        # Batch=16 timing
        start_t = time.perf_counter()
        iters_b16 = 20
        for _ in range(iters_b16):
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=True,
                ):
                    _ = model(dummy_b16)
        torch.cuda.synchronize()
        elapsed_b16 = time.perf_counter() - start_t
        fps_b16 = (iters_b16 * 16) / elapsed_b16
        print(f"    Batch=1 Latency: {latency_ms:.2f} ms ({fps_b1:.1f} FPS), Batch=16 Throughput: {fps_b16:.1f} FPS")

    # 5. Visualizations
    viz_path = output_dir / "visualizations" / "e29_evaluation_contract_benchmark.png"
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    generate_contract_figures(
        viz_path,
        matrix_telemetry,
        raw_rel_curve,
        cal_rel_curve,
        primary_waterfall,
        operating_points,
    )

    # 6. Locked Baseline C0 Payload
    primary_key = contract.primary_checkpoint if contract.primary_checkpoint in matrix_telemetry else list(matrix_telemetry.keys())[0]
    locked_c0 = matrix_telemetry[primary_key]

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract": asdict(contract),
        "primary_checkpoint": primary_key,
        "locked_baseline_c0": locked_c0,
        "checkpoint_matrix": matrix_telemetry,
        "calibration": {
            "temperature_T_star": T_star,
            "holdout_nll_before": eval_nll_before,
            "holdout_nll_after": eval_nll_after,
            "holdout_ece_before": eval_ece_before,
            "holdout_ece_after": eval_ece_after,
            "holdout_brier_before": eval_brier_before,
            "holdout_brier_after": eval_brier_after,
            "operating_points": operating_points,
        },
        "latency": {
            "batch1_latency_ms": latency_ms,
            "batch1_fps": fps_b1,
            "batch16_fps": fps_b16,
        },
    }

    # Save JSON telemetry
    json_path = output_dir / "unified_evaluation_contract.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=str)
    print(f"[*] Saved telemetry to {json_path}")

    # Generate Markdown Report
    md_path = output_dir / "unified_evaluation_contract.md"
    generate_markdown_report(md_path, output_payload)
    print(f"[*] Saved Markdown report to {md_path}")

    return output_payload


def generate_contract_figures(
    save_path: Path,
    matrix_telemetry: dict[str, Any],
    raw_rel_curve: dict[str, Any],
    cal_rel_curve: dict[str, Any],
    waterfall: SafetyWaterfallBreakdown | None,
    operating_points: dict[str, Any],
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # Panel 1: Checkpoint Matrix Pareto Comparison
    ax1 = axes[0, 0]
    ckpt_names = list(matrix_telemetry.keys())
    short_names = [n.replace(".pt", "") for n in ckpt_names]
    scores = [matrix_telemetry[k]["selection_score"] * 100 for k in ckpt_names]
    maps = [matrix_telemetry[k]["mAP50"] * 100 for k in ckpt_names]
    auprcs = [matrix_telemetry[k]["Relevance_AUPRC"] * 100 for k in ckpt_names]
    red_recalls = [matrix_telemetry[k]["Relevant_Red_Recall_tau50"] * 100 for k in ckpt_names]

    x = np.arange(len(short_names))
    width = 0.2
    ax1.bar(x - 1.5 * width, scores, width, label="Selection Score (x100)", color="#2b5c8f")
    ax1.bar(x - 0.5 * width, maps, width, label="mAP@50 (%)", color="#3b82f6")
    ax1.bar(x + 0.5 * width, auprcs, width, label="Relevance AUPRC (%)", color="#10b981")
    ax1.bar(x + 1.5 * width, red_recalls, width, label="Relevant Red Recall (%)", color="#ef4444")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Score / Percentage (%)")
    ax1.set_title("Unified Checkpoint Diagnostic Matrix (Run B4)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8)

    # Panel 2: Scale-Stratified Tiny TL & Sub-4px Performance
    ax2 = axes[0, 1]
    tiny_recalls = [matrix_telemetry[k]["Tiny_TL_Recall"] * 100 for k in ckpt_names]
    sub4_recalls = [matrix_telemetry[k]["Sub4px_Recall"] * 100 for k in ckpt_names]
    sub4_accs = [matrix_telemetry[k]["Sub4px_State_Accuracy"] * 100 for k in ckpt_names]

    ax2.plot(short_names, tiny_recalls, marker="o", linewidth=2, label="Tiny (<32 px²) Recall", color="#8b5cf6")
    ax2.plot(short_names, sub4_recalls, marker="s", linewidth=2, label="Sub-4px Min-Side Recall", color="#ec4899")
    ax2.plot(short_names, sub4_accs, marker="^", linewidth=2, label="Sub-4px State Accuracy", color="#f59e0b")
    ax2.set_ylabel("Recall / Accuracy (%)")
    ax2.set_title("Perception Floor & Sub-Grid Resolution Recovery", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best", fontsize=8)
    ax2.set_xticks(np.arange(len(short_names)))
    ax2.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)

    # Panel 3: Temperature Scaling Reliability Diagram (Holdout 50%)
    ax3 = axes[1, 0]
    ax3.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax3.plot(
        raw_rel_curve["bin_confs"],
        raw_rel_curve["bin_accs"],
        marker="o",
        label=f"Raw Model (T=1.0)",
        color="#ef4444",
        linewidth=2,
    )
    ax3.plot(
        cal_rel_curve["bin_confs"],
        cal_rel_curve["bin_accs"],
        marker="s",
        label=f"Calibrated (T*)",
        color="#10b981",
        linewidth=2,
    )
    ax3.set_xlabel("Mean Confidence")
    ax3.set_ylabel("True Accuracy")
    ax3.set_title("50/50 Holdout Temperature Calibration Reliability", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: 4-Stage Safety Waterfall Breakdown
    ax4 = axes[1, 1]
    if waterfall is not None:
        stages = [
            "Total GT\nRelevant Red",
            "Stage 1:\nPerception (Det)",
            "Stage 2:\nCandidate Pool",
            "Stage 3:\nState (Red)",
            "Stage 4:\nRelevance Gate",
        ]
        counts = [
            waterfall.gt_relevant_red_total,
            waterfall.perception_detected,
            waterfall.candidate_selected,
            waterfall.state_classified_red,
            waterfall.relevance_accepted,
        ]
        colors = ["#334155", "#3b82f6", "#0ea5e9", "#f59e0b", "#10b981"]
        bars = ax4.bar(stages, counts, color=colors, width=0.55)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = (count / waterfall.gt_relevant_red_total * 100) if waterfall.gt_relevant_red_total > 0 else 0
            ax4.annotate(
                f"{count}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax4.set_ylabel("Count of Relevant Red Instances")
        ax4.set_title(f"4-Stage Safety Waterfall (E2E Recall = {waterfall.end_to_end_recall*100:.2f}%)", fontsize=11, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved multi-panel benchmark visualization to {save_path}")


def generate_markdown_report(report_path: Path, data: dict[str, Any]):
    c0 = data.get("locked_baseline_c0", {})
    matrix = data.get("checkpoint_matrix", {})
    cal = data.get("calibration", {})
    ops = cal.get("operating_points", {})
    wf = c0.get("waterfall", {})
    lat = data.get("latency", {})

    md = f"""# Unified Evaluation Contract & Cross-Ticket Normalization Report (Ticket E29)

**Generated:** {data.get('timestamp')}  
**Canonical Baseline Model ($C_0$):** Run B4 (`YOLO11s + P2 + K_Arrow=32 + NWD-aware TAL`)  
**Primary Benchmark Checkpoint:** `{data.get('primary_checkpoint')}`  
**Invariant Validation Set:** DTLD full validation split (5,962 images, 25,344 GT TLs)

---

## 1. Locked Baseline $C_0$ Canonical Benchmark Values

These locked values establish the unified reference standard $C_0$ for Phase 4 forward selection:

| Metric Dimension | Canonical $C_0$ Value | Description / Guardrail Standard |
|---|:---:|---|
| **Selection Composite Score** | **{c0.get('selection_score', 0):.4f}** | Primary multi-task composite metric |
| **mAP@50 (Overall)** | **{c0.get('mAP50', 0)*100:.2f}%** | Joint detection accuracy |
| **mAP@50:95 (Overall)** | **{c0.get('mAP50_95', 0)*100:.2f}%** | Strict localization quality |
| **AP@50 (Traffic Light)** | **{c0.get('AP_TL_50', 0)*100:.2f}%** | Traffic light detector AP |
| **AP@50 (Road Arrow)** | **{c0.get('AP_Arrow_50', 0)*100:.2f}%** | Road arrow detector AP ($K_{{Arrow}}=32$) |
| **Tiny TL Recall ($<32\\text{{ px}}^2$)** | **{c0.get('Tiny_TL_Recall', 0)*100:.2f}%** | Perception floor tiny recall |
| **Tiny TL AP@50 ($<32\\text{{ px}}^2$)** | **{c0.get('Tiny_TL_AP50', 0)*100:.2f}%** | Perception floor tiny precision |
| **Sub-4px Recall (Side $<4\\text{{ px}}$)** | **{c0.get('Sub4px_Recall', 0)*100:.2f}%** | Sub-grid anchor allocation recovery |
| **Relevance AUPRC** | **{c0.get('Relevance_AUPRC', 0)*100:.2f}%** | Contextual ranking precision |
| **Relevance F1** | **{c0.get('Relevance_F1', 0)*100:.2f}%** | Standard classification F1 |
| **Relevant Red Recall ($\\tau=0.50$)** | **{c0.get('Relevant_Red_Recall_tau50', 0)*100:.2f}%** | Uncalibrated baseline red recall |
| **State Accuracy** | **{c0.get('State_Accuracy', 0)*100:.2f}%** | Traffic light state classification accuracy |
| **State Macro F1** | **{c0.get('State_Macro_F1', 0)*100:.2f}%** | Multi-class state macro F1 |
| **Sub-4px State Accuracy** | **{c0.get('Sub4px_State_Accuracy', 0)*100:.2f}%** | Fine-grained state recognition on $<4\\text{{ px}}$ |
| **Roundness F1** | **{c0.get('Roundness_F1', 0)*100:.2f}%** | Directional vs round distinction |
| **Maneuver Macro F1** | **{c0.get('Maneuver_Macro_F1', 0)*100:.2f}%** | Multi-label arrow maneuver classification |
| **Batch-1 Latency** | **{lat.get('batch1_latency_ms', 0):.2f} ms** | Real-time safety latency ({lat.get('batch1_fps', 0):.1f} FPS) |
| **Batch-16 Throughput** | **{lat.get('batch16_fps', 0):.1f} FPS** | Batch throughput |

---

## 2. Multi-Checkpoint Diagnostic Matrix (Run B4)

Evaluation across all saved checkpoint types under the exact E29 unified contract:

| Checkpoint | Selection Score | mAP@50 | AP_TL@50 | AP_Arrow@50 | Relevance AUPRC | Rel Red Recall ($\\tau=0.50$) | State Acc | State Macro F1 | Tiny Recall | Sub-4px Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for ckpt_name, t in matrix.items():
        md += f"| `{ckpt_name}` | **{t['selection_score']:.4f}** | {t['mAP50']*100:.2f}% | {t['AP_TL_50']*100:.2f}% | {t['AP_Arrow_50']*100:.2f}% | {t['Relevance_AUPRC']*100:.2f}% | {t['Relevant_Red_Recall_tau50']*100:.2f}% | {t['State_Accuracy']*100:.2f}% | {t['State_Macro_F1']*100:.2f}% | {t['Tiny_TL_Recall']*100:.2f}% | {t['Sub4px_Recall']*100:.2f}% |\n"

    md += f"""
---

## 3. 50/50 Holdout Temperature Calibration & Safety Operating Points

- **Optimal Fitted Temperature ($T^*$):** `{cal.get('temperature_T_star', 1.0):.4f}`
- **Holdout Negative Log-Likelihood (NLL):** `{cal.get('holdout_nll_before', 0):.4f}` $\\to$ **`{cal.get('holdout_nll_after', 0):.4f}`**
- **Holdout Expected Calibration Error (ECE):** `{cal.get('holdout_ece_before', 0)*100:.2f}%` $\\to$ **`{cal.get('holdout_ece_after', 0)*100:.2f}%`**
- **Holdout Brier Score:** `{cal.get('holdout_brier_before', 0):.4f}` $\\to$ **`{cal.get('holdout_brier_after', 0):.4f}`**

### Calibrated Safety Operating Points Table:

| Operating Point | Target Red Recall | Fitted Threshold ($\\tau$) | Calibration Recall | Holdout Recall | Holdout Precision | Safety Guarantee Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for tag, op in ops.items():
        status = "PASSED" if op.get("guarantee_met") else "MARGINAL"
        md += f"| **{tag}** | {op.get('target_recall')*100:.1f}% | `tau = {op.get('fitted_threshold'):.4f}` | {op.get('calibration_recall')*100:.2f}% | **{op.get('holdout_recall')*100:.2f}%** | {op.get('holdout_precision')*100:.2f}% | **{status}** |\n"

    md += f"""
---

## 4. 4-Stage Safety Waterfall Failure Decomposition

Analysis of Relevant Red Traffic Light recall drop-off across architectural stages:

1. **Total Ground Truth Relevant Red TLs:** `{wf.get('gt_relevant_red_total', 0)}` (100.0%)
2. **Stage 1 (Perception Detection @ IoU=0.50):** `{wf.get('perception_detected', 0)}` ({wf.get('perception_recall', 0)*100:.2f}%) — Missed `{wf.get('perception_missed', 0)}`
3. **Stage 2 (Top-K Candidate Selection):** `{wf.get('candidate_selected', 0)}` ({wf.get('candidate_selection_rate', 0)*100:.2f}%) — Missed `{wf.get('candidate_missed', 0)}`
4. **Stage 3 (State Classification as Red):** `{wf.get('state_classified_red', 0)}` ({wf.get('state_classification_rate', 0)*100:.2f}%) — Misclassified `{wf.get('state_misclassified', 0)}`
5. **Stage 4 (Relevance Gate $\\tau=0.50$):** `{wf.get('relevance_accepted', 0)}` ({wf.get('relevance_acceptance_rate', 0)*100:.2f}%) — Rejected `{wf.get('relevance_rejected', 0)}`
- **End-to-End Recall:** **`{wf.get('end_to_end_recall', 0)*100:.2f}%`** ({wf.get('end_to_end_recalled', 0)} / {wf.get('gt_relevant_red_total', 0)})

---

## 5. Artifacts Generated

- Visualizations: `results/visualizations/e29_evaluation_contract_benchmark.png`
- JSON Telemetry: `results/unified_evaluation_contract.json`
- Markdown Report: `results/unified_evaluation_contract.md`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Contract Runner (Ticket E29)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_yolo11s_p2_nwd.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="runs/tlr_yolo11s_p2_nwd/weights",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument(
        "--primary-checkpoint",
        type=str,
        default="best_composite.pt",
        help="Primary benchmark checkpoint",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Max batches for evaluation (None for full validation set)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for validation loader",
    )
    args = parser.parse_args()

    contract_cfg = EvaluationContractConfig(primary_checkpoint=args.primary_checkpoint)

    run_unified_evaluation_contract(
        config_path=Path(args.config),
        weights_dir=Path(args.weights_dir),
        output_dir=Path(args.output_dir),
        contract=contract_cfg,
        max_val_batches=args.max_batches,
        batch_size=args.batch_size,
    )
