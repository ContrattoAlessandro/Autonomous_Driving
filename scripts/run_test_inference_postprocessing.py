"""Complete inference and evaluation pipeline on the test/val split with full multi-task postprocessing.

Runs:
1. End-to-end multi-task inference across test images using postprocess_multitask_outputs (Class-aware NMS).
2. Computes strict evaluation metrics:
   - Detection: mAP50, mAP50-95, AP_TL50, AP_Arrow50, AP_small, AP_medium
   - Attributes: State Accuracy, State Macro F1, Round F1, Maneuver Macro F1
   - Relevance: AUPRC, F1-Score, Precision, Recall, Relevant Red Recall
3. Generates high-resolution visualization overlays comparing Ground Truth vs Predictions on representative test cases.
4. Outputs comprehensive JSON telemetry and markdown summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import cv2
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from tlr_yolo_mtl.deployment.postprocess import postprocess_multitask_outputs, xywh_to_xyxy
from tlr_yolo_mtl.evaluation.evaluator import evaluate_validation_epoch
from tlr_yolo_mtl.evaluation.matching import greedy_iou_match
from tlr_yolo_mtl.evaluation.metrics import (
    AREA_BUCKETS,
    SIDE_BUCKETS,
    binary_classification_metrics,
    multiclass_confusion_matrix,
    multiclass_metrics,
    multilabel_metrics,
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
from tlr_yolo_mtl.training.engine import build_multitask_criterion

STATE_NAMES = ["red", "yellow", "green", "off"]
STATE_COLORS = {
    "red": (0, 0, 255),       # BGR Red
    "yellow": (0, 215, 255),  # BGR Amber
    "green": (0, 255, 0),     # BGR Green
    "off": (140, 140, 140),   # BGR Gray
}
DIRECTION_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    if not cfg:
        with open(PROJECT_ROOT / "configs" / "train_yolo11s_p2_nwd.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    wrapper = build_detection_model(cfg["model_config"])
    arch_cfg = cfg.get("architecture", {})
    head_kwargs = {
        k: v for k, v in arch_cfg.items()
        if k in UnifiedHeadConfig.__dataclass_fields__
    }
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(**head_kwargs))

    state_dict = payload.get("model", payload)
    wrapper.model.load_state_dict(state_dict, strict=True)
    model = wrapper.model.to(device).eval()
    return model, cfg, payload


def run_test_inference_pipeline(
    checkpoint_path: Path,
    split: str = "test",
    batch_size: int = 16,
    workers: int = 2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_eval_samples: int | None = None,
    num_visualizations: int = 12,
    output_dir: Path = Path("results/inference_test_yolo11s_p2_nwd"),
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*95}")
    print(f"TLR-YOLO-MTL TEST INFERENCE & EVALUATION PIPELINE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split.upper()} | Postprocessing NMS: Conf >= {conf_threshold:.2f}, IoU <= {iou_threshold:.2f}")
    print(f"{'='*95}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, payload = load_checkpoint(checkpoint_path, device)
    criterion = build_multitask_criterion(model, cfg)

    # 1. Dataset & DataLoader
    records_path = PROJECT_ROOT / cfg["records"]
    target_size = tuple(cfg["input_size"])
    dataset = CanonicalMultiTaskDataset(
        records_path,
        split=split,
        target_size=target_size,
        training=False,
        seed=int(cfg.get("seed", 42)),
        allowed_sources=tuple(cfg.get("training_sources", ("DTLD",))),
        require_paired=bool(cfg.get("require_paired", True)),
    )
    print(f"Loaded {len(dataset)} {split} samples from {records_path}.")

    max_batches = None
    if max_eval_samples is not None:
        max_batches = int(np.ceil(max_eval_samples / batch_size))
        print(f"Evaluating subset: {max_eval_samples} samples (~{max_batches} batches).")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=canonical_multitask_collate,
    )

    # 2. Run Comprehensive Quantitative Evaluation with NMS
    print("\n[Step 1/3] Running Quantitative Evaluation across multi-task heads...")
    t0 = time.time()
    eval_results = evaluate_validation_epoch(
        model,
        loader,
        criterion=criterion,
        device=device,
        amp_enabled=True,
        max_batches=max_batches,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        granular_scale_metrics=True,
    )
    eval_time = time.time() - t0
    num_samples = eval_results["samples_evaluated"]
    fps = num_samples / max(1e-6, eval_time)

    det = eval_results["detection"]
    rel = eval_results["relevance"]
    att = eval_results["attributes"]
    loss = eval_results["mean_losses"]
    score = eval_results["selection_score"]

    print(f"\n--- EVALUATION COMPLETED IN {eval_time:.2f}s ({fps:.1f} FPS su {num_samples} immagini) ---")
    print(f"★ COMPOSITE SELECTION SCORE: {score:.4f}")
    print("\n[1. METRICHE DETECTION (mAP con Postprocessing NMS)]")
    print(f"  • mAP@50 (Globale)          : {det['map50']:.4f} ({det['map50']:.1%})")
    print(f"  • mAP@50-95 (Globale)       : {det['map50_95']:.4f} ({det['map50_95']:.1%})")
    print(f"  • AP Traffic Lights @50     : {det['ap_tl_50']:.4f} ({det['ap_tl_50']:.1%})")
    print(f"  • AP Road Arrows @50        : {det['ap_arrow_50']:.4f} ({det['ap_arrow_50']:.1%})")
    print(f"  • AP Tiny Lights (Small)    : {det['ap_small']:.4f} ({det['ap_small']:.1%})")
    print(f"  • AP Medium Lights/Arrows   : {det['ap_medium']:.4f} ({det['ap_medium']:.1%})")
    print(f"  • mAP State Joint           : {det['map_state']:.4f} ({det['map_state']:.1%})")

    print("\n[2. METRICHE RELEVANCE (Cross-Attention & Corsia Ego)]")
    print(f"  • Relevance AUPRC           : {rel['auprc']:.4f}")
    print(f"  • Relevance F1-Score        : {rel['f1']:.4f}")
    print(f"  • Relevance Precision       : {rel['precision']:.4f} ({rel['precision']:.1%})")
    print(f"  • Relevance Recall          : {rel['recall']:.4f} ({rel['recall']:.1%})")

    print("\n[3. METRICHE ATTRIBUTI E CLASSIFICAZIONE]")
    print(f"  • State Accuracy (4-class)  : {att['state_accuracy']:.4f} ({att['state_accuracy']:.1%})")
    print(f"  • State Macro F1            : {att['state_macro_f1']:.4f}")
    print(f"  • Round Signal F1           : {att['round_f1']:.4f}")
    print(f"  • Maneuver Macro F1         : {att['maneuver_macro_f1']:.4f}")

    # 3. Qualitative Visualizations on Rich Test Samples
    print(f"\n[Step 2/3] Generating {num_visualizations} side-by-side visualization overlays...")
    selected_records = []
    seen_ids = set()
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != split:
                continue
            img_id = r.get("image_id")
            if img_id in seen_ids:
                continue

            tls = r.get("traffic_lights", [])
            arrows = r.get("road_arrows", [])
            has_arrows = len(arrows) > 0
            has_tls = len(tls) > 0
            has_rel = any(tl.get("relevance") == 1 for tl in tls)
            has_irrel = any(tl.get("relevance") == 0 for tl in tls)

            # Prioritize rich mixed scenes
            if has_arrows and has_tls and len(selected_records) < (num_visualizations // 2):
                selected_records.append(r)
                seen_ids.add(img_id)
                continue
            elif has_rel and has_irrel and len(selected_records) < (num_visualizations * 3 // 4):
                selected_records.append(r)
                seen_ids.add(img_id)
                continue
            elif has_tls and len(selected_records) < num_visualizations:
                selected_records.append(r)
                seen_ids.add(img_id)
                continue

            if len(selected_records) >= num_visualizations:
                break

    saved_vis_paths = []
    for idx, rec in enumerate(selected_records, 1):
        img_path = Path(rec["image_path"])
        if not img_path.exists():
            continue
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        # Inference on raw sample with postprocess_multitask_outputs
        input_img = cv2.resize(raw_bgr, (target_size[1], target_size[0]))
        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            preds = model(tensor)
            if isinstance(preds, tuple):
                decoded, raw = preds
            else:
                decoded, raw = preds.get(0, preds.get("decoded")), preds

            eleven_tensors = (
                decoded,
                raw["state_logits"],
                raw["round_logits"],
                raw["maneuver_logits"],
                raw["ego_lane_logits"],
                raw["traffic_candidate_indices"],
                raw["traffic_candidate_valid"],
                raw["arrow_candidate_indices"],
                raw["arrow_candidate_valid"],
                raw["relevance_logits"],
                raw["attention_weights"],
            )
            post = postprocess_multitask_outputs(
                eleven_tensors,
                traffic_confidence=conf_threshold,
                arrow_confidence=conf_threshold,
                iou_threshold=iou_threshold,
            )

        vis_img = draw_side_by_side_overlay(raw_bgr, rec, post, target_size)
        out_vis_file = vis_dir / f"test_sample_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_vis_file), vis_img)
        saved_vis_paths.append(str(out_vis_file))

    print(f"Saved {len(saved_vis_paths)} visual overlays in: {vis_dir}")

    # 4. Save Final Report JSON
    print("\n[Step 3/3] Saving Full Telemetry & Summary Report...")
    def make_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    final_report = {
        "schema": "TLR-YOLO-MTL Test Inference and Postprocessing Report v1",
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples_evaluated": num_samples,
        "eval_time_seconds": eval_time,
        "throughput_fps": fps,
        "postprocessing_config": {
            "confidence_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
        },
        "composite_selection_score": float(score),
        "detection_metrics": make_serializable(det),
        "relevance_metrics": make_serializable(rel),
        "attribute_metrics": make_serializable(att),
        "validation_losses": make_serializable(loss),
        "visualizations": saved_vis_paths,
    }

    report_json_path = output_dir / "test_inference_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Markdown Summary
    md_summary = f"""# TLR-YOLO-MTL (YOLO11s + P2 + NWD) Test Inference Report

- **Checkpoint Evaluated**: `{checkpoint_path.name}`
- **Dataset Split**: `{split.upper()}` ({num_samples:,} images evaluated)
- **Inference Throughput**: **{fps:.1f} FPS** ({eval_time:.1f}s total on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})
- **Postprocessing NMS**: Confidence $\\ge {conf_threshold:.2f}$, IoU $\\le {iou_threshold:.2f}$

## 1. Key Metrics Table

| Metric Category | Metric Name | Value | Percentage / Interpretation |
|---|---|:---:|:---:|
| **Composite Score** | `Selection Score` | **{score:.4f}** | Multi-Task Global Harmonic Score |
| **Object Detection** | `mAP@50` | **{det['map50']:.4f}** | **{det['map50']:.1%}** |
| **Object Detection** | `mAP@50-95` | **{det['map50_95']:.4f}** | **{det['map50_95']:.1%}** (Strict Localization) |
| **Traffic Light AP** | `AP_TL@50` | **{det['ap_tl_50']:.4f}** | **{det['ap_tl_50']:.1%}** |
| **Road Arrow AP** | `AP_Arrow@50` | **{det['ap_arrow_50']:.4f}** | **{det['ap_arrow_50']:.1%}** |
| **Small Object AP** | `AP_small` | **{det['ap_small']:.4f}** | **{det['ap_small']:.1%}** (P2 Neck + NWD effect) |
| **Relevance** | `AUPRC` | **{rel['auprc']:.4f}** | Directional & Lane Pertinence |
| **Relevance** | `F1-Score` | **{rel['f1']:.4f}** | Balance between Precision & Recall |
| **Relevance** | `Precision` | **{rel['precision']:.4f}** | **{rel['precision']:.1%}** |
| **Relevance** | `Recall` | **{rel['recall']:.4f}** | **{rel['recall']:.1%}** |
| **Attributes** | `State Accuracy` | **{att['state_accuracy']:.4f}** | **{att['state_accuracy']:.1%}** (Red/Yellow/Green/Off) |
| **Attributes** | `State Macro F1` | **{att['state_macro_f1']:.4f}** | Unweighted Color Class Balance |
| **Attributes** | `Round Signal F1` | **{att['round_f1']:.4f}** | Round vs Directional Identification |
| **Attributes** | `Maneuver Macro F1` | **{att['maneuver_macro_f1']:.4f}** | Arrow/Direction Classification |

---
Report and visual overlays saved to: `{output_dir}`
"""
    report_md_path = output_dir / "test_inference_summary.md"
    report_md_path.write_text(md_summary, encoding="utf-8")
    print(f"Saved Markdown Summary to: {report_md_path}")
    print(f"{'='*95}\n")
    return final_report


def draw_side_by_side_overlay(
    img_bgr: np.ndarray,
    record: dict,
    post: dict,
    target_size: tuple[int, int],
) -> np.ndarray:
    orig_h, orig_w, _ = img_bgr.shape
    vis_w, vis_h = target_size[1], target_size[0]

    left_panel = cv2.resize(img_bgr, (vis_w, vis_h))
    right_panel = left_panel.copy()

    scale_x = vis_w / orig_w
    scale_y = vis_h / orig_h

    # 1. Draw GT on Left Panel
    gt_tls = record.get("traffic_lights", [])
    for tl in gt_tls:
        box = tl.get("bbox_xyxy", [0, 0, 0, 0])
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        state = tl.get("state") or "unknown"
        rel = tl.get("relevance", -1)
        rel_str = "REL" if rel == 1 else ("IRR" if rel == 0 else "N/A")
        color = STATE_COLORS.get(state, (200, 200, 200))
        cv2.rectangle(left_panel, (x1, y1), (x2, y2), color, 2)
        tag = f"GT:{str(state)[:3]}|{rel_str}"
        cv2.putText(left_panel, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    gt_arrows = record.get("road_arrows", [])
    for arr in gt_arrows:
        box = arr.get("bbox_xyxy", [0, 0, 0, 0])
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)
        cv2.rectangle(left_panel, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(left_panel, "GT:ARROW", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1, cv2.LINE_AA)

    # 2. Draw Predictions on Right Panel
    tl_post = post["traffic_lights"]
    pred_tl_boxes = tl_post["boxes_xyxy"][0].cpu().numpy()
    pred_tl_scores = tl_post["detection_scores"][0].cpu().numpy()
    pred_tl_valid = tl_post["valid"][0].cpu().numpy().astype(bool)
    pred_tl_states = tl_post["state_indices"][0].cpu().numpy()
    pred_tl_rel_probs = tl_post["relevance_probabilities"][0].cpu().numpy()

    for i in range(len(pred_tl_valid)):
        if not pred_tl_valid[i]:
            continue
        box = pred_tl_boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        score = pred_tl_scores[i]
        st_id = pred_tl_states[i]
        st_name = STATE_NAMES[st_id] if 0 <= st_id < len(STATE_NAMES) else "unk"
        rel_p = pred_tl_rel_probs[i]
        rel_tag = "REL" if rel_p >= 0.50 else "IRR"

        color = STATE_COLORS.get(st_name, (0, 255, 255))
        cv2.rectangle(right_panel, (x1, y1), (x2, y2), color, 2)
        label = f"{st_name[:3]} {score:.2f}|{rel_tag}({rel_p:.0%})"
        cv2.putText(right_panel, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    arr_post = post["road_arrows"]
    pred_arr_boxes = arr_post["boxes_xyxy"][0].cpu().numpy()
    pred_arr_scores = arr_post["detection_scores"][0].cpu().numpy()
    pred_arr_valid = arr_post["valid"][0].cpu().numpy().astype(bool)
    for i in range(len(pred_arr_valid)):
        if not pred_arr_valid[i]:
            continue
        box = pred_arr_boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        score = pred_arr_scores[i]
        cv2.rectangle(right_panel, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(right_panel, f"ARR {score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    # 3. Header & Side-by-side stitch
    header_h = 40
    header = np.zeros((header_h, vis_w * 2, 3), dtype=np.uint8)
    cv2.putText(header, f"GROUND TRUTH: {record.get('image_id')} (TLs: {len(gt_tls)}, Arrows: {len(gt_arrows)})", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f"PREDICTION + NMS: (TLs: {int(pred_tl_valid.sum())}, Arrows: {int(pred_arr_valid.sum())})", (vis_w + 20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    body = np.concatenate([left_panel, right_panel], axis=1)
    combined = np.concatenate([header, body], axis=0)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/tlr_yolo11s_p2_nwd/weights/best_composite.pt"), help="Model checkpoint path")
    parser.add_argument("--split", type=str, default="val", choices=["test", "val", "train"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="NMS confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples to evaluate")
    parser.add_argument("--num-vis", type=int, default=12, help="Number of visual overlay examples to save")
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference_test_yolo11s_p2_nwd"), help="Output directory")
    args = parser.parse_args()

    run_test_inference_pipeline(
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_eval_samples=args.max_samples,
        num_visualizations=args.num_vis,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
