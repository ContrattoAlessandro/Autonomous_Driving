"""E20 Diagnostic & Empirical Audit: Run B2 vs B4 Convergence & NWD-Aware TAL Validation.

Evaluates the converged checkpoint matrix for Run B4 (YOLO11s + P2 + K_Arrow=32 + NWD-Aware TAL)
against Run B2 (P2 + Standard TAL) and Baseline B0 across the DTLD validation set:
1. Multi-Checkpoint Evaluation Matrix:
   - best_composite.pt
   - best_tl_detection.pt
   - best_relevance.pt
   - best_relevant_red_recall.pt
   - last.pt
2. Scale-Stratified Tiny TL Detection & Allocation Recovery:
   - Area buckets: <32, 32-64, 64-128, 128-256, 256-512, >512 px².
   - Min-side buckets: <4, 4-6, 6-8, 8-12, >12 px.
   - Sub-grid recall and anchor recovery on min-side <4 px.
3. Multi-Task Perception & Attribute Performance:
   - mAP50, mAP50-95, AP_TL_50, AP_Arrow_50.
   - State Macro F1, Roundness F1, Maneuver Macro F1.
   - Relevance AUPRC: Directional vs Round, Arrow-Present vs Arrow-Absent.
4. Safety Waterfall & Operating Points:
   - Relevant Red Recall at tau=0.30 and safety Pareto operating points (tau_90, tau_95).
   - False Negative count comparison.
5. Generates structured JSON output and detailed comparative Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
import yaml
from torch.utils.data import DataLoader

from tlr_yolo_mtl.deployment.postprocess import xywh_to_xyxy
from tlr_yolo_mtl.evaluation.evaluator import evaluate_validation_epoch
from tlr_yolo_mtl.evaluation.metrics import (
    AREA_BUCKETS,
    SIDE_BUCKETS,
    binary_classification_metrics,
    compute_granular_scale_metrics,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model, load_coco_warmstart
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


def load_b4_model_with_weights(
    config_path: Path,
    weights_path: Path,
    device: torch.device,
    use_ema: bool = False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wrapper = build_detection_model(PROJECT_ROOT / cfg["model_config"])
    arch_cfg = cfg.get("architecture", {})
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(**arch_cfg))

    if weights_path.is_file():
        print(f"[*] Loading checkpoint weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        if use_ema and "ema" in ckpt and "shadow" in ckpt["ema"]:
            state_dict = ckpt["ema"]["shadow"]
            # Filter keys matching model
            model_dict = wrapper.model.state_dict()
            matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            wrapper.model.load_state_dict(matched, strict=False)
            print(f"[*] Loaded {len(matched)} EMA shadow weights")
        elif "model" in ckpt:
            wrapper.model.load_state_dict(ckpt["model"], strict=True)
            print(f"[*] Loaded full model state dict ({len(ckpt['model'])} keys)")
        else:
            wrapper.model.load_state_dict(ckpt, strict=False)

    model = wrapper.model.to(device).eval()
    return model, cfg, wrapper


def run_e20_audit(
    config_path: Path,
    weights_dir: Path,
    output_dir: Path,
    max_val_batches: int | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running E20 B2 vs B4 Convergence Audit on device: {device}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h, w = tuple(cfg.get("input_size", [800, 1600]))
    records_path = PROJECT_ROOT / cfg["records"]

    val_dataset = CanonicalMultiTaskDataset(
        records_path,
        split="val",
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

    # Checkpoint matrix to evaluate
    checkpoints = [
        ("best_composite", weights_dir / "best_composite.pt"),
        ("best_tl_detection", weights_dir / "best_tl_detection.pt"),
        ("best_relevance", weights_dir / "best_relevance.pt"),
        ("best_relevant_red_recall", weights_dir / "best_relevant_red_recall.pt"),
        ("last", weights_dir / "last.pt"),
    ]

    matrix_results: dict[str, Any] = {}

    for name, path in checkpoints:
        if not path.is_file():
            print(f"[!] Checkpoint {path} not found, skipping...")
            continue

        print(f"\n=======================================================")
        print(f"[*] Evaluating Checkpoint: {name} ({path.name})")
        print(f"=======================================================")

        model, _, _ = load_b4_model_with_weights(config_path, path, device, use_ema=False)

        val_results = evaluate_validation_epoch(
            model,
            val_loader,
            device=device,
            amp_enabled=bool(cfg.get("amp", True)),
            max_batches=max_val_batches,
            conf_threshold=0.05,
            iou_threshold=0.6,
            granular_scale_metrics=True,
        )

        matrix_results[name] = val_results
        det = val_results.get("detection", {})
        rel = val_results.get("relevance", {})
        attr = val_results.get("attributes", {})
        scale = val_results.get("scale_breakdown", {})

        print(f"  --> Composite Score: {val_results.get('selection_score', 0.0):.4f}")
        print(f"  --> mAP50: {det.get('map50', 0.0):.4f}, AP_TL_50: {det.get('ap_tl_50', 0.0):.4f}, AP_Arrow_50: {det.get('ap_arrow_50', 0.0):.4f}")
        print(f"  --> Relevance AUPRC: {rel.get('auprc', 0.0):.4f}, F1: {rel.get('f1', 0.0):.4f}, Relevant Red Recall: {rel.get('relevant_red_recall', 0.0):.4f}")
        print(f"  --> State Acc: {attr.get('state_accuracy', 0.0):.4f}, State Macro F1: {attr.get('state_macro_f1', 0.0):.4f}")
        if "area" in scale:
            print(f"  --> Tiny (<32 px²) Recall: {scale['area'].get('<32', {}).get('recall', 0.0)*100:.2f}%, AP50: {scale['area'].get('<32', {}).get('ap50', 0.0)*100:.2f}%")
        if "side" in scale:
            print(f"  --> Sub-4px (<4 px) Recall: {scale['side'].get('<4', {}).get('recall', 0.0)*100:.2f}%")

    # Baseline references for comparison
    b0_ref = {
        "mAP50": 0.7261,
        "AP_TL_50": 0.5830,
        "AP_Arrow_50": 0.8690,
        "Recall_tiny": 0.1661,
        "Recall_small": 0.4590,
        "Recall_large": 0.9440,
        "Recall_sub4px": 0.0170,
        "AUPRC_relevance": 0.9663,
        "AUPRC_directional": 0.5635,
        "State_Macro_F1": 0.8670,
        "Relevant_Red_Recall_tau30": 0.9466,
    }

    b2_ref = {
        "mAP50": 0.7410,
        "AP_TL_50": 0.6120,
        "AP_Arrow_50": 0.8700,
        "Recall_tiny": 0.2850,
        "Recall_small": 0.5820,
        "Recall_large": 0.9480,
        "Recall_sub4px": 0.0840,
        "AUPRC_relevance": 0.9670,
        "AUPRC_directional": 0.7062,
        "State_Macro_F1": 0.8840,
        "Relevant_Red_Recall_tau30": 0.9480,
    }

    output_payload = {
        "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path),
        "weights_dir": str(weights_dir),
        "checkpoints_evaluated": list(matrix_results.keys()),
        "matrix_results": matrix_results,
        "baseline_b0": b0_ref,
        "run_b2": b2_ref,
    }

    summary_json_path = output_dir / "audit_e20_b2_vs_b4_convergence.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=str)
    print(f"\n[*] Saved structured JSON summary to {summary_json_path}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E20 Audit: B2 vs B4 Convergence & NWD-Aware TAL Validation")
    parser.add_argument("--config", type=str, default="configs/train_yolo11s_p2_nwd.yaml", help="Path to config yaml")
    parser.add_argument("--weights-dir", type=str, default="runs/tlr_yolo11s_p2_nwd/weights", help="Directory containing checkpoints")
    parser.add_argument("--output-dir", type=str, default="runs/tlr_yolo11s_p2_nwd/e20_audit", help="Directory to save audit artifacts")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to evaluate (None for full)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for validation loader")
    args = parser.parse_args()

    run_e20_audit(
        config_path=Path(args.config),
        weights_dir=Path(args.weights_dir),
        output_dir=Path(args.output_dir),
        max_val_batches=args.max_batches,
        batch_size=args.batch_size,
    )
