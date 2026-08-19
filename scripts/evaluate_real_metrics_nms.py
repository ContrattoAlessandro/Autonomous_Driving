"""Comprehensive evaluation with NMS on the complete validation set.

Computes the real final metrics across all tasks:
- Composite Selection Score
- Detection: mAP50, mAP50-95, AP_TL50, AP_Arrow50, AP_small, AP_medium, mAP_state
- Relevance: AUPRC, F1, Precision, Recall
- Attributes: State Accuracy, State Macro F1, Round F1, Maneuver Macro F1
- Multi-task Validation Losses
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tlr_yolo_mtl.evaluation.evaluator import evaluate_validation_epoch
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig, attach_unified_relevance_head
from tlr_yolo_mtl.training.data import (
    CanonicalMultiTaskDataset,
    canonical_multitask_collate,
)
from tlr_yolo_mtl.training.engine import build_multitask_criterion


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    if not cfg:
        with open("configs/tlr_yolo_mtl_train.yaml", "r", encoding="utf-8") as f:
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


def run_benchmark(
    name: str,
    checkpoint_path: Path,
    device: torch.device,
    val_loader: DataLoader,
    cfg: Mapping[str, Any],
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.60,
    max_batches: int | None = None,
) -> dict[str, Any]:
    print(f"\n{'='*90}")
    print(f"BENCHMARK: {name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"NMS Config: Conf Threshold = {conf_threshold:.2f}, IoU Threshold = {iou_threshold:.2f}")
    print(f"{'='*90}")

    model, _, payload = load_model_from_checkpoint(checkpoint_path, device)
    criterion = build_multitask_criterion(model, cfg)

    t0 = time.time()
    results = evaluate_validation_epoch(
        model,
        val_loader,
        criterion=criterion,
        device=device,
        amp_enabled=True,
        max_batches=max_batches,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    eval_time = time.time() - t0
    num_samples = results["samples_evaluated"]
    fps = num_samples / max(1e-6, eval_time)

    det = results["detection"]
    rel = results["relevance"]
    att = results["attributes"]
    loss = results["mean_losses"]
    score = results["selection_score"]

    print(f"\n--- VALUTAZIONE COMPLETATA IN {eval_time:.2f}s ({fps:.1f} FPS su {num_samples} immagini) ---")
    print(f"★ COMPOSITE SELECTION SCORE: {score:.4f}")
    print("\n[1. METRICHE DETECTION (mAP con NMS)]")
    print(f"  • mAP@50 (Globale)          : {det['map50']:.4f} ({det['map50']:.1%})")
    print(f"  • mAP@50-95 (Globale)       : {det['map50_95']:.4f} ({det['map50_95']:.1%})")
    print(f"  • AP Traffic Lights @50     : {det['ap_tl_50']:.4f} ({det['ap_tl_50']:.1%})")
    print(f"  • AP Road Arrows @50        : {det['ap_arrow_50']:.4f} ({det['ap_arrow_50']:.1%})")
    print(f"  • AP Tiny Lights (Small)    : {det['ap_small']:.4f} ({det['ap_small']:.1%})")
    print(f"  • AP Medium Lights/Arrows   : {det['ap_medium']:.4f} ({det['ap_medium']:.1%})")
    print(f"  • mAP State Joint           : {det['map_state']:.4f} ({det['map_state']:.1%})")

    print("\n[2. METRICHE RELEVANCE (Attenzione Multi-Task & Corsia Ego)]")
    print(f"  • Relevance AUPRC           : {rel['auprc']:.4f}")
    print(f"  • Relevance F1-Score        : {rel['f1']:.4f}")
    print(f"  • Relevance Precision       : {rel['precision']:.4f} ({rel['precision']:.1%})")
    print(f"  • Relevance Recall          : {rel['recall']:.4f} ({rel['recall']:.1%})")

    print("\n[3. METRICHE ATTRIBUTI E CLASSIFICAZIONE]")
    print(f"  • State Accuracy (4-class)  : {att['state_accuracy']:.4f} ({att['state_accuracy']:.1%})")
    print(f"  • State Macro F1            : {att['state_macro_f1']:.4f}")
    print(f"  • Round Signal F1           : {att['round_f1']:.4f}")
    print(f"  • Maneuver Macro F1         : {att['maneuver_macro_f1']:.4f}")

    print("\n[4. LOSSES MEDIE DI VALIDAZIONE]")
    print(f"  • Total Loss                : {loss['total']:.4f}")
    print(f"  • Detection Loss            : {loss['detection']:.4f}")
    print(f"  • State Loss                : {loss['state']:.4f}")
    print(f"  • Round Loss                : {loss['round']:.4f}")
    print(f"  • Maneuver Loss             : {loss['maneuver']:.4f}")
    print(f"  • Relevance Loss            : {loss['relevance']:.4f}")
    print(f"  • NWD Loss                  : {loss['nwd']:.4f}")

    results["eval_time"] = eval_time
    results["fps"] = fps
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizializzazione Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    config_path = Path("configs/tlr_yolo_mtl_train.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build full validation dataset
    val_dataset = CanonicalMultiTaskDataset(
        cfg["records"],
        split="val",
        target_size=tuple(cfg["input_size"]),
        training=False,
        seed=int(cfg["seed"]),
        allowed_sources=tuple(cfg.get("training_sources", ("DTLD",))),
        require_paired=bool(cfg.get("require_paired", True)),
    )
    print(f"Dataset di validazione completo: {len(val_dataset)} immagini accoppiate DTLD.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=canonical_multitask_collate,
    )

    best_ckpt = Path("runs/tlr_yolo_mtl_single_phase_seed42/weights/best.pt")
    ep50_ckpt = Path("runs/tlr_yolo_mtl_single_phase_seed42/weights/epoch_050.pt")

    summary_results = {}

    # 1. Evaluate BEST Checkpoint (Epoch 39) with standard NMS (Conf=0.05, IoU=0.60)
    if best_ckpt.exists():
        res_best = run_benchmark(
            name="BEST CHECKPOINT (Epoch 39) - Standard NMS (Conf=0.05, IoU=0.60)",
            checkpoint_path=best_ckpt,
            device=device,
            val_loader=val_loader,
            cfg=cfg,
            conf_threshold=0.05,
            iou_threshold=0.60,
        )
        summary_results["best_standard_nms"] = res_best

        # Also evaluate with strict deployment NMS (Conf=0.20, IoU=0.45)
        res_best_strict = run_benchmark(
            name="BEST CHECKPOINT (Epoch 39) - Strict Deployment NMS (Conf=0.20, IoU=0.45)",
            checkpoint_path=best_ckpt,
            device=device,
            val_loader=val_loader,
            cfg=cfg,
            conf_threshold=0.20,
            iou_threshold=0.45,
        )
        summary_results["best_strict_nms"] = res_best_strict

    # 2. Evaluate EPOCH 50 Checkpoint with standard NMS (Conf=0.05, IoU=0.60)
    if ep50_ckpt.exists():
        res_ep50 = run_benchmark(
            name="EPOCH 50 CHECKPOINT - Standard NMS (Conf=0.05, IoU=0.60)",
            checkpoint_path=ep50_ckpt,
            device=device,
            val_loader=val_loader,
            cfg=cfg,
            conf_threshold=0.05,
            iou_threshold=0.60,
        )
        summary_results["ep50_standard_nms"] = res_ep50

    # Save complete JSON report
    out_dir = Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "final_real_metrics_nms_report.json"
    
    # Convert numpy / torch values for clean JSON serialization
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

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(summary_results), f, indent=2)

    print(f"\n{'='*90}")
    print(f"Rapporto completo salvato in: {report_file}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
