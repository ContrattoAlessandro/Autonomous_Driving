---
title: "W1: Immutable Baseline B0 & Training Telemetry Contract"
type: task
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

How can we establish an immutable, reproducible single-phase Baseline B0 run and telemetry suite that captures all unweighted/weighted loss curves, optimizer step telemetry, sampling distributions, gradient norms, AMP scaler events, and multi-criterion checkpoints before testing structural modifications?

## Context & Requirements

1. **Single-Phase Setup**:
   - `joint_training_single_phase` (130 epochs @ 100 optimizer steps/epoch, micro_batch 8, grad_accum 4 $\to$ effective batch 32).
   - Warm-start weights (`yolov8s.pt`), Cosine LR (`backbone_lr: 1e-4`, `head_lr: 1e-3`), EMA (0.9999).
   - Relevance-perception gradient warmup scale ($0.0 \to 1.0$).

2. **Telemetry Captured**:
   - Complete configuration and seed tracking (`seed: 42`).
   - Optimizer step telemetry tracking 13,000 steps with unweighted loss decomposition ($\mathcal{L}_{det}, \mathcal{L}_{state}, \mathcal{L}_{round}, \mathcal{L}_{man}, \mathcal{L}_{rel}, \mathcal{L}_{nwd}$).
   - Module-wise Frobenius gradient norms (`compute_module_gradient_norms`: Backbone, Neck, Detect, Attributes, Cross-Attention, Relevance).
   - AMP `GradScaler` events, step overflow count, and gradient clipping trigger rate.
   - Validation evaluation telemetry with task-specific metrics and Relevant Red TL Recall.

3. **Multi-Checkpoint Saving (Pareto Selection)**:
   - `best.pt` / `best_composite.pt`: Highest validation composite score (Score = 0.7192 at Epoch 39).
   - `best_tl_detection.pt`: Highest $AP_{TL}$ ($AP_{TL,50} = 0.5497$, $mAP_{50} = 0.7261$).
   - `best_relevance.pt`: Highest $AUPRC_{rel}$ ($AUPRC = 0.9663$, $F1 = 0.8994$).
   - `best_relevant_red_recall.pt`: Highest Relevant Red TL Recall.
   - `last.pt`: Final step checkpoint.

## Empirical Resolution & Telemetry Summary

- **Run Directory**: `runs/tlr_yolo_mtl_single_phase_seed42/`
- **Peak Selection Score**: `0.7192` (Epoch 39)
- **Peak Detection $mAP_{50}$**: `0.7261` (Epoch 30), $AP_{TL,50} = 0.5497$
- **Peak Relevance $AUPRC$**: `0.9663` (Epoch 35), $F1 = 0.8994$
- **Peak State Accuracy**: `0.9331` (Epoch 38), Macro $F1 = 0.8760$
- **Telemetry Contract**: Fully integrated into `tlr_yolo_mtl/training/engine.py` and `tlr_yolo_mtl/evaluation/evaluator.py`. Multi-checkpoint Pareto savers validated.
