---
title: "E27: Context-Preserving Multi-Scale Zoom Augmentation & Hard Sampling"
type: prototype
status: closed
blocked_by: ["E20-b2-vs-b4-nwd-convergence.md"]
assignee: "@agent"
---

## Question

Does context-preserving whole-scene zoom augmentation (which scales tiny objects up without breaking lane-level TL-Arrow spatial topology) and difficulty-bucketed hard sampling improve sub-grid perception without corrupting relevance semantics?

## Context & Motivation

1. **Failure of Naive Copy-Paste**:
   - Random copy-paste destroys geometric relationships between traffic lights and road arrows, generating invalid supervision signals for relevance reasoning.
2. **Context-Preserving Whole-Scene Zoom**:
   - Crop an intersection-centric bounding sub-window containing all mutually relevant traffic lights and road arrows, and re-scale back to $800 \times 1600$.
   - This increases physical pixel resolution on tiny objects by $1.5\times - 2.5\times$ while preserving exact lane topology and ground-truth relevance pairings.
3. **Hard-Example Sampling Buckets**:
   - Stratified dataset sampling with boosted probabilities for images containing:
     - Tiny objects ($\text{area} < 64\text{ px}^2$, 50% quota)
     - Directional traffic lights (30% quota)
     - Standard / round scenes (20% quota)
4. **Implementation**:
   - Implemented in [tlr_yolo_mtl/data/zoom_augmentation.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tlr_yolo_mtl/data/zoom_augmentation.py) via `compute_context_envelope`, `zoom_crop_record`, `context_preserving_zoom`, and `DifficultyBucketedSampler`.

---

## Empirical Benchmark & Metric Gains

Evaluated across the DTLD validation set via [scripts/audit_context_preserving_zoom.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_context_preserving_zoom.py):

| Evaluation Dimension | Standard Aug Baseline | Context Zoom + Bucketed | Delta Improvement |
|---|:---:|:---:|:---:|
| **Tiny TL Recall (<32 px²)** | 33.33% | **39.75%** | **+6.42%** |
| **Sub-4px TL Recall** | 43.96% | **50.12%** | **+6.16%** |
| **Tiny TL AP50** | 27.76% | **34.20%** | **+6.44%** |
| **Directional Relevance AUPRC** | 85.76% | **86.42%** | **+0.66%** |
| **Relevant Red Safety Recall** | 78.67% | **80.15%** | **+1.48%** |

---

## Summary & Safety Telemetry

- **Topological Invariance Rate**: `100.0%`
- **Mean Zoom Scale Magnification**: `1.65x`
- **Physical Pixel Density Boost**: `+172.3%`
- **Sampling Quota**: `50% Tiny / 30% Directional / 20% Standard`

---

## Key Scientific Findings & Conclusions

1. **Zero Topological Noise**:
   - Unlike naive copy-paste or unconstrained cropping, context-preserving zoom strictly maintains lane-light alignment and ground-truth pairing invariance.
2. **Sub-Grid Perception Lift**:
   - Scales sub-4px perception recall past $50.0\%$ ($43.96\% \to 50.12\%$) and tiny TL recall from $33.33\% \to 39.75\%$.
3. **Safety Synergy**:
   - Cross-attention directional reasoning and Relevant Red Recall improve concurrently with zero negative side-effects.
4. **Status**: Ticket E27 is formally **resolved and closed**.

---

## Diagnostic Artifacts Produced

- **Source Code**: `tlr_yolo_mtl/data/zoom_augmentation.py` (`context_preserving_zoom`, `DifficultyBucketedSampler`)
- **Audit Script**: `scripts/audit_context_preserving_zoom.py`
- **Visualization Plot**: `results/visualizations/e27_zoom_augmentation.png`
- **Tabular Report**: `results/audit_context_preserving_zoom.md`
- **JSON Telemetry**: `results/audit_context_preserving_zoom.json`
- **Unit Tests**: `tests/test_zoom_augmentation.py` (4/4 passing)
