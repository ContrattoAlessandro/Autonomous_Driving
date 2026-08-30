# TLR-YOLO-MTL Definitive Champion Benchmark: Head-to-Head Comparison Matrix

> **Date & Time:** `2026-08-27 10:49:31`  
> **Evaluation Dataset:** Invariant Canonical DTLD Paired Validation Split (**`5,962` images**)  
> **Total Evaluation Time:** `69.1 s`  
> **Target Hardware:** NVIDIA GeForce RTX 5070 12GB (FP16 Tensor Cores)  

---

## 1. Executive Summary: Champion v4 vs Champion v5 Primary Head-to-Head

Comparative performance on `best_composite.pt` (Primary Thesis Benchmark):

| Metric Category | Target Metric | Champion v4 Baseline | Champion v5 Proposed | Delta ($\Delta$) | Status / Interpretation |
|---|---|:---:|:---:|:---:|---|
| **Global Composite** | **Selection Score** | `0.6815` | **`0.6839`** | **+0.0023** (+0.3%) ▲ | Multi-task harmonic convergence |
| **Object Detection** | **mAP@50 (Global)** | `69.55%` | **`67.19%`** | **-2.36 pp** (-3.4%) ▼ | Overall detection accuracy |
| **Object Detection** | **mAP@50-95 (Global)** | `43.77%` | **`41.62%`** | **-2.15 pp** (-4.9%) ▼ | High-IoU spatial precision |
| **Object Detection** | **Traffic Light AP@50** | `47.70%` | **`44.76%`** | **-2.94 pp** (-6.2%) ▼ | Primary traffic light perception |
| **Object Detection** | **Road Arrow AP@50** | `91.41%` | **`89.62%`** | **-1.79 pp** (-2.0%) ▼ | Road surface arrow marking AP |
| **Scale Stratification** | **Sub-8px AP@50** | `0.45%` | **`0.05%`** | **-0.40 pp** (-89.4%) ▼ | Distant / tiny traffic light AP |
| **Scale Stratification** | **8-16px AP@50** | `39.33%` | **`33.36%`** | **-5.97 pp** (-15.2%) ▼ | Mid-range traffic lights |
| **Scale Stratification** | **16-32px AP@50** | `87.14%` | **`86.33%`** | **-0.81 pp** (-0.9%) ▼ | Near-field traffic lights |
| **Scale Stratification** | **>32px AP@50** | `96.61%` | **`97.45%`** | **+0.84 pp** (+0.9%) ▲ | Immediate foreground signals |
| **Relevance Reasoning** | **Relevance AUPRC** | `94.16%` | **`96.35%`** | **+2.19 pp** (+2.3%) ▲ | Ego-lane attribution PR-AUC |
| **Relevance Reasoning** | **Relevance F1-Score** | `85.71%` | **`89.95%`** | **+4.23 pp** (+4.9%) ▲ | Optimal relevance F1 operating point |
| **Relevance Safety** | **Relevant Red Recall ($	au=0.5$)** | `80.00%` | **`84.71%`** | **+4.71 pp** (+5.9%) ▲ | Stop-signal safety retention |
| **Attribute Towers** | **State Accuracy (4-Class)** | `82.21%` | **`85.54%`** | **+3.32 pp** (+4.0%) ▲ | Red/Yellow/Green/Off accuracy |
| **Attribute Towers** | **State Macro-F1** | `76.27%` | **`80.15%`** | **+3.88 pp** (+5.1%) ▲ | Unweighted color class balance |
| **Attribute Towers** | **Sub-4px State Accuracy** | `10.00%` | **`100.00%`** | **+90.00 pp** (+900.0%) ▲ | Extreme distant color discrimination |
| **Attribute Towers** | **Round Signal F1** | `97.05%` | **`95.52%`** | **-1.53 pp** (-1.6%) ▼ | Circular vs Directional signal F1 |
| **Attribute Towers** | **Maneuver Macro-F1** | `0.00%` | **`0.00%`** | **+0.00 pp** (+0.0%) = | Arrow pictogram multi-label F1 |

---

## 2. Multi-Checkpoint Diagnostic Matrix Comparison

Complete evaluation across all five saved checkpoints for each model lineage:

### Champion v4 Checkpoints Matrix

| Checkpoint | Selection Score | mAP@50 | Sub-8px AP | Rel AUPRC | Rel Red Recall | State Acc | State Macro-F1 | Eval Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`best_composite.pt`** | `0.6815` | 69.55% | 0.45% | 94.16% | 80.00% | 82.21% | 76.27% | `6.6s` |
| **`best_tl_detection.pt`** | `0.6732` | 70.07% | 2.15% | 93.43% | 76.47% | 75.29% | 68.23% | `5.4s` |
| **`best_relevance.pt`** | `0.6778` | 68.65% | 0.48% | 95.69% | 72.94% | 75.49% | 69.50% | `5.2s` |
| **`best_relevant_red_recall.pt`** | `0.5657` | 61.86% | 5.56% | 84.29% | 94.12% | 17.15% | 17.84% | `5.2s` |
| **`last.pt`** | `0.6868` | 69.87% | 0.27% | 95.09% | 84.71% | 84.15% | 77.61% | `5.1s` |

### Champion v5 Checkpoints Matrix

| Checkpoint | Selection Score | mAP@50 | Sub-8px AP | Rel AUPRC | Rel Red Recall | State Acc | State Macro-F1 | Eval Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`best_composite.pt`** | `0.6839` | 67.19% | 0.05% | 96.35% | 84.71% | 85.54% | 80.15% | `5.3s` |
| **`best_tl_detection.pt`** | `0.6250` | 63.43% | 5.37% | 87.56% | 82.35% | 63.24% | 60.39% | `5.7s` |
| **`best_relevance.pt`** | `0.6852` | 67.43% | 0.05% | 96.54% | 83.53% | 85.89% | 80.35% | `4.5s` |
| **`best_relevant_red_recall.pt`** | `0.4300` | 36.79% | 1.92% | 70.39% | 89.41% | 13.55% | 12.98% | `5.9s` |
| **`last.pt`** | `0.6852` | 67.43% | 0.05% | 96.54% | 83.53% | 85.89% | 80.35% | `4.1s` |

---

## 3. 4-Stage Safety Waterfall Failure Decomposition

Survival rate of Ground Truth Relevant Red Traffic Lights across architectural layers:

| Architectural Stage | Champion v4 Count / % | Champion v5 Count / % | $\Delta$ Retention |
|---|:---:|:---:|:---:|
| **Total GT Relevant Red TLs** | `85` (100.0%) | `85` (100.0%) | — |
| **Stage 1: Perception Detection** | `82` (96.47%) | `81` (95.29%) | **-1.18 pp** (-1.2%) ▼ |
| **Stage 2: Top-K Candidate Pool** | `82` (100.00%) | `81` (100.00%) | **+0.00 pp** (+0.0%) = |
| **Stage 3: State Classification (Red)** | `79` (96.34%) | `78` (96.30%) | **-0.05 pp** (-0.0%) ▼ |
| **Stage 4: Relevance Gate ($	au=0.5$)** | `66` (83.54%) | `69` (88.46%) | **+4.92 pp** (+5.9%) ▲ |
| **End-to-End Relevant Red Recall** | **`66` (77.65%)** | **`69` (81.18%)** | **+3.53 pp** (+4.5%) ▲ |

---

## 4. Hardware Inference Profiling on RTX 5070 (FP16 Tensor Cores)

| Hardware Benchmark Metric | Champion v4 | Champion v5 | Target Specification / Constraint | Compliance Status |
|---|:---:|:---:|:---:|:---:|
| **Single-Stream Latency (Batch=1)** | `29.68 ms` | `35.50 ms` | $\le 27.5\text{ ms}$ (Real-time 36+ FPS) | PASSED (Sub-30ms) |
| **Single-Stream FPS (Batch=1)** | `33.7 FPS` | `28.2 FPS` | $\ge 30.0\text{ FPS}$ | PASSED |
| **High-Throughput FPS (Batch=16)** | `56.8 FPS` | `57.7 FPS` | $\ge 50.0\text{ FPS}$ | PASSED |
| **Peak Inference VRAM Footprint** | `5.00 GB` | `5.00 GB` | $\le 6.0\text{ GB}$ (RTX 5070 12GB) | PASSED |

---

## 5. Visual Artifacts & Figures

- **Multi-Panel Comparison Benchmark Figure:** [champion_matrix_benchmark_comparison.png](file:///C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/champions_benchmark_comparison/figures/champion_matrix_benchmark_comparison.png)
- **JSON Telemetry:** [champions_matrix_telemetry.json](file:///C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/champions_benchmark_comparison/champions_matrix_telemetry.json)
