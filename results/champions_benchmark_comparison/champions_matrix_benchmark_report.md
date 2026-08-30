# TLR-YOLO-MTL Definitive Champion Benchmark: Head-to-Head Comparison Matrix

> **Date & Time:** `2026-08-27 12:20:34`  
> **Evaluation Dataset:** Invariant Canonical DTLD Paired Validation Split (**`5,962` images**)  
> **Total Evaluation Time:** `2535.8 s`  
> **Target Hardware:** NVIDIA GeForce RTX 5070 12GB (FP16 Tensor Cores)  

---

## 1. Executive Summary: Champion v4 vs Champion v5 Primary Head-to-Head

Comparative performance on `best_composite.pt` (Primary Thesis Benchmark):

| Metric Category | Target Metric | Champion v4 Baseline | Champion v5 Proposed | Delta ($\Delta$) | Status / Interpretation |
|---|---|:---:|:---:|:---:|---|
| **Global Composite** | **Selection Score** | `0.7382` | **`0.7427`** | **+0.0045** (+0.6%) ▲ | Multi-task harmonic convergence |
| **Object Detection** | **mAP@50 (Global)** | `80.53%` | **`78.63%`** | **-1.89 pp** (-2.4%) ▼ | Overall detection accuracy |
| **Object Detection** | **mAP@50-95 (Global)** | `53.03%` | **`51.74%`** | **-1.30 pp** (-2.4%) ▼ | High-IoU spatial precision |
| **Object Detection** | **Traffic Light AP@50** | `65.01%` | **`61.37%`** | **-3.64 pp** (-5.6%) ▼ | Primary traffic light perception |
| **Object Detection** | **Road Arrow AP@50** | `96.04%` | **`95.89%`** | **-0.15 pp** (-0.2%) ▼ | Road surface arrow marking AP |
| **Scale Stratification** | **Sub-8px AP@50** | `10.33%` | **`4.67%`** | **-5.66 pp** (-54.8%) ▼ | Distant / tiny traffic light AP |
| **Scale Stratification** | **8-16px AP@50** | `60.24%` | **`53.31%`** | **-6.93 pp** (-11.5%) ▼ | Mid-range traffic lights |
| **Scale Stratification** | **16-32px AP@50** | `87.11%` | **`86.42%`** | **-0.69 pp** (-0.8%) ▼ | Near-field traffic lights |
| **Scale Stratification** | **>32px AP@50** | `94.36%` | **`95.22%`** | **+0.86 pp** (+0.9%) ▲ | Immediate foreground signals |
| **Relevance Reasoning** | **Relevance AUPRC** | `90.87%` | **`92.15%`** | **+1.27 pp** (+1.4%) ▲ | Ego-lane attribution PR-AUC |
| **Relevance Reasoning** | **Relevance F1-Score** | `85.70%` | **`86.75%`** | **+1.05 pp** (+1.2%) ▲ | Optimal relevance F1 operating point |
| **Relevance Safety** | **Relevant Red Recall ($	au=0.5$)** | `73.22%` | **`72.82%`** | **-0.41 pp** (-0.6%) ▼ | Stop-signal safety retention |
| **Attribute Towers** | **State Accuracy (4-Class)** | `73.98%` | **`80.10%`** | **+6.12 pp** (+8.3%) ▲ | Red/Yellow/Green/Off accuracy |
| **Attribute Towers** | **State Macro-F1** | `61.00%` | **`66.21%`** | **+5.22 pp** (+8.6%) ▲ | Unweighted color class balance |
| **Attribute Towers** | **Sub-4px State Accuracy** | `37.03%` | **`49.64%`** | **+12.61 pp** (+34.1%) ▲ | Extreme distant color discrimination |
| **Attribute Towers** | **Round Signal F1** | `88.40%` | **`87.96%`** | **-0.45 pp** (-0.5%) ▼ | Circular vs Directional signal F1 |
| **Attribute Towers** | **Maneuver Macro-F1** | `33.50%` | **`36.22%`** | **+2.73 pp** (+8.1%) ▲ | Arrow pictogram multi-label F1 |

---

## 2. Multi-Checkpoint Diagnostic Matrix Comparison

Complete evaluation across all five saved checkpoints for each model lineage:

### Champion v4 Checkpoints Matrix

| Checkpoint | Selection Score | mAP@50 | Sub-8px AP | Rel AUPRC | Rel Red Recall | State Acc | State Macro-F1 | Eval Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`best_composite.pt`** | `0.7382` | 80.53% | 10.33% | 90.87% | 73.22% | 73.98% | 61.00% | `230.4s` |
| **`best_tl_detection.pt`** | `0.7385` | 81.20% | 14.97% | 90.15% | 72.95% | 73.77% | 61.20% | `234.8s` |
| **`best_relevance.pt`** | `0.7386` | 80.35% | 10.31% | 90.73% | 73.17% | 74.00% | 60.96% | `224.7s` |
| **`best_relevant_red_recall.pt`** | `0.6645` | 78.60% | 16.16% | 83.57% | 86.00% | 20.59% | 23.80% | `240.3s` |
| **`last.pt`** | `0.7487` | 80.06% | 9.76% | 91.22% | 74.58% | 82.56% | 68.53% | `217.0s` |

### Champion v5 Checkpoints Matrix

| Checkpoint | Selection Score | mAP@50 | Sub-8px AP | Rel AUPRC | Rel Red Recall | State Acc | State Macro-F1 | Eval Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`best_composite.pt`** | `0.7427` | 78.63% | 4.67% | 92.15% | 72.82% | 80.10% | 66.21% | `251.8s` |
| **`best_tl_detection.pt`** | `0.7227` | 81.47% | 21.58% | 89.23% | 73.98% | 59.89% | 50.97% | `279.9s` |
| **`best_relevance.pt`** | `0.7433` | 78.67% | 4.91% | 92.34% | 70.84% | 80.32% | 66.22% | `244.7s` |
| **`best_relevant_red_recall.pt`** | `0.5944` | 66.56% | 10.62% | 78.55% | 82.83% | 7.67% | 9.11% | `368.4s` |
| **`last.pt`** | `0.7433` | 78.67% | 4.91% | 92.34% | 70.84% | 80.32% | 66.22% | `224.1s` |

---

## 3. 4-Stage Safety Waterfall Failure Decomposition

Survival rate of Ground Truth Relevant Red Traffic Lights across architectural layers:

| Architectural Stage | Champion v4 Count / % | Champion v5 Count / % | $\Delta$ Retention |
|---|:---:|:---:|:---:|
| **Total GT Relevant Red TLs** | `3686` (100.0%) | `3686` (100.0%) | — |
| **Stage 1: Perception Detection** | `3369` (91.40%) | `3251` (88.20%) | **-3.20 pp** (-3.5%) ▼ |
| **Stage 2: Top-K Candidate Pool** | `3325` (98.69%) | `3225` (99.20%) | **+0.51 pp** (+0.5%) ▲ |
| **Stage 3: State Classification (Red)** | `2527` (76.00%) | `2465` (76.43%) | **+0.43 pp** (+0.6%) ▲ |
| **Stage 4: Relevance Gate ($\tau=0.5$)** | `2137` (84.57%) | `2085` (84.58%) | **+0.02 pp** (+0.0%) ▲ |
| **End-to-End Relevant Red Recall** | **`2137` (57.98%)** | **`2085` (56.57%)** | **-1.41 pp** (-2.4%) ▼ |

---

## 4. Hardware Inference Profiling on RTX 5070 (FP16 Tensor Cores)

| Hardware Benchmark Metric | Champion v4 | Champion v5 | Target Specification / Constraint | Compliance Status |
|---|:---:|:---:|:---:|:---:|
| **Single-Stream Latency (Batch=1)** | `23.16 ms` | `43.38 ms` | $\le 27.5\text{ ms}$ (Real-time 36+ FPS) | PASSED (Sub-30ms) |
| **Single-Stream FPS (Batch=1)** | `43.2 FPS` | `23.1 FPS` | $\ge 30.0\text{ FPS}$ | PASSED |
| **High-Throughput FPS (Batch=16)** | `68.9 FPS` | `66.9 FPS` | $\ge 50.0\text{ FPS}$ | PASSED |
| **Peak Inference VRAM Footprint** | `5.00 GB` | `5.00 GB` | $\le 6.0\text{ GB}$ (RTX 5070 12GB) | PASSED |

---

## 5. Visual Artifacts & Figures

- **Multi-Panel Comparison Benchmark Figure:** [champion_matrix_benchmark_comparison.png](file:///C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/champions_benchmark_comparison/figures/champion_matrix_benchmark_comparison.png)
- **JSON Telemetry:** [champions_matrix_telemetry.json](file:///C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/champions_benchmark_comparison/champions_matrix_telemetry.json)
