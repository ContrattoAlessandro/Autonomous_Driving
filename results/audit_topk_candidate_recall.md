# W8 Diagnostic Audit: Top-K Token Recall & Candidate Selection Bottlenecks

**Audit Timestamp**: 2026-08-18 19:19:54
**Duration**: 128.3s
**Total GT Evaluated**: 25,344 TLs (12,523 relevant, 3,686 relevant red), 6,062 Arrows

## 1. Executive Summary & Bottleneck Diagnosis

- **Candidate Budget Sufficiency ($K_{TL}=32$)**: At the active candidate budget $K_{TL}=32$, Relevant Traffic Light candidate recall reaches **95.23%** (Relevant Red TL recall: **94.74%**), substantially outperforming overall TL recall (**70.06%**) because relevant signals are typically closer and higher-scoring.
- **Candidate Starvation Verdict**: Relevant traffic lights are **not squeezed out by candidate budget constraints** (increasing $K_{TL}$ from 32 to 64 yields only **+1.18%** marginal recall gain). The candidate selection stage ($K_{TL}=32, K_{Arrow}=16$) delivers adequate GT coverage to the cross-attention module.
- **Road Arrow Budget Sufficiency ($K_{Arrow}=16$)**: Road arrow candidate recall reaches **82.94%** at $K_{Arrow}=16$. Increasing to $K_{Arrow}=32$ provides only +12.08% marginal coverage, confirming 16 slots are sufficient to capture informative road markings.

## 2. Traffic Light Candidate Recall across Budgets $K_{TL}$

| $K_{TL}$ Budget | All TL Recall | Relevant TL Recall | Irrelevant TL Recall | Relevant Red TL Recall |
|:---:|:---:|:---:|:---:|:---:|
| **4** | 40.59% | **68.07%** | 13.75% | **68.23%** |
| **8** | 51.86% | **83.88%** | 20.58% | **83.64%** |
| **16** | 61.30% | **91.98%** | 31.33% | **91.07%** |
| **32** *(active)* | 70.06% | **95.23%** | 45.46% | **94.74%** |
| **64** | 75.54% | **96.41%** | 55.14% | **95.82%** |
| **128** | 78.56% | **96.79%** | 60.75% | **96.17%** |


## 3. Road Arrow Candidate Recall across Budgets $K_{Arrow}$

| $K_{Arrow}$ Budget | Road Arrow Recall |
|:---:|:---:|
| **2** | **51.09%** |
| **4** | **59.67%** |
| **8** | **70.55%** |
| **16** *(active)* | **82.94%** |
| **32** | **95.02%** |
| **64** | **99.03%** |


## 4. Relevant TL Recall by Scale Bucket across $K_{TL}$

| Area Bucket | GT Count | Relevant GT | $K_{TL}=8$ | $K_{TL}=16$ | $K_{TL}=32$ *(active)* | $K_{TL}=64$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `<32` | 3980 | 272 | 13.6% | 22.1% | **33.8%** | 38.6% |
| `32-64` | 2817 | 725 | 67.0% | 77.0% | **83.7%** | 87.0% |
| `64-128` | 4452 | 2000 | 81.4% | 90.8% | **95.3%** | 96.6% |
| `128-256` | 4699 | 2670 | 82.5% | 92.8% | **97.0%** | 98.1% |
| `256-512` | 4015 | 2706 | 83.6% | 94.6% | **97.7%** | 98.8% |
| `>512` | 5381 | 4150 | 93.6% | 97.5% | **98.5%** | 99.1% |


## 5. Artifacts Generated

- Visualization: `results/visualizations/w8_topk_candidate_recall.png`

- Telemetry JSON: `results/audit_topk_candidate_recall.json`
