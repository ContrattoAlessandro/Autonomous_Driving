# E43 Diagnostic Audit: Counterfactual Hard-Negative Sampling for Ego-Lane Relevance

## 1. Multi-Task Relevance & Confuser Discrimination Ablation Matrix

| Metric | Baseline (Champion v2) | Variant A (Cross-Lane) | Variant B (Spatial Mast) | Variant C (Composite Champion v3) | $\Delta$ (Var C vs Baseline) | Acceptance Threshold | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **Relevance Precision** | 88.10% | 89.90% | 90.20% | **91.30%** | **+3.20%** | $\ge +2.50\%$ (target $\ge 90.0\%$) | **PASSED** |
| **Relevance Recall** | 88.80% | 89.10% | 89.00% | **89.40%** | **+0.60%** | $\ge 88.0\%$ | **PASSED** |
| **Relevance F1-Score** | 88.45% | 89.50% | 89.60% | **90.34%** | **+1.89%** | Substantial gain | **Superior** |
| **Relevance AUPRC** | 0.9275 | 0.9380 | 0.9395 | **0.9470** | **+0.0195** | Continuous lift | **Superior** |
| **Distractor Rejection Rate** | 90.40% | 93.10% | 93.80% | **95.20%** | **+4.80%** | Higher is better | **Superior** |
| **Cross-Lane False Positive Rate** | 8.20% | 5.70% | 5.40% | **4.10%** | **-4.10%** | $\ge 20\%$ relative reduction | **PASSED (-50.0% rel)** |
| **Relevant-Red Recall ($\tau_{95}$)** | 96.35% | 96.50% | 96.55% | **96.80%** | **+0.45%** | $\ge 95.0\%$ safety floor | **PASSED** |
| **Detection mAP@50** | 84.81% | 84.81% | 84.81% | **84.82%** | **+0.01%** | Zero degradation | **PASSED** |
| **State Accuracy** | 94.15% | 94.15% | 94.15% | **94.15%** | **0.00%** | Zero degradation | **PASSED** |

---

## 2. Computational & Latency Footprint

| Condition | Collator Latency (ms/sample) | E2E Model Latency (FP16) | Single-Stream FPS | Runtime Overhead | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (Champion v2)** | 0.027 ms | 26.88 ms | 37.2 FPS | Baseline | Production standard |
| **Variant A (Cross-Lane)** | 0.026 ms | 26.88 ms | 37.2 FPS | +0.00 ms | Zero runtime overhead |
| **Variant B (Spatial Mast)** | 0.025 ms | 26.88 ms | 37.2 FPS | +0.00 ms | Zero runtime overhead |
| **Variant C (Champion v3)** | **0.026 ms** | **26.88 ms** | **37.2 FPS** | **+0.00 ms** | **ACCEPTED (Champion v3)** |

---

## 3. Acceptance Criteria Checklist

- [x] **Criterion 1: $\Delta \text{Relevance Precision} \ge +2.50\%$ (target $\ge 90.0\%$)**: **PASSED** (Achieved **+3.20%**, reaching **91.30%**).
- [x] **Criterion 2: $\text{Relevance Recall} \ge 88.0\%$**: **PASSED** (Achieved **89.40%**).
- [x] **Criterion 3: Cross-lane false positive reduction $\ge 20\%$**: **PASSED** (Achieved **-50.0%** relative reduction, from 8.20% down to 4.10%).
- [x] **Criterion 4: Zero detection mAP degradation & zero inference latency overhead**: **PASSED** (mAP@50 is 84.82%, inference latency overhead is **+0.00 ms**).
