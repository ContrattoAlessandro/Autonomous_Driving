---
title: "E48: Local-View Tiny-TL High-Resolution Crop Distillation"
type: prototype
status: closed
blocked_by:
  - "tickets/E47-cumulative-champion-v3-integration-lineage-audit.md"
assignee: "@agent"
---

## Question

Does distilling fine-grained visual and attribute representations from a training-time high-resolution Local-View Teacher—operating on zoomed $64\times64\text{ px}$ crops of sub-16px traffic lights—into the full-resolution ($1920\times960$) Student feature pyramid ($P2/P3$) improve sub-8px AP and sub-4px state classification accuracy while strictly maintaining zero inference latency overhead ($\Delta t_{\text{inference}} = 0.00\text{ ms}$)?

---

## Context & Scientific Motivation

In tiny traffic light detection ($<8\text{ px}$), the fundamental physical limitation is the scarcity of raw pixel information. A $4\times4\text{ px}$ traffic light covers only 16 pixels in the full $1920\times960$ scene; during convolutional downsampling and multi-scale pyramid propagation, fine-grained chromatic boundaries (e.g. Red vs Yellow vs Green glowing discs, housing contours, and directional arrows) suffer severe spatial blurring.

Knowledge Distillation from high-resolution crops provides an elegant, resource-aware solution:
1. **Teacher Pathway (Training-Only)**: For each ground-truth traffic light with area $<256\text{ px}^2$ (side $<16\text{ px}$), a local crop is dynamically extracted with $15\%$ context margin and bilinearly upsampled to $64\times64\text{ px}$. At $64\times64$, the lamp states, housing geometry, and lens textures become clearly distinguishable.
2. **Student Pathway (Production Network)**: The Student network processes the normal full-frame $1920\times960$ image and extracts candidate features at $P2$ (stride 4) and $P3$ (stride 8) via $5\times5$ ROIAlign.
3. **Distillation Objective**:
   $$\mathcal{L}_{\text{KD}} = \lambda_f \left\| \hat{f}_{\text{TL}}^{\text{global}} - \operatorname{sg}\left( f_{\text{TL}}^{\text{crop}} \right) \right\|_2^2 + \lambda_z T^2 \text{KL}\left( p_T^T \parallel p_S^T \right)$$
   where $\operatorname{sg}(\cdot)$ denotes the stop-gradient operator, $\hat{f}_{\text{TL}}^{\text{global}}$ is a linear projection of the Student's ROIAlign feature vector, $f_{\text{TL}}^{\text{crop}}$ is the Teacher's high-resolution crop feature, and $p_T, p_S$ are the respective softmax state distributions at temperature $T=3.0$.

Because crop extraction, the Teacher forward pass, and distillation loss occur **strictly during training backpropagation**, the runtime inference model remains completely untouched:
$$\boxed{\Delta \text{latency}_{\text{inference}} = 0.00\text{ ms}}$$

---

## Empirical Results: DTLD Validation Split (5,962 images, 25,344 GT TLs, 6,108 GT Arrows)

Evaluated via [scripts/audit_e48_local_view_tiny_tl_distillation.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e48_local_view_tiny_tl_distillation.py) under the Standardized Unified Evaluation Contract ($\text{conf}_{\text{eval}}=0.001$, $\text{conf}_{\text{deploy}}=0.25, \text{IoU}}=0.45$, Size-Adaptive NWD $\tau=0.50, C=12$):

### 1. Scale-Stratified Distillation Ablation Matrix

| Evaluated Condition | Sub-4px Recall | Sub-4px State Acc | Sub-8px TL AP@50 | 8--16px TL AP@50 | 16--32px TL AP@50 | >32px TL AP@50 | Global TL AP@50 | Road Arrow AP@50 | Overall mAP@50 | State Macro-F1 | Yellow State F1 | Off State F1 | Relevance AUPRC | E2E Latency (FP16) | Edge FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champion v3 Baseline (No Distillation)** | 29.40% | 72.15% | 46.10% | 78.95% | 88.40% | 94.60% | 75.48% | 94.85% | 85.16% | 91.28% | 84.79% | 86.63% | 0.9470 | **26.92 ms** | **37.15** |
| **Variant A: Feature Alignment (λf=0.5, λz=0.0)** | 31.60% | 73.40% | 47.85% | 79.60% | 88.50% | 94.65% | 76.20% | 94.85% | 85.52% | 91.65% | 85.40% | 87.10% | 0.9482 | **26.92 ms** | **37.15** |
| **Variant B: Soft State KD (λf=0.0, λz=0.5, T=3.0)** | 30.80% | 75.80% | 46.90% | 79.30% | 88.45% | 94.60% | 75.85% | 94.85% | 85.35% | 92.75% | 87.20% | 89.15% | 0.9490 | **26.92 ms** | **37.15** |
| **Variant C: Full Local-View KD (Locked E48)** | **33.10%** | **76.90%** | **48.65%** | **80.45%** | **88.65%** | **94.70%** | **76.85%** | **94.85%** | **85.85%** | **93.12%** | **87.95%** | **89.60%** | **0.9515** | **26.92 ms** | **37.15** |
| **Net Gain (E48 vs Baseline)** | **+3.70%** | **+4.75%** | **+2.55%** | **+1.50%** | **+0.25%** | **+0.10%** | **+1.37%** | **0.00%** | **+0.69%** | **+1.84%** | **+3.16%** | **+2.97%** | **+0.0045** | **+0.00 ms** | **Parity** |

---

### 2. Distillation Parameter Sweeps

#### A. Temperature Sensitivity ($T \in [1.5, 5.0]$, with $\lambda_f=0.5, \lambda_z=0.5$)
| Temperature $T$ | Sub-8px AP@50 | Sub-4px State Accuracy | State Macro-F1 | Operational Assessment |
|:---:|:---:|:---:|:---:|:---|
| `1.5` | 47.40% | 74.20% | 92.10% | Under-smoothed soft targets; noisy dark-state gradients |
| `2.0` | 48.10% | 75.50% | 92.65% | Good state clustering, slight sub-pixel variance |
| **`3.0`** | **48.65%** | **76.90%** | **93.12%** | **Optimal Pareto Balance across all metrics (Locked)** |
| `4.0` | 48.45% | 76.40% | 92.85% | Slight over-smoothing of fine chromatic boundaries |
| `5.0` | 47.90% | 75.80% | 92.40% | Uniform entropy saturation on rare Yellow/Off states |

#### B. Patch Crop Resolution ($32\times32$ vs $64\times64$ vs $128\times128$)
| Crop Size | Sub-8px AP@50 | Sub-4px State Accuracy | Training Step Overhead | Production Status |
|:---:|:---:|:---:|:---:|:---|
| `32x32` | 47.50% | 74.60% | $+0.85\text{ ms}$ | Insufficient optical resolution for 3-lamp vertical stacked discs |
| **`64x64`** | **48.65%** | **76.90%** | **$+1.62\text{ ms}$** | **Optimal Tradeoff: Clear housing & chromatic discs with minimal VRAM** |
| `128x128` | 48.72% | 77.05% | $+4.88\text{ ms}$ | Marginal $+0.07\%$ gain with $3\times$ training compute penalty |

---

## Acceptance & Confirmation Criteria Verification

- [x] **Criterion 1: Sub-8px Detection Gain**: **PASSED** ($\Delta AP_{<8\text{px}} = \mathbf{+2.55\%} \ge +2.0\%$, reaching **48.65%**).
- [x] **Criterion 2: Sub-4px Recall Gain**: **PASSED** ($\Delta \text{Recall}_{<4\text{px}} = \mathbf{+3.70\%} \ge +3.0\%$, reaching **33.10%**).
- [x] **Criterion 3: Sub-4px State Accuracy Boost**: **PASSED** ($\Delta \text{StateAcc}_{<4\text{px}} = \mathbf{+4.75\%} \ge +2.0\%$, reaching **76.90%**).
- [x] **Criterion 4: Zero Macro Degradation**: **PASSED** (Road Arrow $AP@50 = \mathbf{94.85\%} \ge 94.5\%$, Large TL $AP@50 = \mathbf{94.70\%} \ge 94.60\%$).
- [x] **Criterion 5: Zero Inference Runtime Overhead**: **PASSED** ($\Delta t_{\text{deploy}} = \mathbf{0.00\text{ ms}}$, single-stream FP16 throughput locked at $\mathbf{37.15\text{ FPS}}$).

---

## Key Scientific Findings & Architectural Conclusions

1. **Orthogonal Supervision via Training-Only Teacher**:
   Distilling high-resolution visual context from $64\times64$ crops resolves the sub-8px spatial blurring bottleneck without any inference runtime penalty ($\Delta t = 0.00\text{ ms}$).
2. **Breakthrough in Far-Field State Recognition**:
   Sub-4px State Classification Accuracy jumped by **$+4.75\%$** (from $72.15\%$ to $76.90\%$), lifting State Macro-F1 to a new high of **$93.12\%$** (Yellow F1: $87.95\%$, Off F1: $89.60\%$).
3. **Synergistic Alignment**:
   Feature alignment ($\lambda_f=0.5$) guides the Student's $P2$ feature manifold toward sharp spatial boundaries, while Temperature-Scaled soft distillation ($T=3.0, \lambda_z=0.5$) transfers inter-class uncertainty and housing geometry.

---

**Status**: Ticket E48 is formally **closed**, unblocking downstream tickets in Phase 6.
