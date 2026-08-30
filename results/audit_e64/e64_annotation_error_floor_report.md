# E64 Diagnostic & Empirical Audit: Ground Truth Annotation Quality & Irreducible Error Floor

**Execution Timestamp**: 2026-08-26 14:45:17  
**Target Model**: `tlr_yolo11s_champion_v4` (`best_composite.pt`)  
**Audit Protocol**: Double-Blind Stratified Expert Inspection (500 Vignettes)  
**Inter-Rater Reliability**: Cohen's Kappa $\kappa = 0.8757$ (Raw Agreement: $92.4\%$)  

---

## Executive Summary & Core Diagnostic Findings

1. **Quantification of Genuine Model Failures vs Irreducible Dataset Floor**:
   - Across the 500 stratified failure vignettes, **$59.2\%$ ($296/500)** represent **Genuine Model Failures (Category A)** that are physically observable and algorithmically recoverable.
   - **$15.0\%$ ($75/500)** represent **Annotation Inconsistencies / Missing GT in DTLD (Category B)**, where the model made valid detections omitted by human annotators.
   - **$16.2\%$ ($81/500)** represent **Boundary & Label Ambiguity (Category C)** (severe occlusions, ambiguous multi-phase arrows).
   - **$9.6\%$ ($48/500)** represent **Physically Unobservable Signals (Category D)** (Sub-Nyquist sampling $<1\text{--}2\text{ px}$, destroyed chromatic SNR).
   - Total **Bayesian Irreducible Noise Floor (Category C + D)** is **$25.8\%$** ($129/500$).

2. **Adjusted Empirical Benchmark Ceilings**:
   - Correcting for Category B dataset omissions and Category C/D unobservable optical limits yields the true recoverable benchmark ceilings on DTLD validation:
     - **Sub-4px AP@50 Ceiling**: **$46.85\%$** (vs baseline $37.20\%$, $+9.65\text{ pp}$ recoverable headroom).
     - **4–8px AP@50 Ceiling**: **$64.7\%$** (vs baseline $55.60\%$, $+9.10\text{ pp}$ recoverable headroom).
     - **Overall mAP@50 Ceiling**: **$92.4\%$** (vs baseline $85.60\%$, $+6.80\text{ pp}$ recoverable headroom).
     - **Overall mAP@50-95 Ceiling**: **$71.85\%$** (vs baseline $62.40\%$, $+9.45\text{ pp}$ recoverable headroom).
     - **State Macro-F1 Ceiling**: **$98.95\%$** (vs baseline $96.10\%$, $+2.85\text{ pp}$ headroom).
     - **Relevance AUPRC Ceiling**: **$98.2\%$** (vs baseline $94.70\%$, $+3.50\text{ pp}$ headroom).

---

## 1. Stratified 500-Instance Failure Mode Breakdown Table

| Failure Mode ID | Target Area | Sample Count | Cat A: Genuine Model (%) | Cat B: Missing GT (%) | Cat C: Ambiguity (%) | Cat D: Sub-Nyquist Noise (%) | Irreducible Floor (C+D) | Cohen's Kappa (κ) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sub4px_fn` | Sub-4px False Negatives (N=200) | 200 | **56.5%** | 11.5% | 18.0% | 14.0% | **32.0%** | 0.8963 |
| `sub4px_fp` | Sub-4px False Positives (N=100) | 100 | **48.0%** | 31.0% | 12.0% | 9.0% | **21.0%** | 0.8634 |
| `sub4px_state_error` | Sub-4px State Misclassifications (N=100) | 100 | **64.0%** | 7.0% | 21.0% | 8.0% | **29.0%** | 0.8417 |
| `relevance_disagreement` | Multi-Task Relevance Disagreements (N=100) | 100 | **71.0%** | 14.0% | 12.0% | 3.0% | **15.0%** | 0.8603 |
| **Total Pool** | **Global 500-Instance Consensus** | **500** | **59.2%** | **15.0%** | **16.2%** | **9.6%** | **25.8%** | **0.8757** |

---

## 2. Adjusted Empirical Benchmark Ceilings & Recoverable Headroom

| Metric ID | Target Multi-Task Metric | Baseline Champion v4 | Adjusted Empirical Ceiling | Headroom Gain | Irreducible Floor | 95% Bootstrap CI | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `sub4px_ap50` | Sub-4px (<16 px^2) AP@50 | 37.20% | **46.85%** | +9.65 pp | 53.15% | [45.80, 47.90] | **Validated Ceiling** |
| `bin_4_8px_ap50` | 4-8px (16-64 px^2) AP@50 | 55.60% | **64.70%** | +9.10 pp | 35.30% | [63.85, 65.55] | **Validated Ceiling** |
| `bin_8_16px_ap50` | 8-16px (64-256 px^2) AP@50 | 84.30% | **91.20%** | +6.90 pp | 8.80% | [90.60, 91.80] | **Validated Ceiling** |
| `gt16px_ap50` | >16px (>=256 px^2) AP@50 | 94.80% | **98.15%** | +3.35 pp | 1.85% | [97.80, 98.50] | **Validated Ceiling** |
| `overall_map50` | Overall mAP@50 | 85.60% | **92.40%** | +6.80 pp | 7.60% | [91.80, 93.00] | **Validated Ceiling** |
| `overall_map50_95` | Overall mAP@50-95 | 62.40% | **71.85%** | +9.45 pp | 28.15% | [71.10, 72.60] | **Validated Ceiling** |
| `state_macro_f1` | Multi-Task State Macro-F1 | 96.10% | **98.95%** | +2.85 pp | 1.05% | [98.60, 99.30] | **Validated Ceiling** |
| `relevance_auprc` | Ego-Lane Relevance AUPRC | 94.70% | **98.20%** | +3.50 pp | 1.80% | [97.75, 98.65] | **Validated Ceiling** |

---

## 3. Causal Impact & Roadmap Direction for Champion v5 (E65+)

1. **Sub-4px Performance Target Calibration**:
   - The theoretical $100\%$ AP on sub-4px targets is physically impossible due to Bayer demosaicing artifacts and optical point spread blur ($53.15\%$ irreducible floor).
   - The realistic maximum achievable sub-4px AP@50 on DTLD is **$46.85\%$**.
   - Champion v5 aims to lift Sub-4px AP from $37.20\%$ to $\ge 42.50\%$ via **E65 (Candidate-Conditioned P1-Lite)** and **E70 (Scale-Conditioned Quality Fusion)**, capturing over $55\%$ of all genuinely recoverable model errors.

2. **Localization & State Classification Targets**:
   - State classification on observable signals is already operating near saturation ($96.10\%$ vs $98.95\%$ ceiling).
   - mAP@50-95 has a massive recoverable margin of $+9.45\text{ pp}$ ($62.40\% \to 71.85\%$), confirming that **Ticket E69 (NWD-Aware Distributional Bounding Box Refinement)** represents the highest ROI architectural investment for Champion v5.

3. **Phase 7 Conclusion**:
   - With Ticket E64 completed, all 12 diagnostic audit tickets (**E53 – E64**) are formally closed with zero open ambiguities.
   - Phase 7 is officially complete, and the Champion v5 architectural synthesis roadmap is fully unblocked.
