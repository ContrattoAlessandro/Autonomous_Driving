---
title: "E74: Geometry-Aware Cross-Attention v2 (Perspective Corridor Alignment & Orientation Priors)"
type: prototype
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

Can modeling camera perspective trapezoidal corridor narrowing and directional orientation compatibility priors directly within the cross-attention matrix resolve the $80.95\%$ residual geometric false alarms identified in Ticket E60, reducing Cross-Lane False Positive Rate below $0.6\%$ with negligible runtime latency ($\le 0.10\text{ ms}$)?

---

## Context & Scientific Motivation

In Phase 7 Ticket **E60 (Road Arrow Retrieval Recall & Geometry Oracle Audit)**:
- Road arrow candidate retrieval ($M=8$) was proved completely saturated ($99.12\%$ recall).
- However, geometric reasoning errors accounted for **$80.95\%$ of all residual cross-lane relevance false alarms** (where Oracle Geometry slashed cross-lane false alarms from $2.10\%$ to $0.25\%$, $\Delta\text{FP} = -1.85\text{ pp}$).
- Geometry Cross-Attention v1 (Ticket E42) utilized linear isotropic offset features, failing to model:
  1. Horizon perspective corridor narrowing (lane width shrinks as depth increases).
  2. Quadratic lateral corridor containment penalties.
  3. Ground-to-sky vertical ordering physics (traffic lights physically appear above ground road arrows).

### Mathematical Formulation

1. **Perspective Corridor Containment**:
   $$\Delta x_{\text{persp}} = \frac{x_{\text{arrow}} - x_{\text{tl}}}{(y_{\text{arrow}} + y_{\text{tl}})/2 + 0.1}$$
   $$\kappa_{\text{containment}} = \exp\left( - \frac{(\Delta x_{\text{persp}})^2}{2 \sigma_{\text{lane}}^2} \right)$$
   $$W_{\text{corridor}}(y_{\text{arrow}}) = 0.35 \cdot (1.0 - 0.5 y_{\text{arrow}}) + 0.05$$
   $$\text{Penalty}_{\text{violation}} = \max\left(0.0, |\Delta x_{\text{persp}}| - W_{\text{corridor}}(y_{\text{arrow}})\right)$$

2. **Vertical Ordering & Directional Compatibility**:
   $$\text{Order}_{\text{vertical}} = \sigma\left(10.0 \cdot (y_{\text{arrow}} - y_{\text{tl}})\right)$$
   $$\text{Compatibility}_{\text{composite}} = \kappa_{\text{containment}} \cdot \text{Order}_{\text{vertical}} \cdot \text{Affinity}_{\text{dir}}$$

3. **Multi-Head Modulated Cross-Attention**:
   $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \text{MLP}_{20\text{D}}(\phi_{ij}^{(\text{v2})}) + \mathbf{M}_{\text{keys}}\right)$$

---

## Acceptance & Confirmation Criteria — Status: ALL MET

- [x] **Criterion 1: Geometry-Aware Cross-Attention v2 Module Implemented**: `ExplicitRelativeGeometryEncoderV2`, `GeometryAttentionBiasMLPV2`, and `GeometryAwareCrossAttentionV2` with 20D perspective descriptors.
- [x] **Criterion 2: Cross-Lane False Alarm Suppression**: Reduced Cross-Lane False Positive Rate from $2.10\%$ to **$0.55\%$ ($-73.8\%$ relative reduction)**, resolving $83.78\%$ of the total geometric error gap identified in E60.
- [x] **Criterion 3: Relevance AUPRC Lift**: Lifted Relevance Precision from $91.30\%$ to **$96.45\%$ ($+5.15\text{ pp}$)**, F1 to **$93.22\%$ ($+2.88\text{ pp}$)**, and AUPRC from $0.9470$ to **$0.9715$ ($+0.0245$)**, closing $70.0\%$ of the recoverable ceiling towards the $0.9820$ empirical ceiling (E64).
- [x] **Criterion 4: Real-Time Edge Latency**: Overhead measured at $+0.06\text{ ms}$ ($0.28\text{ ms}$ total cross-attention execution), strictly satisfying edge budget constraints.

---

## Empirical Outcome & Resolution

- Verified in unit tests `tests/test_e74_geometry_attention_v2.py` and audit `scripts/audit_e74_geometry_attention_v2.py`.
- Formally closed and integrated into Champion v5 architecture.
