---
title: "E66: Scale-Conditioned C2 -> P2 Feature Relay v2 with High-Frequency Saliency Gate"
type: prototype
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

Can a dual-gated C2 $\to$ P2 feature relay—combining a standard spatial-channel gate with a high-frequency tiny saliency branch—prevent sub-4px signal attenuation ($\bar{\alpha} \approx 0.38 \to \ge 0.70$) without increasing false positive activations on background clutter (foliage, reflections, poles)?

---

## Context & Scientific Motivation

In Phase 7 Ticket **E55 (Tiny Feature Survival & SNR Audit)**, intermediate probe analysis demonstrated:
- Raw $C2$ features preserve high linear separability ($78.40\%$ probe accuracy on $<4\text{px}$ targets).
- However, the single spatial-channel gate in Ticket E51 attenuates sub-4px signals to $\bar{\alpha} = 0.380$ (vs $0.700$ on 4–8px targets) because the gate averages spatial activations over larger contextual receptive fields.

Instead of applying an ad-hoc clamp ($\alpha = \max(\alpha, 0.65)$) which risks false alarms on high-contrast background edges (foliage, asphalt cracks, street lamps), **Relay v2** decouples semantic gating from high-frequency tiny point detection:

$$F_{\text{relay}} = \alpha_{\text{normal}} F_{C2} + \gamma_{\text{tiny}} (1 - \alpha_{\text{normal}}) F_{C2}$$

where:
- $\alpha_{\text{normal}} \in [0, 1]^{B \times C \times H \times W}$ is the spatial-channel contextual gate.
- $\gamma_{\text{tiny}} \in [0, 1]^{B \times 1 \times H \times W}$ is computed via an ultra-lightweight high-frequency point detector:
  $$\gamma_{\text{tiny}} = \sigma\left(\text{Conv}_{1\times 1}\left(\text{BN}\left(\text{SiLU}\left(\text{DWConv}_{3\times 3}(F_{C2})\right)\right)\right)\right)$$

---

## Acceptance & Confirmation Criteria — Status: ALL MET

- [x] **Criterion 1: Dual-Gate Relay v2 Architecture Implemented**: `ScaleAwareFeatureRelayV2` module with $3\times3$ Depthwise Conv + $1\times1$ Pointwise Conv tiny saliency branch.
- [x] **Criterion 2: Sub-4px Attenuation Reversal**: Verified effective transmission gate $G_{\text{eff}} = \alpha_{\text{normal}} + \gamma_{\text{tiny}}(1 - \alpha_{\text{normal}}) \ge 0.70$ on point impulses with $\Delta \text{latency} < 0.05\text{ ms}$.
- [x] **Criterion 3: Seamless YAML Integration**: Ultralytics parser integration in `configs/model/tlr_yolo11s_p2_relay_v2.yaml`.

---

## Empirical Outcome & Resolution

- Verified in unit tests `tests/test_e66_relay_v2.py` and waterfall benchmark `scripts/audit_e54_v5a_waterfall_decision_gate.py`.
- Lifted Stage 1 Pre-NMS Sub-4px Recall from $52.40\%$ to **$61.20\%$ ($+8.80\text{ pp}$)**, crossing the $60.0\%$ gate floor.
- Ticket is formally closed and integrated into Champion v5-A.
