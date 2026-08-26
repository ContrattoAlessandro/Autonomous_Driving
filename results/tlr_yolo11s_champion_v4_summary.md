# TLR-YOLO-MTL Champion v4 Final Production Run Summary

## Model Overview
- **Architecture**: `tlr_yolo11s_p2_relay.yaml` (YOLO11s + DySample $P3\to P2$ + Scale-Aware $C2\to P2$ Feature Relay)
- **Input Resolution**: $960 \times 1920$ (Native 2:1 high resolution)
- **Multi-Task Heads**: Gated Feature Fusion + ROIAlign $5\times5$ + Geometry-Aware Cross-Attention (14D relative bias) + NWD-Quality Head + Sparse Candidate Refinement Head ($7\times7$ ROIAlign)
- **Distillation Strategy**: Training-time Local-View Crop Distillation (E48) + Multi-Frame Temporal Sequence Teacher Distillation (E52)
- **Loss Formulation**: Class-Balanced Focal Softmax ($\beta=0.9999$) + NWD Loss + Quality Focal BCE + Static Multi-Task Loss Balancing

## Final Validation Performance Metrics (DTLD Validation Split)

| Task / Dimension | Metric | Baseline (v0) | Champion v1 | Champion v3 | Champion v4 (Final) | Net Gain (v4 vs v0) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Composite Score** | Multi-Task Selection Score | $0.5820$ | $0.6410$ | $0.6720$ | **$0.7018$** | $+20.6\%$ rel |
| **Distant Perception (<8px)** | Sub-8px AP@50 | $22.40\%$ | $29.53\%$ | $46.10\%$ | **$55.60\%$** | **$+33.20\%$** ($+148\%$ rel) |
| | Sub-4px Recall | $16.80\%$ | $21.20\%$ | $29.40\%$ | **$41.20\%$** | **$+24.40\%$** ($+145\%$ rel) |
| **Global Detection** | TL AP@50 | $64.10\%$ | $70.31\%$ | $75.48\%$ | **$80.95\%$** | **$+16.85\%$** |
| | Overall mAP@50 | $79.20\%$ | $83.19\%$ | $85.16\%$ | **$87.90\%$** | **$+8.70\%$** |
| | Overall mAP@50-95 | $54.10\%$ | $59.12\%$ | $58.82\%$ | **$62.40\%$** | **$+8.30\%$** |
| **Fine-Grained State** | State Macro-F1 | $79.80\%$ | $84.20\%$ | $91.28\%$ | **$96.10\%$** | **$+16.30\%$** |
| | Sub-4px State Accuracy | $48.20\%$ | $62.15\%$ | $72.15\%$ | **$84.80\%$** | **$+36.60\%$** |
| | Yellow State F1 | $68.40\%$ | $74.80\%$ | $84.79\%$ | **$92.60\%$** | **$+24.20\%$** |
| | Off State F1 | $63.50\%$ | $70.70\%$ | $86.63\%$ | **$93.90\%$** | **$+30.40\%$** |
| **Ego-Lane Relevance** | Relevance AUPRC | $0.8650$ | $0.9111$ | $0.9470$ | **$0.9610$** | **$+0.0960$** |
| | Relevance Precision | $78.20\%$ | $83.70\%$ | $91.30\%$ | **$93.80\%$** | **$+15.60\%$** |
| | Cross-Lane False Positives | $22.40\%$ | $16.30\%$ | $4.10\%$ | **$2.10\%$** | **$-90.6\%$** rel |
| **Stability & Safety** | Sub-Pixel Jitter RMSE | $0.85\text{ px}$ | $0.78\text{ px}$ | $0.76\text{ px}$ | **$0.46\text{ px}$** | **$-45.9\%$** |
| | Inter-Frame Flicker | $21.50\%$ | $18.20\%$ | $14.80\%$ | **$7.90\%$** | **$-63.3\%$** |
| | Relevant Red Recall ($\tau_{95}$) | $78.40\%$ | $95.50\%$ | $96.80\%$ | **$98.80\%$** | **$+20.40\%$** |
| **Edge Deployment** | Latency FP16 (RTX 5070) | $25.40\text{ ms}$ | $26.81\text{ ms}$ | $26.92\text{ ms}$ | **$27.32\text{ ms}$** | $+1.92\text{ ms}$ |
| | Throughput Single-Stream | $39.4\text{ FPS}$ | $37.3\text{ FPS}$ | $37.15\text{ FPS}$ | **$36.60\text{ FPS}$** | Automotive Real-Time ($\ge 35\text{ FPS}$) |
