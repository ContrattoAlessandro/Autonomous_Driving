# W6 Diagnostic Audit: TaskAlignedAssigner Positive Allocation & NWD vs CIoU Interaction

**Audit Timestamp**: 2026-08-18 19:13:51
**Duration**: 59.1s

## 1. Executive Summary & Diagnostic Conclusion

- **Positive Anchor Starvation for Tiny Objects**: Ground-truth traffic lights $<32\text{ px}^2$ experience a starvation rate $P(N_{pos}=0)$ of **8.6%**, receiving only **2.29** positive candidate anchors on average (vs **9.89** for large TLs).
- **NWD vs IoU Overlap Sensitivity**: Max IoU with anchors drops to **0.196** for $<32\text{ px}^2$, while NWD retains a continuous gradient signal of **0.686**.
- **Gradient Synergy between CIoU and NWD**: Gradient cosine similarity on the bounding box regression head is strongly positive across all batches ($\mu = \mathbf{+0.612}$, 100.0% positive) and tiny TL batches ($\mu = \mathbf{+0.601}$), demonstrating **synergistic cooperation without antagonistic gradient conflicts**.
- **Architectural Verdict**: CIoU and NWD operate in harmony. However, because standard TaskAlignedAssigner alignment cost $t = s^\alpha \cdot \text{IoU}^\beta$ strictly relies on IoU (which collapses on sub-grid objects), an **NWD-aware assigner alignment metric** or **P2 high-resolution neck** is required to resolve anchor starvation.

## 2. Assigner Candidate Allocation per Area Bucket

| Area Bucket | GT Count | Starved GT ($N_{pos}=0$) | Starvation Rate | Mean $N_{pos}$ | P3 % | P4 % | P5 % | Max IoU | Max NWD | Max Alignment Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `<32` | 2206 | 189 | **8.6%** | **2.29** | 78.3% | 18.4% | 3.4% | 0.196 | 0.686 | 0.0001 |
| `32-64` | 1505 | 17 | **1.1%** | **3.72** | 77.3% | 18.9% | 3.8% | 0.372 | 0.739 | 0.0014 |
| `64-128` | 2247 | 5 | **0.2%** | **5.49** | 76.9% | 19.2% | 4.0% | 0.543 | 0.795 | 0.0102 |
| `128-256` | 2191 | 0 | **0.0%** | **7.14** | 77.1% | 19.3% | 3.6% | 0.711 | 0.849 | 0.0344 |
| `256-512` | 1859 | 0 | **0.0%** | **7.44** | 77.3% | 19.0% | 3.6% | 0.818 | 0.882 | 0.0702 |
| `>512` | 1996 | 0 | **0.0%** | **9.89** | 67.2% | 32.1% | 0.6% | 0.883 | 0.876 | 0.1033 |


## 3. Assigner Candidate Allocation per Min-Side Bucket

| Min-Side Bucket | GT Count | Starved GT ($N_{pos}=0$) | Starvation Rate | Mean $N_{pos}$ | P3 % | P4 % | P5 % | Max IoU | Max NWD | Max Alignment Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `<4` | 3963 | 194 | **4.9%** | **3.36** | 77.6% | 18.9% | 3.4% | 0.284 | 0.721 | 0.0007 |
| `4-6` | 1803 | 12 | **0.7%** | **5.65** | 76.7% | 19.2% | 4.1% | 0.537 | 0.783 | 0.0089 |
| `6-8` | 1904 | 2 | **0.1%** | **7.32** | 77.7% | 19.0% | 3.4% | 0.691 | 0.838 | 0.0304 |
| `8-12` | 1985 | 3 | **0.2%** | **6.24** | 76.2% | 19.9% | 3.9% | 0.799 | 0.872 | 0.0632 |
| `>12` | 2349 | 0 | **0.0%** | **9.53** | 68.8% | 30.2% | 1.0% | 0.876 | 0.878 | 0.1002 |


## 4. CIoU vs NWD Gradient Interaction on Bounding Box Head

| Metric | All Batches | Tiny-TL Batches ($<64\text{ px}^2$) |
|---|:---:|:---:|
| **Batches Analyzed** | 500 | 371 |
| **Mean Cosine Similarity $\cos(g_{CIoU}, g_{NWD})$** | **+0.6123** | **+0.6007** |
| **Std Dev** | 0.1159 | 0.1201 |
| **Median** | +0.6260 | +0.6153 |
| **Synergistic Alignment ($\% > 0$)** | **100.0%** | **100.0%** |
| **Mean ||g_{CIoU}||** | 8.6892 | — |
| **Mean ||g_{NWD}||** | 0.7223 | — |


## 5. Artifacts Generated

- Visualization: `results/visualizations/w6_assigner_allocation_nwd_ciou.png`

- Telemetry JSON: `results/audit_assigner_nwd_ciou.json`
