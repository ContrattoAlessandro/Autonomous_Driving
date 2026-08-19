"""TL <-> Road Arrow Semantic Contrastive Alignment (Shared Maneuver Space) for TLR-YOLO-MTL (Ticket E26).

Enforces an explicit Supervised InfoNCE Contrastive Loss between traffic light
and road arrow candidate embeddings in a shared maneuver space (Left, Straight, Right):

    L_contrastive = -log [ sum_{p in P_i} exp(sim(e_TL,i, e_A,p) / tau) / sum_{a in P_i union N_i} exp(sim(e_TL,i, e_A,a) / tau) ]

Advantages:
1. Aligns latent representations of traffic lights and road arrows sharing identical maneuver intentions.
2. Repels incompatible maneuvers (e.g. Left Turn TL vs Right Turn Arrow).
3. Provides strong causal structure in the cross-attention embedding space.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TLArrowContrastiveProjector(nn.Module):
    """Projects candidate tokens into normalized shared maneuver embedding space."""

    def __init__(self, token_dim: int = 128, embed_dim: int = 64) -> None:
        super().__init__()
        self.tl_proj = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.arrow_proj = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self, traffic_tokens: torch.Tensor, arrow_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and L2-normalize tokens.

        Args:
            traffic_tokens: [B, K_TL, D]
            arrow_tokens: [B, K_Arrow, D]
        Returns:
            tl_embeds: [B, K_TL, embed_dim]
            arrow_embeds: [B, K_Arrow, embed_dim]
        """
        tl_embeds = F.normalize(self.tl_proj(traffic_tokens), p=2, dim=-1)
        arrow_embeds = F.normalize(self.arrow_proj(arrow_tokens), p=2, dim=-1)
        return tl_embeds, arrow_embeds


class TLArrowContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss aligning TL and Arrow maneuvers in shared latent space."""

    def __init__(
        self,
        token_dim: int = 128,
        embed_dim: int = 64,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.projector = TLArrowContrastiveProjector(token_dim=token_dim, embed_dim=embed_dim)
        self.temperature = float(temperature)

    def forward(
        self,
        traffic_tokens: torch.Tensor,
        arrow_tokens: torch.Tensor,
        *,
        traffic_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
        traffic_round: torch.Tensor,
        traffic_valid: torch.Tensor,
        arrow_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Supervised InfoNCE Loss.

        Args:
            traffic_tokens: [B, K_TL, D]
            arrow_tokens: [B, K_Arrow, D]
            traffic_maneuver: [B, K_TL, 3] (left, straight, right)
            arrow_maneuver: [B, K_Arrow, 3] (left, straight, right)
            traffic_round: [B, K_TL] in [0, 1]
            traffic_valid: [B, K_TL] bool
            arrow_valid: [B, K_Arrow] bool
        Returns:
            loss: scalar contrastive loss tensor
            metrics: dictionary of alignment statistics
        """
        B, K_TL, _ = traffic_tokens.shape
        K_Arrow = arrow_tokens.shape[1]

        tl_embeds, arrow_embeds = self.projector(traffic_tokens, arrow_tokens)

        # Cosine similarity matrix in [-1, 1]
        sim_matrix = torch.bmm(tl_embeds, arrow_embeds.transpose(1, 2))  # [B, K_TL, K_Arrow]

        # Maneuver compatibility matrix: [B, K_TL, K_Arrow]
        # True if TL and Arrow share at least one active maneuver class
        tl_man_bin = (traffic_maneuver > 0.5).float()
        ar_man_bin = (arrow_maneuver > 0.5).float()
        maneuver_match = (torch.bmm(tl_man_bin, ar_man_bin.transpose(1, 2)) > 0)  # [B, K_TL, K_Arrow]

        # Directional TL mask (not round, valid TL)
        directional_tl_mask = (traffic_round < 0.5) & traffic_valid  # [B, K_TL]

        total_loss = traffic_tokens.new_tensor(0.0)
        num_valid_queries = 0

        pos_sims: list[float] = []
        neg_sims: list[float] = []

        for b in range(B):
            valid_ar = arrow_valid[b]  # [K_Arrow]
            if not valid_ar.any():
                continue

            for i in range(K_TL):
                if not directional_tl_mask[b, i]:
                    continue

                pos_mask = maneuver_match[b, i] & valid_ar  # [K_Arrow]
                neg_mask = (~maneuver_match[b, i]) & valid_ar  # [K_Arrow]

                if not pos_mask.any() or not neg_mask.any():
                    continue

                sims = sim_matrix[b, i] / self.temperature  # [K_Arrow]
                
                # InfoNCE formulation
                pos_exp = torch.exp(sims[pos_mask])
                all_exp = torch.exp(sims[pos_mask | neg_mask])
                
                loss_i = -torch.log((pos_exp.sum() / (all_exp.sum() + 1e-7)).clamp_min(1e-7))
                total_loss = total_loss + loss_i
                num_valid_queries += 1

                pos_sims.append(float(sim_matrix[b, i][pos_mask].mean().item()))
                neg_sims.append(float(sim_matrix[b, i][neg_mask].mean().item()))

        if num_valid_queries > 0:
            loss = total_loss / num_valid_queries
        else:
            loss = total_loss * 0.0

        mean_pos = float(sum(pos_sims) / max(1, len(pos_sims))) if pos_sims else 0.0
        mean_neg = float(sum(neg_sims) / max(1, len(neg_sims))) if neg_sims else 0.0
        margin = mean_pos - mean_neg

        metrics = {
            "contrastive_loss": float(loss.item()),
            "mean_positive_sim": mean_pos,
            "mean_negative_sim": mean_neg,
            "alignment_margin": margin,
            "valid_queries": num_valid_queries,
        }

        return loss, metrics
