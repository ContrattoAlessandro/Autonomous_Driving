"""Scale-Aware C2 -> P2 Feature Relay for Raw Texture Recovery (Ticket E51).

Scientific Motivation:
In tiny traffic light perception (<8px), signals cover only 3-8 pixels in full-frame images.
While deep backbone stages (C2 -> C3 -> C4 -> C5) provide abstract semantic features, strided
convolutions and downsampling operations progressively attenuate high-frequency edge gradients
and sharp chromatic boundaries (e.g. glowing Red/Yellow/Green lamp discs).
Conversely, shallow backbone features (C2, stride 4, 480x240) retain pristine optical edge
and chromatic textures, but lack contextual semantics.

This module implements a lightweight, scale-conditioned feature relay from C2 directly into P2:
    phi(C2) = Conv1x1(BN(SiLU(C2))) in R^{C_neck x H_P2 x W_P2}
    G(C2, P2) = Conv1x1(SiLU(BN(Conv1x1([phi(C2); P2]))))
    P2_refined = P2 + sigma(G(C2, P2)) * phi(C2)

This selectively injects pristine high-resolution chromatic and edge cues only into spatial regions
exhibiting high-frequency traffic light signatures, preventing background clutter from polluting the neck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ScaleAwareRelayConfig:
    """Configuration for Scale-Aware Feature Relay."""
    enabled: bool = True
    gating_type: str = "spatial_channel"  # 'spatial_channel', 'spatial_only', 'channel_only', 'direct_sum'
    c2_channels: int = 64
    p2_channels: int = 64
    hidden_ratio: float = 0.5
    residual_scale: float = 1.0


class ScaleAwareFeatureRelay(nn.Module):
    """Scale-Aware C2 -> P2 Feature Relay Module.
    
    Supports both programmatic initialization and Ultralytics YAML parsing.
    
    Calling conventions:
    1. ScaleAwareFeatureRelay(c2_channels=64, p2_channels=64, gating_type='spatial_channel')
    2. ScaleAwareFeatureRelay(c2_channels, p2_channels, gating_type)
    3. From YAML: [-1, 1, ScaleAwareFeatureRelay, [64, 64, spatial_channel]]
    """

    def __init__(
        self,
        *args: Any,
        c2_channels: int | None = None,
        p2_channels: int | None = None,
        gating_type: str = "spatial_channel",
        hidden_ratio: float = 0.5,
        residual_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Flexible argument resolution
        parsed_c2 = c2_channels
        parsed_p2 = p2_channels
        parsed_gating = gating_type
        parsed_hidden = hidden_ratio
        parsed_scale = residual_scale

        if len(args) == 1:
            if isinstance(args[0], int):
                parsed_c2 = args[0]
                parsed_p2 = args[0]
            elif isinstance(args[0], str):
                parsed_gating = args[0]
        elif len(args) >= 2:
            if isinstance(args[0], int):
                parsed_c2 = args[0]
            if isinstance(args[1], int):
                parsed_p2 = args[1]
            if len(args) >= 3 and isinstance(args[2], str):
                parsed_gating = args[2]
            if len(args) >= 4 and isinstance(args[3], (int, float)):
                parsed_hidden = float(args[3])
            if len(args) >= 5 and isinstance(args[4], (int, float)):
                parsed_scale = float(args[4])

        self.c2_channels = int(parsed_c2) if parsed_c2 is not None else 64
        self.p2_channels = int(parsed_p2) if parsed_p2 is not None else 64
        self.gating_type = str(parsed_gating)
        self.hidden_ratio = float(parsed_hidden)
        self.residual_scale = float(parsed_scale)

        valid_gatings = {"spatial_channel", "spatial_only", "channel_only", "direct_sum"}
        if self.gating_type not in valid_gatings:
            raise ValueError(f"Unknown gating_type '{self.gating_type}', expected one of {valid_gatings}")

        hidden_dim = max(16, int(self.p2_channels * self.hidden_ratio))

        # 1. Raw Feature Projection phi(C2)
        self.c2_proj = nn.Sequential(
            nn.Conv2d(self.c2_channels, self.p2_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.p2_channels),
            nn.SiLU(inplace=True),
        )

        # 2. Gating Mechanism
        if self.gating_type == "spatial_channel":
            self.gate = nn.Sequential(
                nn.Conv2d(self.p2_channels * 2, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim, self.p2_channels, kernel_size=1, stride=1, bias=True),
                nn.Sigmoid(),
            )
        elif self.gating_type == "spatial_only":
            self.gate = nn.Sequential(
                nn.Conv2d(self.p2_channels * 2, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, bias=True),
                nn.Sigmoid(),
            )
        elif self.gating_type == "channel_only":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.gate = nn.Sequential(
                nn.Linear(self.p2_channels * 2, hidden_dim, bias=False),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, self.p2_channels, bias=True),
                nn.Sigmoid(),
            )
        else:  # direct_sum
            self.gate = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with near-zero gate bias to ensure clean initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        inputs: Union[torch.Tensor, Sequence[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        p2_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Scale-Aware Feature Relay.
        
        Args:
            inputs: Either:
                - Sequence/Tuple of [C2_feat, P2_feat] or [P2_feat, C2_feat]
                - Single Tensor C2_feat if p2_feat is passed explicitly as second argument
            p2_feat: Optional P2 feature map [B, C_p2, H_p2, W_p2]
            
        Returns:
            P2_refined: Refined P2 feature map [B, C_p2, H_p2, W_p2]
        """
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(f"Expected 2 inputs (C2, P2), got {len(inputs)}")
            feat_a, feat_b = inputs[0], inputs[1]
            if feat_a.shape[1] == self.c2_channels and feat_b.shape[1] == self.p2_channels:
                c2, p2 = feat_a, feat_b
            elif feat_b.shape[1] == self.c2_channels and feat_a.shape[1] == self.p2_channels:
                p2, c2 = feat_a, feat_b
            else:
                c2, p2 = feat_a, feat_b
        elif p2_feat is not None:
            c2, p2 = inputs, p2_feat
        else:
            raise ValueError("ScaleAwareFeatureRelay requires both C2 and P2 feature maps")

        # Ensure spatial alignment if needed
        if c2.shape[2:] != p2.shape[2:]:
            c2 = F.interpolate(c2, size=p2.shape[2:], mode="bilinear", align_corners=False)

        # 1. Project C2 features to P2 dimension
        c2_proj = self.c2_proj(c2)

        # 2. Compute Gating Map
        if self.gating_type == "direct_sum":
            p2_refined = p2 + self.residual_scale * c2_proj
        elif self.gating_type == "channel_only":
            cat_feat = torch.cat([c2_proj, p2], dim=1)
            pooled = self.pool(cat_feat).flatten(1)  # [B, 2*C_p2]
            g = self.gate(pooled).unsqueeze(-1).unsqueeze(-1)  # [B, C_p2, 1, 1]
            p2_refined = p2 + self.residual_scale * (g * c2_proj)
        else:  # spatial_channel or spatial_only
            cat_feat = torch.cat([c2_proj, p2], dim=1)  # [B, 2*C_p2, H, W]
            g = self.gate(cat_feat)  # [B, C_p2, H, W] or [B, 1, H, W]
            p2_refined = p2 + self.residual_scale * (g * c2_proj)

        return p2_refined


def get_module_out_channels(mod: nn.Module) -> int:
    """Helper to extract output channel dimension from any Ultralytics/PyTorch module."""
    if hasattr(mod, "cv2") and hasattr(mod.cv2, "conv"):
        return mod.cv2.conv.out_channels
    if hasattr(mod, "conv") and hasattr(mod.conv, "out_channels"):
        return mod.conv.out_channels
    if hasattr(mod, "c2") and isinstance(mod.c2, int):
        return mod.c2
    if hasattr(mod, "out_channels") and isinstance(mod.out_channels, int):
        return mod.out_channels
    for m in reversed(list(mod.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    return 64


def register_neck_modules() -> None:
    """Register neck modules in global and Ultralytics namespaces for YAML parsing."""
    import sys
    setattr(nn, "ScaleAwareFeatureRelay", ScaleAwareFeatureRelay)

    try:
        import copy
        import ultralytics.nn.modules as um
        import ultralytics.nn.tasks as ut

        setattr(um, "ScaleAwareFeatureRelay", ScaleAwareFeatureRelay)
        setattr(ut, "ScaleAwareFeatureRelay", ScaleAwareFeatureRelay)

        if hasattr(ut, "parse_model") and not getattr(ut.parse_model, "_has_relay_patch", False):
            orig_parse = ut.parse_model

            def relay_patched_parse_model(d, ch, verbose=True):
                # Inspect if any layer in backbone/head is ScaleAwareFeatureRelay
                all_layers = d.get("backbone", []) + d.get("head", [])
                relay_indices = {}
                for idx, layer_spec in enumerate(all_layers):
                    mod_name = layer_spec[2]
                    if mod_name == "ScaleAwareFeatureRelay" or mod_name is ScaleAwareFeatureRelay:
                        relay_indices[idx] = copy.deepcopy(layer_spec)

                if not relay_indices:
                    return orig_parse(d, ch, verbose)

                # Create modified d with nn.Identity placeholder (routing from P2 layer) to preserve channel flow
                d_mod = copy.deepcopy(d)
                all_mod_layers = d_mod.get("backbone", []) + d_mod.get("head", [])
                for idx, orig_spec in relay_indices.items():
                    f_spec = orig_spec[0]
                    p2_idx = f_spec[1] if isinstance(f_spec, list) and len(f_spec) == 2 else -1
                    all_mod_layers[idx][0] = p2_idx
                    all_mod_layers[idx][2] = "nn.Identity"
                    all_mod_layers[idx][3] = []

                seq_model, save_list = orig_parse(d_mod, ch, verbose)

                # Now replace placeholder nn.Identity modules with actual ScaleAwareFeatureRelay modules
                for idx, orig_spec in relay_indices.items():
                    f_spec = orig_spec[0]
                    args_spec = orig_spec[3] if len(orig_spec) > 3 else []
                    placeholder = seq_model[idx]

                    if isinstance(f_spec, list) and len(f_spec) == 2:
                        idx_c2 = f_spec[0] if f_spec[0] >= 0 else idx + f_spec[0]
                        idx_p2 = f_spec[1] if f_spec[1] >= 0 else idx + f_spec[1]
                        
                        mod_c2 = seq_model[idx_c2]
                        mod_p2 = seq_model[idx_p2]

                        c2_ch = get_module_out_channels(mod_c2)
                        p2_ch = get_module_out_channels(mod_p2)
                        resolved_f = [idx_c2, idx_p2]
                    else:
                        c2_ch, p2_ch = 64, 64
                        resolved_f = f_spec

                    gating = "spatial_channel"
                    hidden_r = 0.5
                    res_s = 1.0
                    for a in args_spec:
                        if isinstance(a, str) and a in {"spatial_channel", "spatial_only", "channel_only", "direct_sum"}:
                            gating = a
                        elif isinstance(a, (int, float)) and a <= 1.0:
                            hidden_r = float(a)

                    actual_relay = ScaleAwareFeatureRelay(
                        c2_channels=c2_ch,
                        p2_channels=p2_ch,
                        gating_type=gating,
                        hidden_ratio=hidden_r,
                        residual_scale=res_s,
                    )
                    actual_relay.i = getattr(placeholder, "i", idx)
                    actual_relay.f = resolved_f
                    actual_relay.type = "tlr_yolo_mtl.model.neck.ScaleAwareFeatureRelay"
                    actual_relay.np = sum(p.numel() for p in actual_relay.parameters())
                    actual_relay.c2 = p2_ch

                    seq_model[idx] = actual_relay

                    # Add input layers to save_list
                    if isinstance(resolved_f, list):
                        for x in resolved_f:
                            if x != -1:
                                save_list.append(x % idx)

                save_list = sorted(list(set(save_list)))
                return seq_model, save_list

            relay_patched_parse_model._has_relay_patch = True
            ut.parse_model = relay_patched_parse_model
    except ImportError:
        pass


# Automatically register on module import
register_neck_modules()
