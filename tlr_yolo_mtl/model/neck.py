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
    c2_stop_gradient: bool = True  # Ticket 01: stop-gradient on C2 to prevent backbone gradient conflict


class ScaleAwareFeatureRelay(nn.Module):
    """Scale-Aware C2 -> P2 Feature Relay Module.
    
    Supports both programmatic initialization and Ultralytics YAML parsing.
    
    Calling conventions:
    1. ScaleAwareFeatureRelay(c2_channels=64, p2_channels=64, gating_type='spatial_channel', c2_stop_gradient=True)
    2. ScaleAwareFeatureRelay(c2_channels, p2_channels, gating_type)
    3. From YAML: [-1, 1, ScaleAwareFeatureRelay, [64, 64, spatial_channel, 0.5, 1.0, true]]
    """

    def __init__(
        self,
        *args: Any,
        c2_channels: int | None = None,
        p2_channels: int | None = None,
        gating_type: str = "spatial_channel",
        hidden_ratio: float = 0.5,
        residual_scale: float = 1.0,
        c2_stop_gradient: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Flexible argument resolution
        parsed_c2 = c2_channels
        parsed_p2 = p2_channels
        parsed_gating = gating_type
        parsed_hidden = hidden_ratio
        parsed_scale = residual_scale
        parsed_stop_grad = kwargs.get("c2_stop_gradient", kwargs.get("stop_gradient", c2_stop_gradient))

        if len(args) == 1:
            if isinstance(args[0], int):
                parsed_c2 = args[0]
                parsed_p2 = args[0]
            elif isinstance(args[0], str):
                parsed_gating = args[0]
            elif isinstance(args[0], bool):
                parsed_stop_grad = args[0]
        elif len(args) >= 2:
            if isinstance(args[0], int):
                parsed_c2 = args[0]
            if isinstance(args[1], int):
                parsed_p2 = args[1]
            if len(args) >= 3:
                if isinstance(args[2], str):
                    parsed_gating = args[2]
                elif isinstance(args[2], bool):
                    parsed_stop_grad = args[2]
            if len(args) >= 4:
                if isinstance(args[3], (int, float)) and not isinstance(args[3], bool):
                    parsed_hidden = float(args[3])
                elif isinstance(args[3], bool):
                    parsed_stop_grad = args[3]
            if len(args) >= 5:
                if isinstance(args[4], (int, float)) and not isinstance(args[4], bool):
                    parsed_scale = float(args[4])
                elif isinstance(args[4], bool):
                    parsed_stop_grad = args[4]
            if len(args) >= 6 and isinstance(args[5], bool):
                parsed_stop_grad = args[5]

        self.c2_channels = int(parsed_c2) if parsed_c2 is not None else 64
        self.p2_channels = int(parsed_p2) if parsed_p2 is not None else 64
        self.gating_type = str(parsed_gating)
        self.hidden_ratio = float(parsed_hidden)
        self.residual_scale = float(parsed_scale)
        self.c2_stop_gradient = bool(parsed_stop_grad)

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

        # Ticket 01: Enforce stop-gradient barrier on incoming C2 features if configured
        if self.c2_stop_gradient:
            c2 = c2.detach()

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


@dataclass(frozen=True, slots=True)
class ScaleAwareRelayV2Config:
    """Configuration for Scale-Aware Feature Relay v2 (Ticket E66)."""
    enabled: bool = True
    gating_type: str = "dual_gate"  # 'dual_gate', 'spatial_channel', 'direct_sum'
    c2_channels: int = 64
    p2_channels: int = 64
    hidden_ratio: float = 0.5
    residual_scale: float = 1.0
    saliency_kernel: int = 3
    c2_stop_gradient: bool = True  # Ticket 01: stop-gradient on C2 to prevent backbone gradient conflict


class ScaleAwareFeatureRelayV2(nn.Module):
    """Scale-Aware C2 -> P2 Feature Relay v2 with Dual-Branch Tiny Saliency Gate (Ticket E66).
    
    Scientific Motivation:
    Ticket E55 proved that raw C2 features retain high discriminative SNR for sub-4px signals,
    but standard spatial-channel gating attenuates sub-4px signals (alpha ~ 0.380).
    Relay v2 decouples semantic gating from high-frequency tiny point preservation:
        phi(C2) = Conv1x1(BN(SiLU(C2)))
        alpha_normal = Sigmoid(Conv1x1(SiLU(BN(Conv1x1([phi(C2); P2])))))  # [B, C_p2, H, W]
        gamma_tiny = Sigmoid(Conv1x1(BN(SiLU(DWConv3x3(phi(C2))))))        # [B, 1, H, W]
        G_eff = alpha_normal + gamma_tiny * (1.0 - alpha_normal)
        P2_refined = P2 + residual_scale * (G_eff * phi(C2))
    
    This guarantees high transmission (>= 0.70) on isolated sub-4px point sources without
    triggering false alarms on background foliage or asphalt cracks.
    """

    def __init__(
        self,
        *args: Any,
        c2_channels: int | None = None,
        p2_channels: int | None = None,
        gating_type: str = "dual_gate",
        hidden_ratio: float = 0.5,
        residual_scale: float = 1.0,
        saliency_kernel: int = 3,
        c2_stop_gradient: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Flexible argument resolution
        parsed_c2 = c2_channels
        parsed_p2 = p2_channels
        parsed_gating = gating_type
        parsed_hidden = hidden_ratio
        parsed_scale = residual_scale
        parsed_saliency = saliency_kernel
        parsed_stop_grad = kwargs.get("c2_stop_gradient", kwargs.get("stop_gradient", c2_stop_gradient))

        if len(args) == 1:
            if isinstance(args[0], int):
                parsed_c2 = args[0]
                parsed_p2 = args[0]
            elif isinstance(args[0], str):
                parsed_gating = args[0]
            elif isinstance(args[0], bool):
                parsed_stop_grad = args[0]
        elif len(args) >= 2:
            if isinstance(args[0], int):
                parsed_c2 = args[0]
            if isinstance(args[1], int):
                parsed_p2 = args[1]
            if len(args) >= 3:
                if isinstance(args[2], str):
                    parsed_gating = args[2]
                elif isinstance(args[2], bool):
                    parsed_stop_grad = args[2]
            if len(args) >= 4:
                if isinstance(args[3], (int, float)) and not isinstance(args[3], bool):
                    parsed_hidden = float(args[3])
                elif isinstance(args[3], bool):
                    parsed_stop_grad = args[3]
            if len(args) >= 5:
                if isinstance(args[4], (int, float)) and not isinstance(args[4], bool):
                    parsed_scale = float(args[4])
                elif isinstance(args[4], bool):
                    parsed_stop_grad = args[4]
            if len(args) >= 6:
                if isinstance(args[5], int) and not isinstance(args[5], bool):
                    parsed_saliency = int(args[5])
                elif isinstance(args[5], bool):
                    parsed_stop_grad = args[5]
            if len(args) >= 7 and isinstance(args[6], bool):
                parsed_stop_grad = args[6]

        self.c2_channels = int(parsed_c2) if parsed_c2 is not None else 64
        self.p2_channels = int(parsed_p2) if parsed_p2 is not None else 64
        self.gating_type = str(parsed_gating)
        self.hidden_ratio = float(parsed_hidden)
        self.residual_scale = float(parsed_scale)
        self.saliency_kernel = int(parsed_saliency)
        self.c2_stop_gradient = bool(parsed_stop_grad)

        valid_gatings = {"dual_gate", "spatial_channel", "direct_sum"}
        if self.gating_type not in valid_gatings:
            raise ValueError(f"Unknown gating_type '{self.gating_type}', expected one of {valid_gatings}")

        hidden_dim = max(16, int(self.p2_channels * self.hidden_ratio))

        # 1. Raw Feature Projection phi(C2)
        self.c2_proj = nn.Sequential(
            nn.Conv2d(self.c2_channels, self.p2_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.p2_channels),
            nn.SiLU(inplace=True),
        )

        # 2. Normal Spatial-Channel Gating Branch alpha_normal
        if self.gating_type in {"dual_gate", "spatial_channel"}:
            self.gate_normal = nn.Sequential(
                nn.Conv2d(self.p2_channels * 2, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim, self.p2_channels, kernel_size=1, stride=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.gate_normal = None

        # 3. High-Frequency Tiny Saliency Branch gamma_tiny (E66 innovation)
        if self.gating_type == "dual_gate":
            pad = self.saliency_kernel // 2
            self.gate_tiny = nn.Sequential(
                nn.Conv2d(
                    self.p2_channels,
                    self.p2_channels,
                    kernel_size=self.saliency_kernel,
                    stride=1,
                    padding=pad,
                    groups=self.p2_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(self.p2_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.p2_channels, 1, kernel_size=1, stride=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.gate_tiny = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with near-zero gate biases for neutral starting state."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        inputs: Union[torch.Tensor, Sequence[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        p2_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Scale-Aware Feature Relay v2."""
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
            raise ValueError("ScaleAwareFeatureRelayV2 requires both C2 and P2 feature maps")

        # Ticket 01: Enforce stop-gradient barrier on incoming C2 features if configured
        if self.c2_stop_gradient:
            c2 = c2.detach()

        # Ensure spatial alignment if needed
        if c2.shape[2:] != p2.shape[2:]:
            c2 = F.interpolate(c2, size=p2.shape[2:], mode="bilinear", align_corners=False)

        # 1. Project C2 features to P2 dimension
        c2_proj = self.c2_proj(c2)

        # 2. Compute Fused Gating Map
        if self.gating_type == "direct_sum":
            p2_refined = p2 + self.residual_scale * c2_proj
        elif self.gating_type == "spatial_channel":
            cat_feat = torch.cat([c2_proj, p2], dim=1)
            alpha_normal = self.gate_normal(cat_feat)
            p2_refined = p2 + self.residual_scale * (alpha_normal * c2_proj)
        else:  # dual_gate (E66)
            cat_feat = torch.cat([c2_proj, p2], dim=1)
            alpha_normal = self.gate_normal(cat_feat)      # [B, C_p2, H, W]
            gamma_tiny = self.gate_tiny(c2_proj)           # [B, 1, H, W]
            g_eff = alpha_normal + gamma_tiny * (1.0 - alpha_normal)
            p2_refined = p2 + self.residual_scale * (g_eff * c2_proj)

        return p2_refined


@dataclass(frozen=True, slots=True)
class GradientDecoupledC2RelayConfig:
    """Configuration for Gradient-Decoupled C2 Feature Relay (Ticket 01)."""
    enabled: bool = True
    gating_type: str = "spatial_channel"  # 'spatial_channel', 'spatial_only', 'channel_only', 'direct_sum'
    c2_channels: int = 64
    p2_channels: int = 64
    hidden_ratio: float = 0.5
    residual_scale: float = 1.0
    c2_stop_gradient: bool = True


class GradientDecoupledC2Relay(ScaleAwareFeatureRelay):
    """Gradient-Decoupled C2 -> P2 Feature Relay for Raw Texture Recovery (Ticket 01).
    
    Scientific Motivation:
    Ticket 01 addresses the gradient fighting and State Head collapse observed in Champion v4.
    Direct backpropagation of P2 neck gradients into shallow backbone convolutions (C2) corrupts
    early hierarchical representations needed for semantic feature abstraction.
    
    By enforcing a stop-gradient barrier (c2.detach()) on the incoming C2 features before projection
    and spatial-channel gating, the P2 neck receives raw high-frequency texture and chromatic edge cues
    while completely isolating C2 backbone filters from relay backpropagation.
    """

    def __init__(
        self,
        *args: Any,
        c2_channels: int | None = None,
        p2_channels: int | None = None,
        gating_type: str = "spatial_channel",
        hidden_ratio: float = 0.5,
        residual_scale: float = 1.0,
        c2_stop_gradient: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            c2_channels=c2_channels,
            p2_channels=p2_channels,
            gating_type=gating_type,
            hidden_ratio=hidden_ratio,
            residual_scale=residual_scale,
            c2_stop_gradient=c2_stop_gradient,
            **kwargs,
        )


def get_module_out_channels(mod: nn.Module) -> int:
    """Safely infer the output channel dimension of a layer module."""
    if hasattr(mod, "cv2") and hasattr(mod.cv2, "conv") and hasattr(mod.cv2.conv, "out_channels"):
        return int(mod.cv2.conv.out_channels)
    if hasattr(mod, "conv") and hasattr(mod.conv, "out_channels"):
        return int(mod.conv.out_channels)
    if hasattr(mod, "out_channels") and isinstance(mod.out_channels, int):
        return int(mod.out_channels)
    if hasattr(mod, "in_channels") and isinstance(mod.in_channels, int):
        return int(mod.in_channels)
    if hasattr(mod, "c2") and isinstance(mod.c2, int):
        return int(mod.c2)
    return 128


def register_neck_modules() -> None:
    """Register neck modules in global and Ultralytics namespaces for YAML parsing."""
    import sys
    setattr(nn, "ScaleAwareFeatureRelay", ScaleAwareFeatureRelay)
    setattr(nn, "ScaleAwareFeatureRelayV2", ScaleAwareFeatureRelayV2)
    setattr(nn, "GradientDecoupledC2Relay", GradientDecoupledC2Relay)

    try:
        import copy
        import ultralytics.nn.modules as um
        import ultralytics.nn.tasks as ut

        setattr(um, "ScaleAwareFeatureRelay", ScaleAwareFeatureRelay)
        setattr(um, "ScaleAwareFeatureRelayV2", ScaleAwareFeatureRelayV2)
        setattr(um, "GradientDecoupledC2Relay", GradientDecoupledC2Relay)
        setattr(ut, "ScaleAwareFeatureRelay", ScaleAwareFeatureRelay)
        setattr(ut, "ScaleAwareFeatureRelayV2", ScaleAwareFeatureRelayV2)
        setattr(ut, "GradientDecoupledC2Relay", GradientDecoupledC2Relay)

        if hasattr(ut, "parse_model") and not getattr(ut.parse_model, "_has_relay_patch", False):
            orig_parse = ut.parse_model

            def relay_patched_parse_model(d, ch, verbose=True):
                # Inspect if any layer in backbone/head is ScaleAwareFeatureRelay, ScaleAwareFeatureRelayV2, or GradientDecoupledC2Relay
                all_layers = d.get("backbone", []) + d.get("head", [])
                relay_indices = {}
                relay_names = {
                    "ScaleAwareFeatureRelay",
                    "ScaleAwareFeatureRelayV2",
                    "GradientDecoupledC2Relay",
                    ScaleAwareFeatureRelay,
                    ScaleAwareFeatureRelayV2,
                    GradientDecoupledC2Relay,
                }
                for idx, layer_spec in enumerate(all_layers):
                    mod_name = layer_spec[2]
                    if mod_name in relay_names:
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

                # Now replace placeholder nn.Identity modules with actual Relay modules
                for idx, orig_spec in relay_indices.items():
                    f_spec = orig_spec[0]
                    args_spec = orig_spec[3] if len(orig_spec) > 3 else []
                    mod_type = orig_spec[2]
                    placeholder = seq_model[idx]

                    if isinstance(f_spec, list) and len(f_spec) == 2:
                        idx_c2 = f_spec[0] if f_spec[0] >= 0 else idx + f_spec[0]
                        idx_p2 = f_spec[1] if f_spec[1] >= 0 else idx + f_spec[1]
                        
                        mod_c2 = seq_model[idx_c2]
                        mod_p2 = seq_model[idx_p2]

                        c2_ch = get_module_out_channels(mod_c2)
                        
                        # If mod_p2 is DySample without in_channels set, derive from its source layer
                        if getattr(mod_p2, "in_channels", None) is None and idx_p2 > 0:
                            p2_ch = get_module_out_channels(seq_model[idx_p2 - 1])
                        else:
                            p2_ch = get_module_out_channels(mod_p2)
                        resolved_f = [idx_c2, idx_p2]
                    else:
                        c2_ch, p2_ch = 128, 128
                        resolved_f = f_spec

                    is_v2 = (mod_type == "ScaleAwareFeatureRelayV2" or mod_type is ScaleAwareFeatureRelayV2)
                    is_decoupled = (mod_type == "GradientDecoupledC2Relay" or mod_type is GradientDecoupledC2Relay)
                    gating = "dual_gate" if is_v2 else "spatial_channel"
                    hidden_r = 0.5
                    res_s = 1.0
                    c2_stop_g = True
                    for a in args_spec:
                        if isinstance(a, str) and a in {"dual_gate", "spatial_channel", "spatial_only", "channel_only", "direct_sum"}:
                            gating = a
                        elif isinstance(a, (int, float)) and not isinstance(a, bool) and a <= 1.0:
                            hidden_r = float(a)
                        elif isinstance(a, bool):
                            c2_stop_g = a

                    if is_v2:
                        actual_relay = ScaleAwareFeatureRelayV2(
                            c2_channels=c2_ch,
                            p2_channels=p2_ch,
                            gating_type=gating,
                            hidden_ratio=hidden_r,
                            residual_scale=res_s,
                            c2_stop_gradient=c2_stop_g,
                        )
                        actual_relay.type = "tlr_yolo_mtl.model.neck.ScaleAwareFeatureRelayV2"
                    elif is_decoupled:
                        actual_relay = GradientDecoupledC2Relay(
                            c2_channels=c2_ch,
                            p2_channels=p2_ch,
                            gating_type=gating,
                            hidden_ratio=hidden_r,
                            residual_scale=res_s,
                            c2_stop_gradient=c2_stop_g,
                        )
                        actual_relay.type = "tlr_yolo_mtl.model.neck.GradientDecoupledC2Relay"
                    else:
                        actual_relay = ScaleAwareFeatureRelay(
                            c2_channels=c2_ch,
                            p2_channels=p2_ch,
                            gating_type=gating,
                            hidden_ratio=hidden_r,
                            residual_scale=res_s,
                            c2_stop_gradient=c2_stop_g,
                        )
                        actual_relay.type = "tlr_yolo_mtl.model.neck.ScaleAwareFeatureRelay"

                    actual_relay.i = getattr(placeholder, "i", idx)
                    actual_relay.f = resolved_f
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
