"""Dynamic Point-Sampling and Content-Aware Upsampling Modules (Ticket E40).

Implements:
1. DySample (Liu et al., ICCV 2023): Ultra-lightweight dynamic point-sampling upsampler
   generating continuous sub-pixel coordinate offsets with minimal parameter/latency overhead.
2. CARAFE (Wang et al., ICCV 2019): Content-Aware ReAssembly of FEatures dynamic convolution baseline.
3. BilinearUpsample: Explicit bilinear interpolation module for static baseline comparisons.
4. Model surgery utilities to replace static upsamplers in YOLO P2 necks (lateral P3 -> P2 path).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DySample(nn.Module):
    """Dynamic Point-Sampling Upsampler (DySample, Liu et al., ICCV 2023).

    Instead of generating large dynamic convolution kernels (like CARAFE),
    DySample generates continuous 2D point offsets that dynamically resample the
    input feature map via bilinear grid sampling:
        offset = OffsetGen(X)
        Y = grid_sample(X, base_grid + offset, mode='bilinear')

    Supports both programmatic construction and YAML model parsing.
    """

    def __init__(
        self,
        *args: Any,
        in_channels: int | None = None,
        out_channels: int | None = None,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Flexible argument parsing supporting multiple calling conventions:
        # 1. DySample(in_channels, out_channels, scale, style, groups, dyscope)
        # 2. DySample(scale, style, groups) from YAML [2, lp, 4]
        # 3. DySample(None, scale, style, groups) from YAML [null, 2, lp, 4]
        parsed_in_channels = in_channels
        parsed_out_channels = out_channels
        parsed_scale = scale
        parsed_style = style
        parsed_groups = groups
        parsed_dyscope = dyscope

        if len(args) == 1:
            if isinstance(args[0], (int, float)) and int(args[0]) in (2, 3, 4, 8):
                parsed_scale = int(args[0])
            elif isinstance(args[0], int) and args[0] > 8:
                parsed_in_channels = args[0]
        elif len(args) >= 2:
            if args[0] is None or (isinstance(args[0], int) and args[0] > 8):
                parsed_in_channels = args[0]
                if len(args) >= 2 and args[1] is not None:
                    parsed_scale = int(args[1])
                if len(args) >= 3 and isinstance(args[2], str):
                    parsed_style = str(args[2])
                if len(args) >= 4 and isinstance(args[3], (int, float)):
                    parsed_groups = int(args[3])
                if len(args) >= 5:
                    parsed_dyscope = bool(args[4])
            elif isinstance(args[0], (int, float)) and int(args[0]) in (2, 3, 4, 8):
                parsed_scale = int(args[0])
                if isinstance(args[1], str):
                    parsed_style = str(args[1])
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    parsed_groups = int(args[2])
                if len(args) >= 4:
                    parsed_dyscope = bool(args[3])

        self.scale = int(parsed_scale)
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

        self.style = str(parsed_style)
        self.groups = int(parsed_groups)
        self.dyscope = bool(parsed_dyscope)
        self.in_channels = int(parsed_in_channels) if parsed_in_channels is not None else None
        self.out_channels = int(parsed_out_channels) if parsed_out_channels is not None else None

        self.offset_conv: nn.Module | None = None
        self.scope_conv: nn.Module | None = None
        self.proj: nn.Module | None = None

        if self.in_channels is not None:
            self._init_layers(self.in_channels, self.out_channels)

    def _init_layers(self, in_channels: int, out_channels: int | None = None) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        actual_groups = min(self.groups, self.in_channels)
        while actual_groups > 1 and self.in_channels % actual_groups != 0:
            actual_groups -= 1
        self.groups = max(1, actual_groups)

        out_offset_channels = 2 * self.groups * (self.scale**2)
        if self.style == "lp":
            self.offset_conv = nn.Conv2d(
                self.in_channels,
                out_offset_channels,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            if self.dyscope:
                self.scope_conv = nn.Conv2d(
                    self.in_channels,
                    out_offset_channels,
                    kernel_size=1,
                    bias=True,
                )
                nn.init.constant_(self.scope_conv.weight, 0.0)
                nn.init.constant_(self.scope_conv.bias, 0.0)
        elif self.style == "pl":
            hidden_channels = 2 * self.groups
            self.offset_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, out_offset_channels, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.offset_conv[-1].weight)
            nn.init.zeros_(self.offset_conv[-1].bias)
            if self.dyscope:
                self.scope_conv = nn.Conv2d(
                    self.in_channels,
                    out_offset_channels,
                    kernel_size=1,
                    bias=True,
                )
                nn.init.constant_(self.scope_conv.weight, 0.0)
                nn.init.constant_(self.scope_conv.bias, 0.0)
        else:
            raise ValueError(f"unsupported DySample style: {self.style!r} (choose 'lp' or 'pl')")

        if self.out_channels != self.in_channels:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def _generate_base_grid(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate uniform sampling grid in normalized coordinates [-1, 1] with align_corners=False."""
        sH = height * self.scale
        sW = width * self.scale
        y = torch.linspace(-1.0 + 1.0 / sH, 1.0 - 1.0 / sH, sH, device=device, dtype=dtype)
        x = torch.linspace(-1.0 + 1.0 / sW, 1.0 - 1.0 / sW, sW, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample feature tensor using dynamic point offsets.

        Args:
            x: Input feature tensor [B, C, H, W]
        Returns:
            Upsampled feature tensor [B, C_out, s*H, s*W]
        """
        B, C, H, W = x.shape
        if self.offset_conv is None:
            self._init_layers(C, self.out_channels)
            self.to(device=x.device, dtype=x.dtype)

        s = self.scale
        g = self.groups
        sH, sW = H * s, W * s

        offset = self.offset_conv(x)
        if self.dyscope and self.scope_conv is not None:
            scope = self.scope_conv(x).sigmoid()
            offset = offset * scope
        else:
            offset = offset * 0.5

        offset = offset.view(B * g, 2 * (s**2), H, W)
        offset = F.pixel_shuffle(offset, s)
        offset = offset.view(B, g, 2, sH, sW).permute(0, 1, 3, 4, 2)

        offset = offset.clone()
        offset[..., 0] = offset[..., 0] * (2.0 / W)
        offset[..., 1] = offset[..., 1] * (2.0 / H)

        base_grid = self._generate_base_grid(H, W, x.device, x.dtype)
        sample_grid = (base_grid + offset).view(B * g, sH, sW, 2)

        x_grouped = x.view(B * g, C // g, H, W)
        sampled = F.grid_sample(
            x_grouped,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        out = sampled.view(B, C, sH, sW)
        return self.proj(out) if self.proj is not None else out


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures (Wang et al., ICCV 2019).

    Predicts dynamic content-aware convolution kernels to reassemble features
    within a local region.
    """

    def __init__(
        self,
        *args: Any,
        in_channels: int | None = None,
        out_channels: int | None = None,
        scale: int = 2,
        k_up: int = 5,
        k_enc: int = 3,
        c_mid: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        parsed_in_channels = in_channels
        parsed_out_channels = out_channels
        parsed_scale = scale
        parsed_k_up = k_up
        parsed_k_enc = k_enc
        parsed_c_mid = c_mid

        if len(args) == 1:
            if isinstance(args[0], (int, float)) and int(args[0]) in (2, 3, 4, 8):
                parsed_scale = int(args[0])
            elif isinstance(args[0], int) and args[0] > 8:
                parsed_in_channels = args[0]
        elif len(args) >= 2:
            if args[0] is None or (isinstance(args[0], int) and args[0] > 8):
                parsed_in_channels = args[0]
                if len(args) >= 2 and args[1] is not None:
                    parsed_scale = int(args[1])
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    parsed_k_up = int(args[2])
                if len(args) >= 4 and isinstance(args[3], (int, float)):
                    parsed_k_enc = int(args[3])
                if len(args) >= 5 and isinstance(args[4], (int, float)):
                    parsed_c_mid = int(args[4])
            elif isinstance(args[0], (int, float)) and int(args[0]) in (2, 3, 4, 8):
                parsed_scale = int(args[0])
                if isinstance(args[1], (int, float)):
                    parsed_k_up = int(args[1])
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    parsed_k_enc = int(args[2])
                if len(args) >= 4 and isinstance(args[3], (int, float)):
                    parsed_c_mid = int(args[3])

        self.scale = int(parsed_scale)
        self.k_up = int(parsed_k_up)
        self.k_enc = int(parsed_k_enc)
        self.c_mid_target = int(parsed_c_mid)
        self.in_channels = int(parsed_in_channels) if parsed_in_channels is not None else None
        self.out_channels = int(parsed_out_channels) if parsed_out_channels is not None else None

        self.compressor: nn.Module | None = None
        self.encoder: nn.Module | None = None
        self.proj: nn.Module | None = None

        if self.in_channels is not None:
            self._init_layers(self.in_channels, self.out_channels)

    def _init_layers(self, in_channels: int, out_channels: int | None = None) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        c_mid = int(min(self.c_mid_target, self.in_channels))

        self.compressor = nn.Conv2d(self.in_channels, c_mid, kernel_size=1)
        self.encoder = nn.Conv2d(
            c_mid,
            (self.scale * self.k_up) ** 2,
            kernel_size=self.k_enc,
            padding=self.k_enc // 2,
        )
        if self.out_channels != self.in_channels:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.compressor is None or self.encoder is None:
            self._init_layers(C, self.out_channels)
            self.to(device=x.device, dtype=x.dtype)

        s = self.scale
        k_up = self.k_up
        sH, sW = H * s, W * s

        compressed = self.compressor(x)
        kernel = self.encoder(compressed)
        kernel = F.pixel_shuffle(kernel, s)
        kernel = F.softmax(kernel, dim=1)

        pad = (k_up - 1) // 2
        x_unfolded = F.unfold(x, kernel_size=k_up, padding=pad, stride=1).view(
            B, C, k_up**2, H, W
        )
        x_unfolded = F.interpolate(
            x_unfolded.view(B, C * (k_up**2), H, W),
            scale_factor=s,
            mode="nearest",
        ).view(B, C, k_up**2, sH, sW)

        out = (x_unfolded * kernel.unsqueeze(1)).sum(dim=2)
        return self.proj(out) if self.proj is not None else out


class BilinearUpsample(nn.Module):
    """Explicit Bilinear Interpolation Upsampling Layer."""

    def __init__(
        self,
        *args: Any,
        scale: int = 2,
        align_corners: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        parsed_scale = scale
        if len(args) >= 1:
            if isinstance(args[0], (int, float)) and int(args[0]) in (2, 3, 4, 8):
                parsed_scale = int(args[0])
            elif len(args) >= 2 and isinstance(args[1], (int, float)):
                parsed_scale = int(args[1])
        self.scale = float(parsed_scale)
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=self.align_corners,
        )


def replace_p2_upsampler(
    model_or_wrapper: Any,
    mode: str = "dysample",
    *,
    layer_index: int = 17,
    groups: int = 4,
    style: str = "lp",
    dyscope: bool = False,
) -> nn.Module:
    """Replace the lateral P3 -> P2 upsampling layer (default index 17) in a YOLO-P2 architecture.

    Args:
        model_or_wrapper: Ultralytics YOLO model or wrapper.
        mode: Upsampler type ('dysample', 'carafe', 'bilinear', 'nearest').
        layer_index: Index of the lateral P3 -> P2 upsampler (17 in standard TLR-YOLO-MTL P2).
        groups: Groups for DySample.
        style: DySample style ('lp' or 'pl').
        dyscope: Dynamic scope flag for DySample.

    Returns:
        The newly instantiated upsampling module.
    """
    model = model_or_wrapper.model if hasattr(model_or_wrapper, "model") else model_or_wrapper
    layers = model.model if hasattr(model, "model") else model

    if layer_index >= len(layers):
        raise IndexError(f"layer_index {layer_index} exceeds model layer count {len(layers)}")

    prev_layer = layers[layer_index - 1]
    if hasattr(prev_layer, "cv2") and hasattr(prev_layer.cv2, "conv"):
        in_channels = prev_layer.cv2.conv.out_channels
    elif hasattr(prev_layer, "conv"):
        in_channels = prev_layer.conv.out_channels
    elif hasattr(prev_layer, "c2"):
        in_channels = prev_layer.c2
    else:
        in_channels = 256

    if mode == "dysample":
        new_module = DySample(
            in_channels=in_channels,
            scale=2,
            style=style,
            groups=min(groups, in_channels),
            dyscope=dyscope,
        )
    elif mode == "carafe":
        new_module = CARAFE(
            in_channels=in_channels,
            scale=2,
            k_up=5,
            k_enc=3,
        )
    elif mode == "bilinear":
        new_module = BilinearUpsample(scale=2, align_corners=False)
    elif mode == "nearest":
        new_module = nn.Upsample(scale_factor=2, mode="nearest")
    else:
        raise ValueError(f"unknown upsampling mode: {mode!r}")

    # Transfer device and dtype
    orig_module = layers[layer_index]
    target_param = next(layers[layer_index - 1].parameters(), None)
    if target_param is not None:
        new_module = new_module.to(device=target_param.device, dtype=target_param.dtype)

    # Preserve Ultralytics task metadata (i, f, type, np)
    new_module.i = getattr(orig_module, "i", layer_index)
    new_module.f = getattr(orig_module, "f", -1)
    new_module.type = str(type(new_module))
    new_module.np = sum(p.numel() for p in new_module.parameters())

    layers[layer_index] = new_module
    return new_module


def register_dysample_modules() -> None:
    """Register DySample, CARAFE, and BilinearUpsample in Ultralytics and PyTorch namespaces."""
    import sys

    setattr(nn, "DySample", DySample)
    setattr(nn, "CARAFE", CARAFE)
    setattr(nn, "BilinearUpsample", BilinearUpsample)

    try:
        import ultralytics.nn.modules as u_modules
        import ultralytics.nn.tasks as u_tasks

        setattr(u_modules, "DySample", DySample)
        setattr(u_modules, "CARAFE", CARAFE)
        setattr(u_modules, "BilinearUpsample", BilinearUpsample)

        setattr(u_tasks, "DySample", DySample)
        setattr(u_tasks, "CARAFE", CARAFE)
        setattr(u_tasks, "BilinearUpsample", BilinearUpsample)
    except ImportError:
        pass


register_dysample_modules()
