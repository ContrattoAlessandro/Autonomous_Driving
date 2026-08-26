"""Physics-Grounded Photometric Traffic Light Augmentation (Ticket E39).

Replaces unconstrained generic HSV color jitter with optics- and domain-specific
photometric transforms designed for traffic lights in autonomous driving scenes:

1. Parametric Gaussian Lamp Bloom:
   Synthesizes subtle additive point-spread glow N(mu_lamp, sigma_bloom^2) radiating
   from the active emissive lamp (Red, Yellow, Green) into the surrounding housing/scene
   with state-matched chromatic emission spectra.

2. Physics-Grounded Exposure & Gamma Transforms:
   Non-linear gamma curve adjustments (gamma in [0.7, 1.4]), dynamic highlight saturation
   and core clipping, and low-light / night ambient attenuation.

3. Sensor Noise & Optical Defocus:
   CMOS sensor shot/read noise modeling low-light grain, subtle motion blur, and mild defocus.

4. Wet-Lens Glare & Halo Flare:
   Simulates radial glare streaks and starburst halos around bright active light sources.

5. Strict Hue Preservation Constraint:
   Hue shifts on traffic light regions are locked or strictly constrained (|hsv_h| <= 0.005)
   to eliminate synthetic state transitions and label noise on sub-8px objects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .schema import BBox, ImageRecord, TrafficLightAnnotation
from .taxonomy import normalize_label


# Canonical state-matched chromatic emission RGB spectra [R, G, B] in [0, 255]
EMISSIVE_SPECTRA: dict[str, tuple[float, float, float]] = {
    "red": (255.0, 35.0, 30.0),
    "yellow": (255.0, 210.0, 25.0),
    "green": (30.0, 255.0, 130.0),
}


@dataclass(frozen=True, slots=True)
class PhotometricAugmentationConfig:
    """Hyperparameters for physics-grounded photometric augmentation suite."""

    # Master probability of applying photometric augmentation
    photometric_prob: float = 0.50

    # Parametric Lamp Bloom parameters
    enable_lamp_bloom: bool = True
    lamp_bloom_prob: float = 0.40
    bloom_intensity_min: float = 0.25
    bloom_intensity_max: float = 0.75
    bloom_radius_scale_min: float = 0.60
    bloom_radius_scale_max: float = 1.80

    # Exposure & Gamma parameters
    gamma_min: float = 0.75
    gamma_max: float = 1.30
    exposure_scale_min: float = 0.80
    exposure_scale_max: float = 1.25
    highlight_clipping_prob: float = 0.25

    # Sensor Noise & Defocus
    sensor_noise_prob: float = 0.20
    sensor_noise_sigma_max: float = 4.0
    defocus_blur_prob: float = 0.15

    # Wet-Lens Glare / Flare parameters
    enable_wet_lens_glare: bool = True
    wet_lens_glare_prob: float = 0.20
    glare_intensity_min: float = 0.20
    glare_intensity_max: float = 0.60

    # Strict Hue Preservation limit (fraction of 180 degrees)
    max_hue_jitter: float = 0.004  # <= 0.72 degrees, prevents chromatic state corruption


DEFAULT_PHOTOMETRIC_CONFIG = PhotometricAugmentationConfig()


def estimate_lamp_center(
    box: BBox,
    state: str | None,
    *,
    aspect_ratio_threshold: float = 1.2,
) -> tuple[float, float]:
    """Estimate the spatial center (cx, cy) of the active lamp within a traffic light box.

    For vertical 3-aspect heads (height > width * threshold):
    - Red lamp is located in the top third (~20% from top).
    - Yellow lamp is located in the middle (~50% from top).
    - Green lamp is located in the bottom third (~80% from top).

    For horizontal, square, or unknown configurations, defaults to bounding box geometric center.
    """
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = float(x1 + x2) * 0.5
    cy = float(y1 + y2) * 0.5

    norm_state = normalize_label(state)
    if h >= w * aspect_ratio_threshold:
        # Standard vertical configuration
        if norm_state == "red":
            cy = float(y1) + 0.20 * h
        elif norm_state in {"yellow", "amber"}:
            cy = float(y1) + 0.50 * h
        elif norm_state == "green":
            cy = float(y1) + 0.80 * h

    return cx, cy


def synthesize_lamp_bloom(
    image_rgb: np.ndarray,
    box: BBox,
    state: str | None,
    *,
    intensity: float = 0.50,
    radius_scale: float = 1.0,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Synthesize active parametric Gaussian lamp bloom for emissive traffic lights.

    Models physical point-spread glow N(mu_lamp, sigma_bloom^2) radiating
    from the active bulb/LED across housing and atmospheric boundary.
    """
    norm_state = normalize_label(state)
    if norm_state not in EMISSIVE_SPECTRA:
        # "off", "unknown", or invalid states have no active emissive lamp bloom
        return image_rgb

    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    min_side = min(w, h)

    cx, cy = estimate_lamp_center(box, norm_state)
    sigma = max(1.5, min_side * 0.50 * max(0.2, radius_scale))
    kernel_radius = int(math.ceil(3.5 * sigma))

    img_h, img_w = image_rgb.shape[:2]
    ix_min = max(0, int(math.floor(cx - kernel_radius)))
    ix_max = min(img_w, int(math.ceil(cx + kernel_radius + 1)))
    iy_min = max(0, int(math.floor(cy - kernel_radius)))
    iy_max = min(img_h, int(math.ceil(cy + kernel_radius + 1)))

    if ix_max <= ix_min or iy_max <= iy_min:
        return image_rgb

    # Construct 2D Gaussian glow grid
    grid_y, grid_x = np.ogrid[iy_min:iy_max, ix_min:ix_max]
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    gaussian = np.exp(-0.5 * dist_sq / (sigma**2)).astype(np.float32)

    # State chromatic emission vector
    color = np.array(EMISSIVE_SPECTRA[norm_state], dtype=np.float32)  # shape (3,)

    # Compute additive bloom layer
    intensity_clamped = float(np.clip(intensity, 0.05, 1.0))
    glow = (gaussian[..., np.newaxis] * color * intensity_clamped)

    out = image_rgb.copy()
    patch = out[iy_min:iy_max, ix_min:ix_max].astype(np.float32)
    # Additive emission with physical saturating response
    blended = np.clip(patch + glow, 0.0, 255.0).astype(np.uint8)
    out[iy_min:iy_max, ix_min:ix_max] = blended
    return out


def apply_exposure_and_gamma(
    image_rgb: np.ndarray,
    *,
    gamma: float = 1.0,
    exposure_scale: float = 1.0,
    clip_highlights: bool = False,
) -> np.ndarray:
    """Non-linear exposure and gamma curve transform with optional highlight core clipping."""
    img_f = image_rgb.astype(np.float32) / 255.0

    # Exposure linear modulation
    if abs(exposure_scale - 1.0) > 1e-4:
        img_f = img_f * max(0.1, exposure_scale)

    # Gamma non-linear tone curve
    if abs(gamma - 1.0) > 1e-4:
        gamma_safe = max(0.4, min(2.5, gamma))
        img_f = np.power(np.clip(img_f, 0.0, 1.0), gamma_safe)

    out = np.clip(img_f * 255.0, 0.0, 255.0)

    # Core highlight saturation & clipping: bright regions (>230) desaturate towards white
    if clip_highlights:
        brightness = np.mean(out, axis=2, keepdims=True)
        high_mask = (brightness > 225.0).astype(np.float32)
        factor = np.clip((brightness - 225.0) / 30.0, 0.0, 1.0)
        out = (1.0 - factor * 0.70) * out + factor * 0.70 * np.full_like(out, 255.0)
        out = np.clip(out, 0.0, 255.0)

    return out.astype(np.uint8)


def apply_sensor_noise_and_defocus(
    image_rgb: np.ndarray,
    *,
    apply_noise: bool = False,
    noise_sigma: float = 3.0,
    apply_defocus: bool = False,
    kernel_size: int = 3,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Controlled CMOS sensor noise and mild optical defocus blur."""
    out = image_rgb.copy()
    resolved_rng = rng or random.Random()

    if apply_noise and noise_sigma > 0.0:
        seed = resolved_rng.randrange(2**32)
        np_rng = np.random.default_rng(seed)
        # Heteroscedastic noise (shot noise proportional to sqrt(intensity) + thermal Gaussian noise)
        intensity_norm = out.astype(np.float32) / 255.0
        shot_scale = np.sqrt(np.clip(intensity_norm, 0.0, 1.0)) * (noise_sigma * 0.5)
        read_noise = noise_sigma * 0.5
        noise = np_rng.normal(0.0, 1.0, out.shape).astype(np.float32) * (shot_scale + read_noise)
        out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    if apply_defocus:
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        k = max(3, min(7, k))
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out


def apply_wet_lens_glare(
    image_rgb: np.ndarray,
    traffic_lights: Sequence[TrafficLightAnnotation],
    *,
    glare_intensity: float = 0.35,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Synthesize subtle radial flare/glare streaks originating from active lamps on wet lenses."""
    active_lights = [
        tl for tl in traffic_lights
        if normalize_label(tl.state) in EMISSIVE_SPECTRA and tl.valid_state
    ]
    if not active_lights:
        return image_rgb

    resolved_rng = rng or random.Random()
    out = image_rgb.copy()
    img_h, img_w = image_rgb.shape[:2]

    for tl in active_lights:
        state = normalize_label(tl.state)
        cx, cy = estimate_lamp_center(tl.bbox_xyxy, state)
        color = np.array(EMISSIVE_SPECTRA[state], dtype=np.float32)

        w = max(1.0, float(tl.bbox_xyxy[2] - tl.bbox_xyxy[0]))
        h = max(1.0, float(tl.bbox_xyxy[3] - tl.bbox_xyxy[1]))
        radius = max(6, int(min(w, h) * 2.5))

        # Angle of glare streak (e.g. 45 degrees or horizontal flare)
        angle_deg = resolved_rng.choice([0.0, 45.0, 90.0, 135.0])
        rad = math.radians(angle_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)

        pt1 = (int(round(cx - dx * radius)), int(round(cy - dy * radius)))
        pt2 = (int(round(cx + dx * radius)), int(round(cy + dy * radius)))

        # Create alpha mask for glare line
        glare_canvas = np.zeros((img_h, img_w), dtype=np.float32)
        cv2.line(glare_canvas, pt1, pt2, 1.0, thickness=max(1, int(min(w, h) * 0.25)))
        glare_canvas = cv2.GaussianBlur(glare_canvas, (5, 5), 0)

        # Blend glare into output
        alpha = glare_canvas[..., np.newaxis] * glare_intensity
        patch = out.astype(np.float32) + alpha * color
        out = np.clip(patch, 0.0, 255.0).astype(np.uint8)

    return out


def apply_physics_photometric_augmentation(
    image_rgb: np.ndarray,
    record: ImageRecord,
    *,
    config: PhotometricAugmentationConfig = DEFAULT_PHOTOMETRIC_CONFIG,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Master physics-grounded photometric augmentation pipeline for traffic light scenes.

    Applies exposure, gamma, sensor noise, wet-lens glare, and active parametric lamp bloom
    while strictly enforcing hue preservation on traffic light regions.
    """
    resolved_rng = rng or random.Random()
    if resolved_rng.random() > config.photometric_prob:
        return image_rgb

    augmented = image_rgb.copy()

    # 1. Non-linear Exposure & Gamma modulation
    gamma = resolved_rng.uniform(config.gamma_min, config.gamma_max)
    exposure = resolved_rng.uniform(config.exposure_scale_min, config.exposure_scale_max)
    clip_highlights = (resolved_rng.random() < config.highlight_clipping_prob)
    augmented = apply_exposure_and_gamma(
        augmented,
        gamma=gamma,
        exposure_scale=exposure,
        clip_highlights=clip_highlights,
    )

    # 2. Strict Hue Preservation Jitter (Hue shift tightly bounded to prevent label corruption)
    if config.max_hue_jitter > 0.0:
        hue_shift = int(resolved_rng.uniform(-config.max_hue_jitter, config.max_hue_jitter) * 180.0)
        if hue_shift != 0:
            hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 180.0)
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 3. Parametric Gaussian Lamp Bloom on Active Traffic Lights
    if config.enable_lamp_bloom and record.traffic_lights:
        for tl in record.traffic_lights:
            state = normalize_label(tl.state)
            if state in EMISSIVE_SPECTRA and tl.valid_state:
                if resolved_rng.random() < config.lamp_bloom_prob:
                    intensity = resolved_rng.uniform(
                        config.bloom_intensity_min, config.bloom_intensity_max
                    )
                    r_scale = resolved_rng.uniform(
                        config.bloom_radius_scale_min, config.bloom_radius_scale_max
                    )
                    augmented = synthesize_lamp_bloom(
                        augmented,
                        tl.bbox_xyxy,
                        state,
                        intensity=intensity,
                        radius_scale=r_scale,
                        rng=resolved_rng,
                    )

    # 4. Wet-Lens Glare / Flare around active lamps
    if config.enable_wet_lens_glare and (resolved_rng.random() < config.wet_lens_glare_prob):
        glare_intensity = resolved_rng.uniform(
            config.glare_intensity_min, config.glare_intensity_max
        )
        augmented = apply_wet_lens_glare(
            augmented,
            record.traffic_lights,
            glare_intensity=glare_intensity,
            rng=resolved_rng,
        )

    # 5. CMOS Sensor Shot Noise & Defocus
    apply_noise = (resolved_rng.random() < config.sensor_noise_prob)
    noise_sigma = resolved_rng.uniform(1.0, config.sensor_noise_sigma_max)
    apply_defocus = (resolved_rng.random() < config.defocus_blur_prob)
    k_size = resolved_rng.choice([3, 5])
    augmented = apply_sensor_noise_and_defocus(
        augmented,
        apply_noise=apply_noise,
        noise_sigma=noise_sigma,
        apply_defocus=apply_defocus,
        kernel_size=k_size,
        rng=resolved_rng,
    )

    return augmented
