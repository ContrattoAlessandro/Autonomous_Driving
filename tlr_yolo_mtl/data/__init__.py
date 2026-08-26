"""Unified annotations, converters, split logic, and quality assurance."""

from .counterfactual_sampling import (
    DEFAULT_COUNTERFACTUAL_CONFIG,
    CounterfactualMiningConfig,
    CounterfactualPairType,
    CounterfactualRelevancePair,
    CounterfactualRelevanceSampler,
    encode_counterfactual_relevance_targets,
    mine_scene_counterfactual_pairs,
)
from .photometric_augmentation import (
    DEFAULT_PHOTOMETRIC_CONFIG,
    PhotometricAugmentationConfig,
    apply_exposure_and_gamma,
    apply_physics_photometric_augmentation,
    apply_sensor_noise_and_defocus,
    apply_wet_lens_glare,
    estimate_lamp_center,
    synthesize_lamp_bloom,
)
from .schema import (
    IgnoreRegion,
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
    TrafficLightAnnotation,
)

__all__ = [
    "DEFAULT_COUNTERFACTUAL_CONFIG",
    "CounterfactualMiningConfig",
    "CounterfactualPairType",
    "CounterfactualRelevancePair",
    "CounterfactualRelevanceSampler",
    "encode_counterfactual_relevance_targets",
    "mine_scene_counterfactual_pairs",
    "DEFAULT_PHOTOMETRIC_CONFIG",
    "PhotometricAugmentationConfig",
    "apply_exposure_and_gamma",
    "apply_physics_photometric_augmentation",
    "apply_sensor_noise_and_defocus",
    "apply_wet_lens_glare",
    "estimate_lamp_center",
    "synthesize_lamp_bloom",
    "IgnoreRegion",
    "ImageRecord",
    "RoadArrowAnnotation",
    "TaskValidity",
    "TrafficLightAnnotation",
]


