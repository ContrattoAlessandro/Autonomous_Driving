"""Model-building utilities for TLR-YOLO-MTL."""

from .dysample import (
    CARAFE,
    BilinearUpsample,
    DySample,
    register_dysample_modules,
    replace_p2_upsampler,
)
from .local_plus import (
    LocalPlusRelevanceBranch,
    LocalPlusResidualBlock,
    LocalPlusTrafficControlDetect,
    attach_local_plus_relevance_head,
)
from .milestone2 import (
    DEFAULT_CONFIG,
    EXPECTED_STRIDES,
    INPUT_SIZE,
    build_detection_model,
    export_detection_onnx,
    load_coco_warmstart,
    run_forward_smoke,
)
from .roialign_attributes import (
    CandidateAttributeTower,
    CandidateMultiScaleROIAlign,
    CandidateMultiScaleROIAlignPipeline,
    TaskSpecificAttributeTower,
    TaskSpecificGatedROIAlign,
    TaskSpecificROIAlignPipeline,
)
from .geometry_attention import (
    ExplicitRelativeGeometryEncoder,
    GeometryAttentionBiasMLP,
    GeometryAwareCrossAttention,
    GeometryAwareUnifiedDetect,
    attach_geometry_aware_unified_relevance_head,
)
from .neck import (
    ScaleAwareFeatureRelay,
    ScaleAwareFeatureRelayV2,
    ScaleAwareRelayConfig,
    ScaleAwareRelayV2Config,
    register_neck_modules,
)
from .quality import (
    NWDQualityConfidenceHead,
    QualityScoringConfig,
    compute_quality_aware_scores,
    compute_scale_conditioned_alpha,
    compute_scale_conditioned_quality_scores,
)
from .refinement import (
    SparseCandidateRefinementHead,
    SparseRefinementConfig,
    select_dynamic_refinement_budget,
)
from .unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
)

# Ensure custom modules are registered in Ultralytics / PyTorch
register_dysample_modules()
register_neck_modules()

__all__ = [
    "DEFAULT_CONFIG",
    "EXPECTED_STRIDES",
    "INPUT_SIZE",
    "build_detection_model",
    "export_detection_onnx",
    "load_coco_warmstart",
    "run_forward_smoke",
    "UnifiedHeadConfig",
    "UnifiedTrafficControlDetect",
    "attach_unified_relevance_head",
    "LocalPlusResidualBlock",
    "LocalPlusRelevanceBranch",
    "LocalPlusTrafficControlDetect",
    "attach_local_plus_relevance_head",
    "DySample",
    "CARAFE",
    "BilinearUpsample",
    "replace_p2_upsampler",
    "register_dysample_modules",
    "CandidateMultiScaleROIAlign",
    "CandidateAttributeTower",
    "CandidateMultiScaleROIAlignPipeline",
    "TaskSpecificGatedROIAlign",
    "TaskSpecificAttributeTower",
    "TaskSpecificROIAlignPipeline",
    "ExplicitRelativeGeometryEncoder",
    "GeometryAttentionBiasMLP",
    "GeometryAwareCrossAttention",
    "GeometryAwareUnifiedDetect",
    "attach_geometry_aware_unified_relevance_head",
    "SparseCandidateRefinementHead",
    "SparseRefinementConfig",
    "select_dynamic_refinement_budget",
    "NWDQualityConfidenceHead",
    "QualityScoringConfig",
    "compute_quality_aware_scores",
    "compute_scale_conditioned_alpha",
    "compute_scale_conditioned_quality_scores",
    "ScaleAwareFeatureRelay",
    "ScaleAwareFeatureRelayV2",
    "ScaleAwareRelayConfig",
    "ScaleAwareRelayV2Config",
    "register_neck_modules",
]



