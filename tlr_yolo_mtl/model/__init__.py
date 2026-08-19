"""Model-building utilities for TLR-YOLO-MTL."""

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
from .unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
)

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
]

