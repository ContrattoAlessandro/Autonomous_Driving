"""TLR-YOLO-MTL research implementation.

The first implementation milestone is the loss-aware unified dataset.  Model
code is intentionally built on top of this explicit contract so that a missing
annotation can never silently become a negative target.
"""

from .data.schema import ImageRecord, SCHEMA_VERSION

__all__ = ["ImageRecord", "SCHEMA_VERSION"]

