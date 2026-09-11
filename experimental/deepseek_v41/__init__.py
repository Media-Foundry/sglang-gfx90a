"""Experimental DeepSeek-V4.1 bring-up helpers.

This package is intentionally not registered with SGLang's model registry yet.
It contains metadata-only and host-memory components that can be exercised while
the upstream checkpoint is still being downloaded.  The production V4 path is
not imported from here.
"""

from .engram_host import (
    EngramHostTable,
    EngramLayerHostTable,
    EngramPrefetchResult,
    EngramPrefetcher,
    EngramRowMapping,
    PinnedStagingPool,
    RawMMapRowStore,
    SafetensorsRowStore,
)
from .metadata import (
    DeepSeekV41MetaConfig,
    SafetensorsTensorHeader,
    audit_checkpoint,
    read_safetensors_header,
)

__all__ = [
    "DeepSeekV41MetaConfig",
    "EngramHostTable",
    "EngramLayerHostTable",
    "EngramPrefetchResult",
    "EngramPrefetcher",
    "EngramRowMapping",
    "PinnedStagingPool",
    "RawMMapRowStore",
    "SafetensorsRowStore",
    "SafetensorsTensorHeader",
    "audit_checkpoint",
    "read_safetensors_header",
]
