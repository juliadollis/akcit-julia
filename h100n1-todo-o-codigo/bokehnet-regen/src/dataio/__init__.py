"""Contrato de dados: o que uma amostra é, como é codificada e como é dividida."""
from .encoding import (
    DEPTH_LONG_SIDE, EncodedDepth, decode_depth_m, decode_disparity, encode_depth,
    quantization_coc_error_px, resize_depth_nearest,
)
from .sample import (
    FOCUS_SOURCE_TO_MASK_SOURCE, REQUIRED_METADATA_FIELDS, ControlLabel,
    FocusRegionRecord, KSource, MaskSource, Sample, SampleProvenance, SampleRefs,
    validate_metadata,
)
from .split import (
    LeakReport, SceneSplit, build_scene_split, check_no_leak, split_from_source,
)
from .writer import FileSampleWriter, estimate_disk_budget, iter_manifest, read_metadata

__all__ = [
    "DEPTH_LONG_SIDE", "EncodedDepth", "decode_depth_m", "decode_disparity",
    "encode_depth", "quantization_coc_error_px", "resize_depth_nearest",
    "FOCUS_SOURCE_TO_MASK_SOURCE", "REQUIRED_METADATA_FIELDS", "ControlLabel",
    "FocusRegionRecord", "KSource", "MaskSource", "Sample", "SampleProvenance",
    "SampleRefs", "validate_metadata",
    "LeakReport", "SceneSplit", "build_scene_split", "check_no_leak", "split_from_source",
    "FileSampleWriter", "estimate_disk_budget", "iter_manifest", "read_metadata",
]
