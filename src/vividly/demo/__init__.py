"""Vividly demo platform: upload a video, pick a model of record, watch the harness.

Separate from the clinical dashboard (src/vividly/dashboard) on purpose: this package is
the demo-facing face of the models (engineering demo, never a medical claim), reusing the
pipeline runner in-process rather than modifying it. Import discipline matches the
dashboard's: cv2/numpy/onnxruntime only ever inside functions, so the package imports
cleanly on a machine with none of the model stack installed
(tests/test_demo_platform.py enforces this by AST and by subprocess).
"""

from vividly.demo.catalog import infer_onnx_input_size, scan_catalog
from vividly.demo.media import (
    ALLOWED_VIDEO_SUFFIXES,
    AnnotatedWriter,
    annotated_argv,
    ensure_demo_dirs,
    normalize_argv,
    sanitize_filename,
)

__all__ = [
    "ALLOWED_VIDEO_SUFFIXES",
    "AnnotatedWriter",
    "annotated_argv",
    "ensure_demo_dirs",
    "infer_onnx_input_size",
    "normalize_argv",
    "sanitize_filename",
    "scan_catalog",
]
