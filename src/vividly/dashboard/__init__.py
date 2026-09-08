"""Local review dashboard for replay mode.

One FastAPI app over one PipelineState: a frame, the people currently tracked, their
waiting timers, the alert feed with its two feedback buttons, the manual staff tag, and a
clearly labelled MOCK chart binding. It runs on loopback, next to the pipeline that feeds
it, and it is the only review surface in replay mode.

What it is not. Not patient-facing, not a deployment, not an integration. The chart
binding is a string built from a tracker id (see :func:`~vividly.dashboard.app.mock_mrn`)
and the page prints MOCK beside every one of them; there is no record system behind it. No
face detection, embedding, matching or localisation of any kind (POLICY.md, rule 2), and no
vital-signs extraction of any kind (POLICY.md, rule 3). Staff tagging is a button a reviewer
presses, not a classifier.

Re-association is proposed here and is never performed here. Inside the tracker's own short
re-association window association stays automatic and untouched. Beyond it this page shows
a pending proposal with a confidence and two buttons, and no route in this package binds an
identity by itself; both answers are written as labels through the versioned alert store,
and a confirmation reaches the pipeline's merge hook only when a person selects it. The
gallery those proposals come from is session-scoped and is deleted at session end
(CQ-16-session-scoped-retention in docs/COUNSEL_QUESTIONS.md), while the cards outlive it
because the rows hold no embedding, no crop and no appearance vector. See POLICY.md
section 4, re-identification scope, and src/vividly/reid/gallery.py.

Import discipline. fastapi, uvicorn and websockets are pure python and cross-platform, so
this package imports on the Linux CI runner where numpy and cv2 are absent. cv2 appears
only inside the body of :func:`~vividly.dashboard.app.encode_jpeg_b64`, and the one numpy
annotation is a string under TYPE_CHECKING. tests/test_dashboard.py pins that at the AST
level.

Thresholds shown on the page come from configs/pipeline/*.yaml and every one of them is
PROVISIONAL until both of these exist: fine-tuned weights for the lane whose outputs feed
the rule, and pilot shadow-mode data. No count on that page is a false-alarm rate. See
docs/CALIBRATION.md.
"""

from __future__ import annotations

from vividly.dashboard.app import (
    BUSY_RETRY_AFTER_S,
    CROPS_DIRNAME,
    MAX_CROP_BYTES,
    MAX_VIA_CHARS,
    MOCK_LABEL,
    PAGE_HTML,
    STAGES,
    PipelineState,
    create_app,
    crop_name,
    encode_jpeg_b64,
    find_crop,
    identity_color,
    is_servable_crop_name,
    mock_mrn,
    resolve_crop_path,
)

__all__ = [
    "BUSY_RETRY_AFTER_S",
    "CROPS_DIRNAME",
    "MAX_CROP_BYTES",
    "MAX_VIA_CHARS",
    "MOCK_LABEL",
    "PAGE_HTML",
    "STAGES",
    "PipelineState",
    "create_app",
    "crop_name",
    "encode_jpeg_b64",
    "find_crop",
    "identity_color",
    "is_servable_crop_name",
    "mock_mrn",
    "resolve_crop_path",
]
