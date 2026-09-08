"""Session-scoped re-identification: the session gallery, its metrics and its calibration.

Session-scoped is the whole product claim: an identity lives for one visit, is never
persisted across visits, and is never a cross-site identifier.

What may live in this package. Appearance embeddings over whole-person crops, colour and
texture, produced by `vividly.models.reid_backend`, and seat-position priors. That is the
entire signal inventory, and it is deliberate:

  * No face detection, no face embedding, no face-localisation branch, ever. The
    load-bearing element of the independent claims in Philips US 11,157,726 is exactly a
    face-location output combined with non-facial features.
  * No gait, no pose-sequence identity signal, no body-shape or anthropometric identity
    signal. Those are gated on a freedom-to-operate opinion and on biometric law (CQ-12,
    CQ-13, CQ-18; gait is expressly enumerated under the Washington My Health My Data Act),
    and they stay out of the codebase until counsel clears them.
  * No vital signs of any kind.

Headline metric. The false-merge rate, not rank-1. A false merge joins two patients'
identities, and in a chart-bound deployment that is a patient-safety event. Every
session-gallery table in this repository reports it first; the embedder lane page
(final_models/reid/README.md) reports the corpus retrieval protocols instead.

Modules.

  * `vividly.reid.synth` builds a track-cut fixture out of arithmetic: identity centroids
    on a unit hypersphere, gap tiers, gap distractors, new identities for the open-set leg,
    and MOT ground truth. No footage, no weights, no GPU, no network.
  * `vividly.reid.metrics` is the open-set scoring layer (rank-1, detection and
    identification rate at a false-accept rate, false-merge rate) over cosine similarity.
  * `vividly.reid.gallery` is the session-scoped open-set gallery: it scores a probe
    against the people seen this visit and proposes, never binds.
  * `vividly.reid.stage` wires that gallery into the replay pipeline: it enrols reaped
    tracks, probes new ones, and writes a pending proposal for a person to answer.
  * `vividly.reid.calibrate` chooses the gallery's tau at a stated false-accept budget and
    records the evidence in the same file as the number.

Nothing here decides anything about a real person, and no number produced from
`vividly.reid.synth` is evidence about real re-identification performance. Read that
module's docstring before quoting anything it produces.

This package exports no names of its own on purpose: import from the module that owns the
symbol (`from vividly.reid.synth import build_scenario`) so a partially-built package can
never turn one missing module into an import error for every other one.
"""
