"""Event-verifier training machinery (2026-08-26).

The trained verifier backends live in vividly.models (the seam is
vividly.rules.verify); this package holds what trains them: the synthetic
convulsion motion generator (synth), the track-space augmentation that POLICY.md names as
the implementation home for the augmentation carve-out (augment), and the windowing and
feature code shared between training and inference-time feature extraction (features).

Every model this package helps train is an interim prototype under the carve-out in
rule 4 of POLICY.md: prototype weight class, internal demonstration only, never quoted
as clinical detection performance, never deployed commercially without counsel sign-off,
and slated for replacement the moment clinical footage exists (CQ-21 and CQ-22).
"""
