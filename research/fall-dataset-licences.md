# Second acted-fall datasets: the licence record behind two admissions (2026-08-27)

Why now: fall v2's split verdict (research/event-verifier-v1-record.md)
measured the rule that negatives must match deployment geometry, and named the
fall program's next lever, positive-side diversity across subjects, scenes and
angles. le2i was then the only fall set granted for training under the interim
prototype carve-out; ur_fall and multicam are the untouchable exams. A second
training set needed an owner decision admitting it under that carve-out (acted
events, prototype-class weights, licence text copied into the registry entry on
admission). The carve-out is rule 4 of POLICY.md. Both candidates below were
admitted on 2026-08-27, and their registry entries are
`configs/datasets/gmdcsa24.yaml` and `configs/datasets/caucafall.yaml`.

Both were licence-checked 2026-08-27 against primary sources before that decision.
Neither carries a research-only or non-commercial clause; the acted nature, not
the licence, is what keeps anything trained on them prototype-class under the
carve-out.

## Candidate A, GMDCSA-24, admitted 2026-08-27

- What: human fall detection videos; 4 subjects, multiple rooms and lighting
  (day and night), falls plus ADLs; per the Data-in-Brief paper (PMC11416611).
- Where: github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
  (Zenodo mirrors: records 11216408 v1.1.0, 12921216 v2.0, versioned and hashable).
- Licence: dual, **CC BY 4.0** on the Zenodo deposit and **MIT** on the GitHub
  repository (repo LICENSE, "Copyright (c) 2024 Ekram Alam"). The MIT grant is
  "to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies"; attribution and inclusion of the licence text are required. Both
  grants permit commercial use and redistribution, and neither carries a
  non-commercial or share-alike clause; the stricter attribution requirement,
  that of CC BY 4.0, is the one followed. Both are recorded word for word in the
  registry entry, with the dual grant carried in its `license_conflict_note`
  (`configs/datasets/gmdcsa24.yaml`).
- Flags for counsel: none in the licence itself. Subjects are actors; the
  paper states consent. Standard prototype-class handling applies regardless.
- Role: training, joining le2i, for subject, scene and lighting diversity. It
  contributes 26 positive and 183 negative windows under `contain` labelling
  (research/fall-noise-floor.md).

## Candidate B, CAUCAFall (CC BY 4.0), admitted 2026-08-27

- What: 10 subjects, 5 fall types plus 5 ADLs, uncontrolled home environment
  (occlusions, lighting changes, background motion), per-frame fall and no-fall
  labels. The record page advertises 1080x960 and the paper 23 fps; every
  container measured on landing is 720x480 at 20.00 fps, so both published
  figures are wrong for this distribution and the measured shape is what the
  registry entry records (`configs/datasets/caucafall.yaml`).
- Where: data.mendeley.com/datasets/7w7fccy7ky/4 (Mendeley Data, versioned).
- Licence: **CC BY 4.0**. Attribution required; commercial use and
  redistribution permitted.
- Flags for counsel: CC BY is a content licence, not a software licence; the
  attribution string must be recorded in the registry entry. Per-frame labels need a
  converter (text files per image).
- Role: training. Both candidates were admitted to training, so neither was held
  out as a third internal test set, and under the carve-out a dataset that trains
  loses its test role permanently. CAUCAFall contributes 1 positive and 130
  negative windows under `contain` labelling (research/fall-noise-floor.md).

## Not proposed, and still not admitted

- OmniFall (HF simplexsigil2/omnifall): unifies 8 staged and wild sources with
  per-source licences. Attractive later, but per-source licence review is a
  bigger lift than one dataset, so it was left for after A and B. A later licence
  review found its annotations released under a non-commercial licence and its
  constituent sources mostly non-commercial, so it was not admitted
  (research/reid-licence-permissive-recent.md).

## What admission produced

Both registry entries under configs/datasets/ carry license_verbatim, the
attribution string and the acquisition URLs, and each records the admitting
decision of 2026-08-27 under the carve-out in POLICY.md, rule 4, which the
repository entry document (CLAUDE.md) names as the single source for the standing
rules. Each annotation format has a converter,
`src/vividly/data/convert/gmdcsa24.py` and
`src/vividly/data/convert/caucafall.py`; the extraction configs are
`configs/events/extract_gmdcsa24_v2c.yaml` and
`configs/events/extract_caucafall_v2c.yaml`; and the fall v3 recipe was
pre-declared at commit b78fc4d, recorded in
research/event-verifier-v1-record.md, section "fall v3 (matched-geometry
positive diet)". The archives were downloaded
only after that decision and are pinned by sha256 in the entries: 9 files for
gmdcsa24 and 1 archive for caucafall, with terms recorded 2026-08-27
(DATASETS.md).
