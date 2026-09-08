# Licence allowlist

Recorded exceptions to the shippable-licence rule, one entry per package, each scoped to the
fork it lives in. Nothing here is an opinion about what counsel will conclude; it is a record of
what is installed, where, and why the exception is believed to be safe.

Scope of this document: the two packages below, both of which arrive transitively with
`fiftyone` in the `curate` fork, plus the standing weight-source rules for re-identification. It
is deliberately not a full transitive licence audit. Classifying all 200 or more locked
distributions needs installed package metadata or per-fork package-index calls, which is a
separate job with its own tooling; this file is not a record of that job having been done.

What is mechanically enforced today: `scripts/guards/check_face_deps.py` fails on any
face-biometric distribution in `pyproject.toml` or in `uv.lock`, transitive arrivals included,
and `scripts/guards/check_agpl_imports.py` plus the `uv` extra conflicts keep AGPL code out of
every shipped path. Both rules are stated in [../POLICY.md](../POLICY.md).

## fiftyone-db 1.4.1, Server Side Public Licence (SSPL) v1

`fiftyone-db` bundles a MongoDB build, and MongoDB is SSPL, which is not an OSI-approved open
source licence and is not shippable.

Why the exception holds:

- **Curate fork only.** It arrives under `fiftyone>=1.20,<2`, declared in the `curate` extra in
  `pyproject.toml`, and `curate` conflicts by declaration with both `cuda` and `eval-agpl`. It is
  installed only into `.venv-curate`
  (`UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate --no-default-groups`).
- **Development tool, never a product component.** FiftyOne is used to look at datasets and to
  compute curation statistics. No `src/vividly/` module imports `fiftyone`, so nothing in the
  training, evaluation or replay-pipeline path can pull it in.
- **Never shipped and never network-served.** The SSPL's operative condition, section 13, bites
  when the licensed program is offered as a service to third parties. The embedded MongoDB runs
  on the developer's own machine, backing a local FiftyOne session. It is not redistributed in
  any artefact produced here and is not exposed to any third party.
- **Not in any shipped artefact.** Checkpoints, ONNX exports, metric JSON and the replay
  dashboard carry none of it.

Conditions on the exception: if FiftyOne is ever hosted for someone outside this project, or if
any part of it is bundled into something handed to a customer, this exception no longer applies
and the question goes to counsel before the change lands.

## universal-analytics-python3 1.1.1, telemetry rather than a licence problem

Logged here because it is a data-egress question rather than a licence one. It is FiftyOne's
Google Analytics client and it is what makes FiftyOne report usage.

Required mitigation: set `FIFTYONE_DO_NOT_TRACK=1` in the environment wherever the `curate` fork
is synced or run, so no usage events leave the machine. This is a standing condition on using
the curate fork, not a one-time step, and it matters beyond preference: the machines that run
curation also hold dataset paths and filenames.

Nothing in `src/vividly/` imports it, and it cannot reach any other fork, since `curate` cannot
be co-installed with `cuda` or `eval-agpl`.

## Re-identification weight sources

Standing rules for every appearance-embedding weight file, adopted on 2026-08-07 and recorded
here because the traps are licence traps. The evidence sits in the re-identification licence
reports under `research/`, summarised in `research/reid-data-licences.md`.

- **LUPerson-derived weights are never shippable and are benchmarking material only.** The
  LUPerson repository states: "LUPerson can only be used for research, commercial usage
  is forbidden." That covers SOLIDER, whose repository confirms "We use LUPerson as our training
  data", TransReID-SSL, LUPerson-NL descendants, and any weight whose pretraining lineage passes
  through LUPerson. Any such weight carries `shippable: "no"` in its provenance record and its
  model card, and `assert_shippable()` refuses it. `scripts/guards/check_face_deps.py` has no
  role here; the enforcement is the provenance layer plus the fetcher denylist under
  `scripts/guards/`, both built on 2026-08-07 alongside the rules recorded here.
- **OSNet weights come only from the canonical MIT torchreid distribution**
  (KaiyangZhou/deep-person-reid), ImageNet-pretrained variant preferred. The kadirnar/osnet
  Hugging Face repack is labelled GPL-3.0 and is banned outright: same bytes or not, its licence
  labelling is wrong for a shipped path, and pulling from it would launder that ambiguity into
  this tree. Market-1501-trained OSNet variants are additionally subject to Market-1501's own
  research terms and hold an evaluation-only posture until counsel says otherwise.
- **torchreid itself, which is MIT, may live in the `curate` fork only**, plus a by-hand install
  from the pinned clone on the Linux training pod that [../TRAINING.md](../TRAINING.md) records.
  The macOS development fork stays free of torch; development-side embedding inference goes
  through an ONNX export over onnxruntime, the same pattern rtmlib set.

The refusal messages that quote this document are in `src/vividly/models/reid_backend.py` at
load time and `scripts/reid/export_osnet_onnx.py` at export time, and
`scripts/guards/check_reid_weight_sources.py` scans the tree for banned sources.
