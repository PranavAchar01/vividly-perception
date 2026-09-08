# Vendored-trainer layer

Nothing in this directory is imported by `src/vividly`, and no third-party code is committed to
this repository. What lives here is the reproducible recipe for materialising the vendored
trainers on a pod:

- `PINS.yaml` records the pinned repositories (URL, commit SHA and licence findings), the
  COCO-lineage checkpoints admitted under the checkpoint-provenance rule, and the prepared
  contingencies that were deliberately not applied.
- `patches/<repo>/*.patch` are unified diffs generated against the pinned commit SHAs, except
  `rtdetrv4/0005`, which is generated against the tree with `0001` already applied: its context
  line `opset_version=17,` is `0001`'s output, so on a pristine clone, at opset 16, it fails,
  and it applies only as part of the series. `git apply --check` on a pristine clone therefore
  verifies the series in order, not each patch alone. The initial series was verified on
  2026-08-31; the three later RT-DETRv4 patches, `0003` through `0005`, were added between
  2026-09-02 and 2026-09-05, each when the upstream defect it fixes was hit on the training
  pod, and `scripts/pod/setup_pod.sh` applies the whole directory as a series. The only
  patch that touches training semantics is the torchvision hook shim for DEIM at torchvision
  0.21 or later, which is a hard blocker; the rest touch export and checkpoint loading only.
  One mechanical caution: a blank diff context line is a lone space, which a whitespace fixer
  would eat and a diff cannot survive. `.pre-commit-config.yaml` therefore excludes
  `^third_party/patches/` from both the trailing-whitespace and the end-of-file hook, and has
  done so since the commit that introduced the patches, so no committed diff has been through
  either. Three of the eight patches carry lone-space context lines; four carry bare empty
  ones instead, which `git apply`, what `scripts/pod/setup_pod.sh` uses, tolerates. Anyone
  regenerating a patch re-runs the `--check` pass after the commit hooks have run, not before.
- `configs/` holds the wrapper configurations for the detector trainers, three training
  configurations and one evaluation configuration, plus one Python recipe,
  `rtmpose_m_person_v2_ft.py`, for the pose trainer. Each carries a provenance card listing
  every delta from the published recipe and the reason for it. Each detector configuration was
  composed through its repository's own `load_config` and its effective values asserted, that
  is schedules, batch size, losses and the teacher state each arm declares, before being
  committed. They reach the vendored trees by different routes:
  `scripts/pod/setup_pod.sh` installs `deim_s_person_v2.yml` and
  `rtv4_s_person_v2_noteacher.yml`, `scripts/pod/eyes_chain.sh` installs
  `rtv4_s_person_v2_dinov2t.yml` in its teacher-fetch stage, `rtmpose_m_person_v2_ft.py` is
  installed by the command TRAINING.md gives for the joints lane, and
  `deim_s_person_v2_zeroshot_eval.yml` is copied by hand as its own header records.

## Guard boundary

These configurations name pod bundle paths rather than repository paths, so the containment
half of the training-data rule is enforced at the point the pod bundle is produced. The files
themselves are still on the guarded surface: `scripts/guards/check_train_data_paths.py` names
`third_party/configs/*.yml` and `*.py` in its surface and refuses any evaluation-only path in
them, and `scripts/guards/check_config_registry.py` resolves each staging path to a registry
entry and applies the bucket rule. The pod layout those paths belong to is described in
[../TRAINING.md](../TRAINING.md).

The bundle comes only from registry-resolved conversions (`scripts/data/convert_smartspaces.py`
and `scripts/data/merge_coco.py`), which run inside the licence buckets with containment checks,
and the merge configurations name train-side scenes only. The SmartSpaces validation scenes are
exam-side: they never enter `person_v2_train.json` or `person_v2_devval.json`, and the
development validation split is Warehouse_005, a train scene held out by scene. The
evaluation-only corpora, that is MVOR, the acted-fall sets and the epilepsy-monitoring
recordings, never leave local machines at all: none of them appears in any pod bundle.

The rule itself is in [../POLICY.md](../POLICY.md): evaluation-only data never enters a training
manifest, and an exam is frozen after inspection and never trains.
