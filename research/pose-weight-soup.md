# Pose v2 weight soup: built, hashed, unscored

Dated 2026-09-05. Uniform 3-seed weight soup of the v2 RTMPose-m person fine-tune,
built on the Mac from the pod backups. **No accuracy number exists for this file and
none may be quoted** (POLICY.md, rule 9); see the section on why it is unscored below.

## The four digests

| role | file | sha256 | bytes |
|---|---|---|---|
| ingredient P0 (seed 0, e38, **model of record**, AP 0.774115) | `outputs/pod_backup/p0_best_e38_a0a035fe.pth` | `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102` | 112,155,925 |
| ingredient P1 (seed 1, e30, AP 0.774000) | `outputs/pod_backup/p1_best_e30_8d990435.pth` | `8d99043597076a672e7f4bc1ce886111d06a62de287d4801e489080561fa40ae` | 111,528,021 |
| ingredient P2 (seed 2, e32, AP 0.774057) | `outputs/pod_backup/p2_best_e32_5bcca5bd.pth` | `5bcca5bd3a33c18fbba70db89a6d9d8f9c2703679b3c820dc267090546cf5f07` | 111,679,253 |
| **soup (this artefact)** | `outputs/pose_soup_v2/pose_soup_v2_3seed.pth` | **`46e6a8e87ea0f8e0b6a5ff370ba747f29cc669f40a7f645b4d05c43d16d85966`** | 54,579,707 |

The three ingredient digests match `research/pose-seed-floor-and-champion.md` exactly. The
per-seed APs in the table are the trainer-side COCO-val gt-bbox numbers from that verdict,
reproduced here only to identify the ingredients; they are not properties of the soup.

Built by the command below, in the `curate` environment
(`UV_PROJECT_ENVIRONMENT=.venv-curate`). The script imports torch at module scope and the
default `uv sync` environment installs no torch, so the block does not run under the default
harness; TRAINING.md carries the same invocation with the same note.

    uv run python scripts/pose/soup_checkpoints.py \
        --ckpt outputs/pod_backup/p0_best_e38_a0a035fe.pth \
        --ckpt outputs/pod_backup/p1_best_e30_8d990435.pth \
        --ckpt outputs/pod_backup/p2_best_e32_5bcca5bd.pth \
        --out  outputs/pose_soup_v2/pose_soup_v2_3seed.pth

Sidecar with the full record: `outputs/pose_soup_v2/pose_soup_v2_3seed.pth.sidecar.json`.
Both the `.pth` and the sidecar are gitignored (`/outputs/`); this doc is the committed
record of the digests.

## What went in, exactly

- **362 `state_dict` keys**, key sets / shapes / dtypes identical across all three seeds
  (checked, not assumed: the script exits 4 on any difference). **305** float tensors
  uniformly averaged in float64 and cast back to float32; **57** integer
  `num_batches_tracked` buffers copied from ingredient 0 (averaging a step counter is
  meaningless and `torch.mean` on int64 raises).
- Verification after the write: worst `|soup - mean(P0,P1,P2)|` over 40 sampled float
  tensors = **5.09e-06**, i.e. float32 round-trip only; all 57 integer buffers are
  byte-equal to ingredient 0.
- **`state_dict` is the EMA-swapped weight set**, which is what the soup wants.
  mmengine's `EMAHook._swap_ema_state_dict` swaps `state_dict` with
  `ema_state_dict['module.*']` in `before_save_checkpoint`, so after the swap the file's
  `state_dict` holds the EMA weights (the ones `best_coco_AP_*` was selected on and the
  ones `scripts/pose/export_rtmpose_onnx.py` exported) while `ema_state_dict` holds the
  raw online weights. Souping `ema_state_dict` would have averaged the wrong half; the
  script's `--source` makes the choice explicit and defaults to `state_dict`.
- **Written:** `{'meta': <ingredient 0's meta, copied unchanged>, 'state_dict': <soup>}`.
  **Dropped:** `message_hub` (top-level in these files, per-run mmengine logging state,
  and the only unpicklable part) and `ema_state_dict` (one seed's online weights are not
  a property of the soup). No optimizer state: a soup has no training trajectory.

### Deviation from the expectation, recorded

A key count of about 664 `state_dict` keys was expected for this architecture. The measured
count is **362** on all three checkpoints (`len(ckpt['state_dict'])`, key sets identical).
305 of those are float
tensors; `ema_state_dict` has 363 entries (the same 362 under a `module.` prefix, plus
`steps`). Nothing in this repository records a 664 for this architecture, so the expectation
appears to have come from another lane: the v2 **detector** checkpoints carry 794 keys
(`scripts/detector/soup_v2.py`, measured 2026-09-05). Recording the discrepancy rather
than silently matching it.

## How it was read without mmengine

The macOS `curate` environment supplies torch and carries **no mmengine, and there will not
be one** (mmengine belongs to the training environments; the mmengine of record,
onedl-mmengine 0.10.12, is installed only in the vendored pose trainer environment). The
default `uv sync` environment carries no torch at all. The pose checkpoints pickle live
framework objects, `mmengine.logging.history_buffer.HistoryBuffer` instances inside the
top-level
`message_hub`, so the default unpickler dies before it reaches a single weight.

`scripts/pose/soup_checkpoints.py` resolves every `mmengine.*` global to one inert stub
class and prints which globals it faked (exactly one here:
`mmengine.logging.history_buffer.HistoryBuffer`). Two details that are not obvious and
cost a debug cycle each:

1. `weights_only=False` is **required**. torch at or above 2.6 defaults to a restricted unpickler
   that ignores `pickle_module` entirely, so the stub would never be consulted. The trust
   boundary here is the sha256 of the pod's own output, not the unpickler.
2. The stub needs a **metaclass `__getattr__`**. The pickle stream contains
   `getattr(HistoryBuffer, 'min')` (the statistics methods HistoryBuffer stores by
   reference), and a plain stub class raises `AttributeError: type object 'Stub' has no
   attribute 'min'` there.

Nothing stubbed is ever read: every stub lives in `message_hub`, which the soup drops.

### The filename cannot carry the digest

Measured: `torch.save` writes a zip whose internal archive prefix is the output file's own
basename, so the identical soup written to `pose_soup_v2_3seed.pth` hashes to
`46e6a8e8…` and written to `pose_soup_v2_3seed_46e6a8e8.pth` hashes to `5dc02ce9…`.
Naming a `.pth` after its own digest therefore invalidates that digest, so the
sha8-in-the-filename convention the ONNX exports use does not transfer. The name here is
digest-free on purpose and the digest lives in this table, in the sidecar and on
`final_models/joints/README.md`, in the Full digests table. Re-running with the same `--out`
reproduces `46e6a8e8…` byte for byte.

## Why it is unscored, and what would score it

**Unscored on this Mac.** Scoring an RTMPose `.pth` needs mmpose + mmcv + mmengine on a
CUDA box (top-down COCO-val gt-bbox protocol, `tools/test.py`); none of that exists here
and the v2 lane deliberately does not put it here. Nothing in this document is an accuracy
claim about the soup, and no number should be attached to `46e6a8e8…` until a run
produces one.

The look is **pre-declared, single, and pod-side**, on the same discipline the detector
lane's soup rider uses (`research/eyes-instrument-protocol.md` I3 item 2):

- **Bar:** the recorded pose bar is **0.7742**, the model of record's 0.774115 plus two sigma, where the 3-seed
  noise floor is 0.77406 plus or minus 0.00006 (`research/pose-seed-floor-and-champion.md`). A soup that
  lands below the bar is recorded as **no result**, not as a rival, and `a0a035fe` remains
  the model of record.
- **Discrepancy resolved (2026-09-05):** an earlier bar of 0.7747 was a transcription error
  and appears nowhere in this repository. The bar of record is **0.7742** (the model of
  record plus two sigma,
  research/pose-seed-floor-and-champion.md), fixed here while the soup is still unscored
  on every machine, so the bar precedes the measurement as required.
- **Cost:** one `--test-only` pass, seconds of GPU time. It is a rider on the next pod
  session, not a session of its own.

Sketch of the pod-side command (unrun, so it is a sketch, not a recipe):

    .venv/bin/python "$MMPOSE_TOOLS/test.py" \
        third_party/configs/rtmpose_m_person_v2_ft.py \
        outputs/pose_soup_v2/pose_soup_v2_3seed.pth

## What the ingredients actually disagree about

Worth recording before a soup result is read: the three seeds are near-identical in
accuracy (spread 0.000115) but **not** near-identical in weights. The largest per-tensor
disagreement between P0 and P1 is in the batch-norm running statistics:
`backbone.stage1.0.bn.running_var` differs by up to 59.6 and `backbone.stage4.1.conv2.bn.
running_mean` by 38.3, while the learned convolution weights agree far more closely.
A uniform soup averages those BN statistics along with everything else, which is standard
model-soup practice and is what the detector lane does too; it is also the most likely
place for a soup to lose accuracy without any single weight looking wrong.

## Files

- `scripts/pose/soup_checkpoints.py`: the instrument (stub unpickler, family guard on
  recorded sha256, structural guard, uniform average, sidecar).
- `outputs/pose_soup_v2/pose_soup_v2_3seed.pth` (plus `.sidecar.json`): the artefact,
  gitignored.
- `final_models/joints/README.md`, section Identity and provenance: the model of record
  ingredient's provenance.
