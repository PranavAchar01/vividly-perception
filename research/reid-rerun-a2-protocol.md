# Re-identification rerun A2: the Phase A recipe on a longer schedule

Dated 2026-09-07. Declared before any training step of the run, so the bars, the looks and
the promotion rule are fixed before the numbers exist (POLICY.md, section 5). The outcome
section at the end is empty until the run artefacts are read back.

## Why a rerun, from the record

The Phase A run of record (`final_models/reid/README.md`, embedder `60164a99`, checkpoint
`66623239`) trained 30 epochs and evaluated at epochs 10, 20 and 30. Its general-protocol
test mAP was 0.137423, 0.156771 and 0.204782 at those points, rank-1 0.389241, 0.439873 and
0.490506, and its training accuracy was 85.45 at epoch 30. The largest gain came after the
learning-rate step at epoch 20, and the curve had not flattened when the schedule ended. The
cheapest lever consistent with that record is a longer schedule with the drop kept at the
same fraction of the run, everything else identical. It is the only lever this rerun pulls.

Not pulled, and why: no metric term, because the two triplet-engine runs on record both
collapsed to a constant feature; no augmentation, resolution, batch or sampler change,
because none of them is implicated by the record; no data change, because MEVID is the only
admitted corpus and the entry is unchanged.

## The one arm

`configs/reid/train_mevid_a2_long.yaml` is a clone of `configs/reid/train_mevid_a1.yaml`
with exactly two declared deltas: `epochs` 30 to 60 and `stepsize` 20 to 40. The `out_dir`
and `onnx_name` keys are renamed so nothing overwrites the run of record. Seed 20260907,
the OSNet-x1_0 model, the ImageNet-only initialisation, the softmax recipe, the sampler,
the augmentations and the three evaluation protocols are unchanged.

Provenance of the inputs on the training host, which was rebuilt from nothing for this run:
the three MEVID archives were fetched again from the public bucket named in
`configs/datasets/mevid.yaml` and verified against the three sha256 digests recorded there
before unpacking; the initialisation file `osnet_x1_0_imagenet.pth` was copied from the
development machine and verified at sha256 `fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79`, the digest the run of record
recorded; the library is the pinned clone at `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`,
installed as TRAINING.md gives it.

## Bars, looks and the promotion rule

- **Shipping bar, unchanged.** The external same-clothes floor of 39.0 mAP on the 54-identity,
  113-outfit test set. External, no holder hash. It is reported against either way, and
  not clearing it is the expected outcome of a schedule change alone.
- **Promotion bar, internal.** General-protocol test mAP 0.204782, held by the model of
  record, embedder `60164a99`.
- **Looks.** The trainer evaluates the held-out test split at epochs 10, 20, 30, 40, 50 and
  60 on the general protocol, which gives six curve points, and then computes the three
  protocols from one feature extraction of the final-epoch weights. The final-epoch weights
  are also what the trainer exports; the best-on-test epoch is recorded in the summary but
  selects nothing. One run, one candidate, no second arm.
- **Promotion rule.** The final-epoch weights replace the model of record on
  `final_models/reid/` if, and only if, their general-protocol mAP exceeds 0.204782 and the
  export clears the parity gate (torch against onnxruntime cosine similarity above 0.999).
  Otherwise the run is recorded on the lane page and nothing is promoted. A continuation
  beyond 60 epochs, should the curve still be rising, is a new declaration and not part of
  this one.
- **Budget.** One training run of about one hour of GPU on the training host, at the
  recorded 36 seconds per epoch plus six evaluations, on a balance of about nine dollars.

## Outcome, measured 2026-09-07

**Not promoted.** The final-epoch weights score general-protocol test mAP 0.188552 against
the promotion bar of 0.204782, so the rule above keeps the model of record, embedder
`60164a99`. The parity gate passed (cosine similarity 1.000000), so the graph is usable; it
is simply not better.

Run identity: trained on the rebuilt training host from `configs/reid/train_mevid_a2_long.yaml`,
seed 20260907, ImageNet initialisation `fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79`, corpus archives verified against the three
registry digests before unpacking. Training loop 34 minutes 45 seconds for 60 epochs, about 35
seconds per epoch, chain start to chain end 03:25:32Z to 04:03:14Z on 2026-09-08. Final
checkpoint `model-best.pth.tar`, sha256 `78ba6b55f2884ed04855e7ccda389fa82069787088c88c3478b7b37ccfa64145`, 36,233,078 bytes, epoch 60, byte-identical to
the epoch-60 file, so the best-on-test epoch and the final epoch coincide. Exported graph
`osnet_x1_0_mevid_a2_long.onnx`, sha256 `151eda30c44fe61bf3c0abf2f25c0feb99c2b10270629595593eacfad7a0e5d8`, 8,729,490 bytes, opset 17, 512-dimensional,
parity cosine minimum 1.000000. Curve file `final_models/reid/curves/reid_mevid_a2_long.csv`.

The six looks, general protocol, embedder `151eda30`:

| epoch | cross-entropy | train accuracy | learning rate | test mAP | test rank-1 |
|---|---|---|---|---|---|
| 10 | 2.4224 | 55.72 | 0.0015 | 0.137423 | 0.389241 |
| 20 | 1.9821 | 68.39 | 0.0015 | 0.156771 | 0.439873 |
| 30 | 1.7983 | 74.05 | 0.0015 | 0.130376 | 0.329114 |
| 40 | 1.6966 | 76.92 | 0.0015 | 0.130791 | 0.360759 |
| 50 | 1.2876 | 89.34 | 0.00015 | 0.183268 | 0.455696 |
| 60 | 1.2239 | 91.10 | 0.00015 | 0.188552 | 0.458861 |

The three protocols at epoch 60, one feature extraction, embedder `151eda30`, with the run of
record's values beside them for the same protocol:

| protocol | rerun A2 mAP | rerun A2 rank-1 | run of record mAP | run of record rank-1 |
|---|---|---|---|---|
| General, euclidean on raw features | 18.86 | 45.89 | 20.48 | 49.05 |
| Normalised cosine | 22.23 | 46.52 | 23.83 | 49.05 |
| Camera debias | 34.25 | 54.11 | 34.89 | 52.85 |
| Same-clothes approximation, in-house | 27.24 | 49.31 | 29.31 | 51.74 |

Reading. The epoch-10 and epoch-20 values reproduce the run of record's to six decimals,
which is the two runs being identical up to the point where their schedules diverge, and is
the first exact reproduction this lane has recorded. From there the longer high-rate phase
moved the number the wrong way: test mAP fell from 0.156771 at epoch 20 to 0.130 at epochs
30 and 40 while training accuracy kept rising, the step at epoch 40 recovered it to 0.183 and
then 0.189, and the run ended at 91.1 percent training accuracy against the run of record's
85.4 at a lower test mAP. Twenty more epochs at the high rate fit the 104 training identities
more tightly and generalised worse. What the record now holds: a longer schedule is not the
lever; the value of the schedule is in when the rate drops, not in how long it runs. An
earlier drop, a cosine schedule or more identities are the levers left, and each would be a
new declaration. The external same-clothes floor of 39.0 mAP remains unreached under the
general protocol, as expected.
