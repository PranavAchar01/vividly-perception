# Seizure push 3, pre-declared protocol

Dated 2026-09-05. Seizure push 3 is cleared to proceed, as recorded at commit `a522f6a`.
This satisfies the closing clause of `research/seizure-push-2-protocol.md` item 4-H2h, which
requires a harmonized-diet training arm to be a newly declared push rather than a
continuation, and of `research/seizure-ajile12-negative-control.md`, the Reading section item 4, which
requires any re-cut of the diet with WU-like fingerprint channels to be a new pre-declared
step. **Written before any run and before any exam look.**

Standing framing unchanged: interim prototype under the event-model carve-out in POLICY.md
rule 4, motion signature only, never quoted as clinical detection performance, and the claim
that it detects seizures is never made.

**Stated up front, so the push cannot be read as more than it is.** Push 2's H2h control
measured **+0.02450** of mean val AUC for the harmonized AJILE12 negative diet over three
seeds on one fold, report-only. That is a val-side, single-fold, three-seed number. Push 3
asks exactly one question, *does it survive as a training arm and produce a candidate worth
an irreversible exam look?*, and it is the last question this lever gets at the data
layer. What push 3 does **not** promise: an E2 (convulsive) crown change, a false-alarm
rate, or any statement that the AJILE12 dose is shortcut-free. The push-1 evidence that
0.95 convulsive AUC is out of reach on 5 PG patients stands, and rotation val does not
track E2 (arm A/B/C val 0.798/0.801/0.836 against E2 0.581/0.575/0.510).

## Exams and bars (unchanged from pushes 1 and 2)

- **E1 focal** (pat06/08/10/12, the permanent exam-side allocation): bar = final B
  `b9e04d03` window AUC **0.87422**
  (1,430 windows / 177 positive, `outputs/verifier/conv_finalB/eval_E1`, under `outputs/`,
  which is not in version control, git `bc61cab`).
- **E2 convulsive** (pat03 + pat07, locked): bar = the model of record `07c399be` window AUC
  **0.72916** (919 windows / 70 positive,
  `outputs/verifier/conv_champion_reslice/eval_E2`, git `bc61cab`).
  Context on the same surface: final B `b9e04d03` scored **0.57469**.
- Selection never touches E1 or E2. **Budget: one promoted candidate, one E1 look and one E2
  look, recorded in the ledger below.** Nothing else looks at an exam. Push 2's own
  two-candidate budget went entirely unspent and does not carry over; this push declares
  its own.

## The one variable

Push 3 is the push-2 **final recipe** (`configs/events/rot_train_conv_finalB.yaml`,
whose parsed config is identical to `rot_train_conv_rotB_f2_s1.yaml` in every key except
`out_dir`, verified 2026-09-05, with only the comment headers differing) with **one field
changed**: the AJILE12 negative-diet source directory,

    outputs/events/ajile12_neg_tracks_v1  ->  outputs/events/ajile12_neg_tracks_v1_h2h

Everything else is pinned unchanged: `window` (12.0 s / stride 3.0 / 64 steps / policy
`overlap` / `overlap_min_s` 4.0 / `min_present` 0.35 / `scene_channels` true), the eight
rotation-pool WU patients, `holdout_fraction` 0.2, `holdout_locations` [pat02, pat09],
model 64/3/0.15, 60 epochs, batch 256, lr 1e-3, wd 1e-4, `pos_weight: auto` (resolves to
`min(10.0, 6975/387) = 10.0`), device cpu.

**Explicitly off, and stated because push 2 measured all three and promoted none:**

| lever | push-3 status | why |
|---|---|---|
| track-space augmentation (`augment:`) | **absent** | push-2 item 3 moved the 4x3 grid by +0.00129 against a 0.03075 bar and was louder out of fold (+18% at or above 0.5, +64% at or above 0.8). Push-2's final recipe carries no `augment:` key at all (verified in `conv_finalB` and `conv_rotB/f2_s1` configs, which predate the key; the four H2h runs, which postdate it, record `augment: null`), so keeping it off is what makes the diet the only variable. |
| hard-negative class weights (`negative_class_weights:`) | **absent** | push-2 item 2 steps 2-3 failed the F2 gate at both alphas and tripled the seed spread. |
| camera-geometry augmentation (`geo_augment:`) | **absent** | fall-lane arm F3 only; never declared for the seizure lane. |

One variable per push. A config carrying any of the three keys is not a push-3 run.

## Data rules restated (nothing here is new; all three are load-bearing)

- **Permanent exam-side allocation.** WU-SAHZU pat06/08/10/12 are exam-side permanently.
  They appear in no push-3 training source. The training `sources` list is
  pat01/02/04/05/09/11/13/14 and nothing
  else; pat03/pat07 (E2) are likewise absent. Verified against every push-3 parent config.
- **Diet rule.** AJILE12 negatives come only from spans positively labelled as normal
  activity (`scripts/data/assemble_ajile12_diet.py`, and the diet rule in
  `configs/datasets/ajile12.yaml`: blocklist spans and all unlabelled time are excluded, and
  an everything-minus-seizures cut is barred).
  Patients' seizure windows never become positives. **The harmonized tree inherits this
  unchanged and was verified, not assumed:** it is a per-row copy of the assembled tree, with
  the same 61 segment keys, the same 61 meta keys, `has_seizure: false`, `events_t: []`, and
  the original `provenance.diet_rule` preserved on every meta, with the harmonization recorded
  additively under `provenance.h2h_control`. Only `kpc` and `angle` differ per row.
- **Augmentation carve-out.** It exists; push 3 does not use it (see the table above).

## Scope: this is a fold-F2 push, and that is a measured constraint, not a shortcut

Push 2's augmentation arm was judged on a full 4-fold by 3-seed grid. **Push 3 cannot be, on
this tree, and the reason is arithmetic.** H2h built the harmonized diet by transplanting the
`(kpc, angle)` series of real WU tracks drawn from the fold-F2 **train** patients only, so
that no F2 val-side statistic could reach a train row. Counting the donors actually used
(`outputs/events/ajile12_neg_tracks_v1_h2h/h2h_harmonization.json`, 61 assignments,
218,746 rewritten rows):

| donor patient | segments | rows rewritten | share | held out by |
|---|---:|---:|---:|---|
| pat04 | 42 | 150,444 | 68.8% | **fold F1** |
| pat01 | 12 | 43,177 | 19.7% | **fold F1** |
| pat05 | 3 | 10,774 | 4.9% | fold F3 |
| pat11 | 3 | 10,751 | 4.9% | fold F3 |
| pat14 | 1 | 3,600 | 1.6% | fold F4 |
| pat02, pat09 | 0 | 0 | 0.0% | **fold F2, clean by construction** |

Training fold F1 on this tree would put **88.5% of the negative diet's confidence and posture
texture** in the hands of the two patients F1 holds out; F3 sits at 9.8% and F4 at 1.6%.
Only F2 is clean. A legitimate 4-fold grid therefore needs **four** per-fold harmonized
trees, which needs the harmonizer promoted out of scratch into `scripts/data/`, a code
change, and the separate declared step that `research/seizure-ajile12-negative-control.md`
names in its Artefacts section. It is not this push.

That restriction costs less than it looks like, because **fold F2's holdout is the final-B
selection surface.** `rot_train_conv_finalB.yaml` selects its checkpoint on
`holdout_locations: [pat02, pat09]`. Judging push 3 on that surface uses the same ruler the
model of record was chosen on, not a switched ruler.

## Arms

**(a) Baseline, with no retrain of the model of record required.** Final B `b9e04d03`
exists complete on disk with both exam looks already taken:
`outputs/verifier/conv_finalB/` (`report.json` val_auc_best 0.75479, onnx
`b9e04d03…9552aecc`, ckpt `535141c1`, git `bc61cab`) plus `eval_E1/` and `eval_E2/` with
`report.json`, `metrics_deep.json`, `breakout.json`, `windows.jsonl`. **No exam look is
re-spent on the baseline.** Its bars above are quoted from those recorded artefacts.

The baseline's *selection-surface* arm is re-run, for a reason given in the next section:
five seeds of the unchanged recipe, under the current committed tree.

**(b) Candidate, `p3harm`, the push-2 final recipe with the harmonized diet swapped in.**
Five seeds. **The promoted candidate is fixed here as seed 20260826**, the seed
`rot_train_conv_finalB.yaml` pins, *not* the median-val seed. Choosing the candidate by val
would be a second variable, and it is exactly the kind of freedom that turns a spread-sized
delta into a selected maximum. (Recorded for completeness: at three seeds, 20260826 happened
to also be the median. It is pinned by the recipe regardless of where it lands at five.)

## Why every seed is re-run under HEAD (the regression gate)

The recorded numbers this push builds on were produced under **three different code
states**, and two code generations of drift sit between them and HEAD:

| artefact | code state | git_commit in report.json |
|---|---|---|
| arm-B F2 seeds `b9e04d03` / `12aeb901` / `4e0890a2`, and finalB | `bc61cab` (2026-09-04 21:59) | `bc61cab` |
| H2h baseline + 3 harmonized runs | scratch snapshot of `08aa327` (2026-09-05 02:56) | **null** |
| HEAD | `a522f6a` (2026-09-05 13:35) | not applicable |

Since `08aa327`, commit `7a3d857` (fall F3) changed `scripts/events/train_event_verifier.py`
(+92 lines), `src/vividly/verifier/augment.py` (+367) and, the one that matters,
`src/vividly/verifier/features.py` (+65: `WindowSource`, `with_sources`, `_window_source`).
Inspection says the featurizer change is purely additive and `with_sources` defaults to
False, and fall F3 proved default-off byte-identity **on the fall path**. It has never been
proved on the convulsion path. Push 3 proves it before reading anything.

Re-running all ten seeds costs about six extra minutes and buys three things: one code
state for both arms, a populated `git_commit` on the candidate that may spend an exam look
(POLICY.md, section 5), and the removal of the caveat that H2h ran outside the repository
from any promoted number.

**Regression gate, with pre-declared expected values. A mismatch is a stop-and-report, not a
result:**

| run | expected val_auc_best | expected onnx sha8 |
|---|---:|---|
| `p3base_s1` (seed 20260826) | 0.75479 | `b9e04d03` |
| `p3base_s2` (seed 20260901) | 0.77060 | `12aeb901` |
| `p3base_s3` (seed 20260905) | 0.75569 | `4e0890a2` |
| `p3harm_s1` (seed 20260826) | 0.78876 | `06b43397` |
| `p3harm_s2` (seed 20260901) | 0.78954 | `70afe929` |
| `p3harm_s3` (seed 20260905) | 0.77629 | `dafe0cbf` |

## Seeds and the selection surface

- **Seeds: 20260826, 20260901, 20260905** (H2h's three, and the rotation grid's three),
  **plus 20260910 and 20260915**, the two the fall lane already uses for its five-seed
  control (`train_fall_F0_s20260910.yaml`, `train_fall_F2kp_s20260915.yaml`). Five per arm.
- **Selection surface: the fold-F2 patient holdout, pat02 and pat09, 902 windows, 136
  positive, 766 negative.** Selection statistic: `val_auc_best`, the same statistic arm B,
  final B and every push-2 arm were selected on. Dataset shape is pinned and checked:
  **7,362 windows / 387 positive / 6,975 negative / 20 videos** in both arms, which is the
  figure all four H2h `report.json` files carry, and it holds because the harmonization is
  geometry-preserving (H2h, the section on what the control changes: both trees produce 2,250 AJILE12
  windows under this WindowSpec). Any deviation is a stop-and-report.
- **E1 and E2 are untouched during selection.** No exam patient is built, scored, or named
  in any selection config. The first and only exam contact this push can make is the single
  promoted candidate's two looks, and only after the bar below is cleared.

## The promotion bar (fixed here, before any run)

**Baseline of record on the selection surface**, from the three recorded seeds:

    (0.75479 + 0.77060 + 0.75569) / 3 = 2.28108 / 3 = 0.76036
    seed spread (max - min) = 0.77060 - 0.75479 = 0.01581

**Noise floor.** Two F2 readings are already measured, and per the fall-F3 precedent the
wider floor fixed in advance governs: `research/fall-push-v3-protocol.md` records that the
wider floor was fixed in advance and governs, and that switching rulers after seeing the
number is not available.

    baseline F2 seed spread   = 0.01581, declared 0.016 in research/seizure-push-2-protocol.md item 1
    harmonized F2 seed spread = 0.01325   (H2h)
    floor = max(0.016, 0.01325) = 0.016

### P1, margin (primary, exact)

> **The difference, mean(p3harm, 5 seeds) minus mean(p3base, 5 seeds), must exceed 0.016.**

Reference in absolute terms: if the five-seed baseline mean lands at the recorded
**0.76036**, the harmonized five-seed mean must exceed **0.77636**.

At three seeds the arm sits at a difference of 0.78486 - 0.76036 = **0.02450**, which is **1.53 times the
bar**, that is **+0.00850 of headroom**. The two new seeds have to not erase that.

**The reading that is not adopted, declared now so it cannot be adopted later.** Push 2's
primary bar was the pooled four-fold spread, mean(0.023, 0.016, 0.014, 0.070) = **0.03075**,
which on this fold would demand 0.76036 + 0.03075 = **0.79111**, and H2h's 0.78486 would
**miss it by 0.00625**. That ruler is not used here, for a stated reason: 0.03075 is the
noise scale of a **four-fold grid mean**, and push 3's statistic is a **single-fold** mean.
Applying a grid-scaled bar to a fold-scaled statistic is a ruler switch in the direction
that happens to be harder, which is no more legitimate than one in the easier direction.
The number is recorded so the reader can see the whole picture, and it is why P1 alone does
not promote anything; P2 does the statistical work.

### P2, separation (the actual certificate, exact and distribution-free)

> **All five p3harm seeds must score strictly above all five p3base seeds.**

Under exchangeability of the ten runs, complete separation is 2 of the C(10,5) = 252 equally
likely rank assignments: **exact two-sided p = 2/252 = 0.0079**. This is the Wilcoxon
rank-sum statistic at its extreme, it needs no normality and no variance estimate, and it is
invariant to any monotone rescaling of AUC.

**This is why five seeds and not three.** At 3 against 3 the best attainable two-sided p is
2/C(6,3) = 2/20 = **0.10**, so H2h's reading, that the worst harmonized seed beat the best
baseline seed three times out of three, could never have been a certificate at any
conventional level, no matter how clean it looked.
Five per arm is the smallest design that can reach p < 0.05 on this evidence, and it can
only reach it by total separation, which is exactly the condition being required.

### P3, stability

> **spread(p3harm, 5 seeds) at or below 2 times spread(p3base, 5 seeds).**

Pre-declared because push 2 killed a lever partly on this axis and did so post-hoc. Applied
retroactively the cap has teeth: HN alpha 2 was 0.035/0.016 = 2.2 times, HN alpha 4
0.046/0.016 = 2.9 times, and track-space augmentation 0.034/0.016 = 2.1 times, so all three
of push 2's failed levers would have tripped a twofold cap. H2h's harmonized arm sits at
0.01325/0.01581 = **0.84 times**.

### P4, the not-louder gate (negative side, calibration-free)

The product priority is avoiding false positives ahead of marginal recall, and push 2's
augmentation arm died on exactly this axis after clearing a val gate. So the negative side
is a **gate, not a footnote**, and it is pre-declared.

> On the fold-F2 val surface (766 negative windows), at the threshold that gives each run a
> matched **recall of 0.10, 0.25 and 0.50** on the 136 val positives, the mean false-positive
> count of p3harm over its 5 seeds must be **at or below** that of p3base over its 5 seeds, **at all
> three recall points.**

Matched-recall FP counts are rank-based, so they are invariant to each run's fitted
temperature. That invariance is not a technicality here: the fitted temperatures across
these runs are 1.0251 (baseline s1), 1.7672, 0.4878 and 1.3459 (harmonized s1/s2/s3), so
**any fixed-threshold comparison across the two arms is confounded and cannot be the gate.**
Concretely, from the recorded `val_operating_points`, at a fixed 0.5 the harmonized s1 looks
louder (128 FP against the baseline's 80), but its recall is also higher (0.522 against
0.441), and at a matched **23 FP** the harmonized s1 catches recall 0.213 against the
baseline's 0.132.
The fixed-threshold table will still be reported as context, clearly labelled as
temperature-confounded, and it will never be the gate.

### The verdict rule

**P1 ∧ P2 ∧ P3 ∧ P4.** All four, conjunctively. Any one failing means **not promoted, no
exam look spent, result recorded on this page.** In particular P1 passing while P2 fails
(say 4-of-5 separation) is a fail. There is no tiebreak, no second reading to adjudicate,
and no ruler is changed after a number is seen.

## Exam policy

Only a candidate clearing all four conditions is declared final and spends **one look
on E1 and one look on E2**, recorded in the ledger below and nowhere else.

- **E1** (`b9e04d03` bar **0.87422**): AUC is the whole bar, and that is a stated limitation
  of E1, not a choice, because E1 has **zero negative videos**
  (`metrics_deep.negatives.n_videos = 0`, every `fp_windows_per_hour` null), so its
  `video_fp 0` is vacuous and no negative-side read exists there. Context recorded alongside:
  final B catches 10/12 events @0.5 and 7/12 @0.8.
- **E2** (`07c399be` bar **0.72916**): to take the convulsive crown a candidate must clear
  0.72916 **and** not regress the negative side at the demo operating point. That two-part
  form is how the standing E2 bar is recorded on `final_models/seizure/README.md`. Incumbent
  values on the same 0.184 negative hours (pat03__free): the model of record `07c399be` catches 4/4
  @0.8 with **331.57** FP windows/hr @0.8; final B `b9e04d03` catches 2/4 @0.8 with 206.55.
- Falling short of a bar is recorded, and the look is still spent, because a declared look is
  spent when it is taken, not when it succeeds. Falling short of the *promotion* bar (P1 to
  P4) means no look is taken at all.
- The model record in `final_models/README.md` and the model-of-record entries change **only**
  through a ledger row below. Absent that, the broad seizure-motion head stays final B
  `b9e04d03` and the convulsive head stays the model of record `07c399be`.

## The `scene_n_tracks` caveat, carried forward, still open

Carried unchanged from `research/seizure-ajile12-negative-control.md`, still unclosed:
`scene_n_tracks` is 0.125 on every AJILE12 window (the assembler emits exactly one track per
segment; 1/8 is the channel's normalizer) and only 14 of 4,210 WU windows sit at or below
it. The rule that `scene_n_tracks` at or below 0.125 implies AJILE12 has **precision 0.9938 and recall
1.000** on that pool, a *better* source flag than the `kpc` and `angle_valid` fingerprint
H2h removed ever was. It cannot be removed at the data layer (a ghost track long enough to
raise the density would generate windows of its own; a short one spikes `scene_births`).
Closing it needs a multi-track assembler or a featurizer that can be told a source has no
scene context: a code change, and a separate declared step.

**How it affects push 3's read, stated in both directions:**

1. It **cannot explain a push-3 delta.** `scene_n_tracks` is bit-identical in both arms
   (source AUC 0.0017 in both H2h columns). Whatever shortcut capacity it carries is
   present in the baseline and the candidate equally, so it cancels in the difference. This is the same
   argument H2h made and it survives here unchanged.
2. It **does bound what a win means.** A cleared bar licenses only the statement that the
   harmonized diet trains a better-ranking model on this val surface. It does not license
   any statement that the AJILE12 dose is shortcut-free, and no sentence in the results
   section will say so.
3. On the **exam** surfaces the flag is absent by construction, because E1 and E2 windows are
   all real WU windows, so a model leaning on a flag-implies-negative shortcut simply finds
   no flagged rows there. That is an argument about direction, not a proof, and it is
   recorded as such: it predicts an exam look would not *inherit* the artefact, not that the
   model is clean.

## Compute, run order and interruption tolerance

Mac, CPU only, torch 2.7.1 / numpy 2.4.6 under the default `uv` environment (verified
2026-09-05). **At most 2 heavy jobs concurrently** (standing Mac rule); the run order below
is written serial and is safe either way.

**The trainer has no `--resume` and no mid-run checkpointing** (`train_event_verifier.py`
takes `--config` and nothing else), so resilience is designed at **seed granularity**: each
run is one short process that writes its own `out_dir`, with `report.json` written last, so
a directory containing `report.json` is complete and a directory without one is discardable
and simply re-run. An interruption loses **at most one seed (~3.5 min)**. This matters:
the machine running the push may become unavailable at short notice.

The order is **paired and front-loaded with reproductions**, so any prefix is readable:

| # | run | out_dir | seed | status |
|---|---|---|---|---|
| 1 | `p3base_s1` | `outputs/verifier/conv_p3base/f2_s1` | 20260826 | **regression gate** |
| 2 | `p3harm_s1` | `outputs/verifier/conv_p3harm/f2_s1` | 20260826 | reproduction + **the candidate** |
| 3 | `p3base_s2` | `outputs/verifier/conv_p3base/f2_s2` | 20260901 | reproduction |
| 4 | `p3harm_s2` | `outputs/verifier/conv_p3harm/f2_s2` | 20260901 | reproduction |
| 5 | `p3base_s3` | `outputs/verifier/conv_p3base/f2_s3` | 20260905 | reproduction |
| 6 | `p3harm_s3` | `outputs/verifier/conv_p3harm/f2_s3` | 20260905 | reproduction |
| 7 | `p3base_s4` | `outputs/verifier/conv_p3base/f2_s4` | 20260910 | **new** |
| 8 | `p3harm_s4` | `outputs/verifier/conv_p3harm/f2_s4` | 20260910 | **new** |
| 9 | `p3base_s5` | `outputs/verifier/conv_p3base/f2_s5` | 20260915 | **new** |
| 10 | `p3harm_s5` | `outputs/verifier/conv_p3harm/f2_s5` | 20260915 | **new** |

Runs 1-6 all have pre-declared expected values, so an interruption before run 7 destroys no
new evidence. After every even-numbered run the two arms hold equal seed counts, so any
even prefix is a balanced (if under-powered) read. **Runs 1-6 must all match the regression
table before run 7 starts.** If any does not, the push stops and reports rather than
continuing on a moved floor.

**Wall clock.** Recorded cost of this recipe on this Mac: 75 to 100 s of training per run
(`research/seizure-push-2-arms.md`), and H2h's observed serial cadence was **~3.4 min per
completed run** (03:26:30, 03:29:44, 03:33:32, 03:36:42, that is gaps of 194 / 228 / 190 s)
including the per-run window build over 218,746 AJILE12 rows plus the WU tree. Budgeting at
the conservative cadence rather than the 75 to 100 s training time alone:

- 10 training runs at about 3.4 min each, so **about 34 min serial**, 18 to 20 min at the 2-wide cap.
- P4 scoring pass: 10 evals by 902 windows through a small ONNX **about 3 min**.
- If promoted: 2 exam evals plus `metrics_deep`, **about 5 min**.
- **Total approximately 42 min serial, worst case under 50 min end to end.**

## Exact commands the runs will use

Configs to be created under `configs/events/` with the `rot_` prefix (naming follows push 2's
`rot_train_conv_p2aug_f*_s*.yaml` pattern). Each is a copy of
`configs/events/rot_train_conv_finalB.yaml` with the fewest fields the arm and the seed
require. A `p3base` config changes `out_dir` and `train.seed`, and the seed 20260826 config
changes `out_dir` alone, since `rot_train_conv_finalB.yaml` already pins that seed. A
`p3harm` config changes the same fields plus the second source's `dir`, which is the one
variable of this push, so it changes at most three. No other key differs, and no `augment:`,
`negative_class_weights:` or `geo_augment:` key is present in any of them.

    # Arms (10 runs, in the order of the table above)
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3base_f2_s1.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3harm_f2_s1.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3base_f2_s2.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3harm_f2_s2.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3base_f2_s3.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3harm_f2_s3.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3base_f2_s4.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3harm_f2_s4.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3base_f2_s5.yaml
    uv run python scripts/events/train_event_verifier.py --config configs/events/rot_train_conv_p3harm_f2_s5.yaml

    # P4 gate: score the fold-F2 val surface with the committed eval script, one config
    # per run (sources = wu_sahzu_tracks_v2pose restricted to locations [pat02, pat09],
    # same WindowSpec, model.onnx = that run's verifier.onnx). Each must report
    # n_windows 902 / n_window_pos 136; a deviation means the eval surface is not the
    # trainer's val split and is a stop-and-report. windows.jsonl carries the per-window
    # scores the matched-recall counts are computed from.
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3base_s1.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3harm_s1.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3base_s2.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3harm_s2.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3base_s3.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3harm_s3.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3base_s4.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3harm_s4.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3base_s5.yaml
    uv run python scripts/events/eval_event_verifier.py --config configs/events/rot_valF2_p3harm_s5.yaml

    # Exam looks, only if P1 and P2 and P3 and P4 all hold. Candidate = p3harm seed 20260826.
    # The two exam configs are written only when the looks are actually taken, so they are
    # not carried in the tree; the paragraph below states what they contain. Run
    # eval_event_verifier.py once against each, then:
    uv run python scripts/events/metrics_deep.py --eval-dir outputs/verifier/conv_p3harm/f2_s1/eval_E1 outputs/verifier/conv_p3harm/f2_s1/eval_E2

The two exam configs are clones of `configs/events/rot_eval_E1_finalB.yaml` and
`configs/events/rot_eval_E2_finalB.yaml` with `model.onnx` and `out_dir` repointed at
the candidate; their `sources` blocks (pat06/08/10/12 and pat03/pat07) are copied unchanged
so the exam surface is bit-for-bit the one final B and the model of record were measured on.

## Deliverables

- **Results are appended to this page**, as an items table and the exam-looks ledger below,
  in push-2 style. No separate verdict page unless a look is spent.
- **Every quoted number carries its onnx sha8** (POLICY.md, rule 10); checkpoint sha8s go in
  the results table alongside.
- Artifacts land gitignored under `outputs/verifier/conv_p3base/f2_s*` and
  `outputs/verifier/conv_p3harm/f2_s*` (plus `.../eval_*`), each with `report.json`,
  `verifier.pt`, `verifier.onnx`.
- Per-epoch curves added to `scripts/analysis/extract_training_curves.py` `RUNS` as
  `seizure_p3_{base,harm}_f2_s<seed>` and regenerated into `final_models/seizure/curves/`, matching
  push 2's pattern.
- The model record in `final_models/README.md` is edited **only** if a ledger row below records a bar being
  cleared.

## Preflight record (2026-09-05, before any run, with nothing trained)

| check | result |
|---|---|
| harmonized tree present and resolves | `outputs/events/ajile12_neg_tracks_v1_h2h`, 61 sequences, 61 metas, `h2h_harmonization.json` (218,746 rows, 61 assignments, seed 20260905, 42 distinct donors) |
| harmonized against original tree parity | sequence names identical, meta names identical (`diff` clean both ways) |
| diet rule survives harmonization | every meta keeps `has_seizure: false`, `events_t: []`, `provenance.diet_rule`; h2h record added additively |
| all 4 H2h configs + `rot_train_conv_finalB.yaml` load and every `sources.dir` resolves | pass |
| the one-variable claim is proven, not assumed | parsed `rot_train_conv_finalB.yaml` against `h2h_conv_rotB_f2_s1_harmonized.yaml`: the only differing key (ignoring `out_dir`) is `sources`, and inside it only the AJILE12 `dir`; swapping that one string makes the two `sources` blocks equal |
| H2h run artefacts reproduce the doc exactly | `a_baseline` 0.75479 `b9e04d03`, `b_harmonized` 0.78876 `06b43397`, `_s2` 0.78954 `70afe929`, `_s3` 0.77629 `dafe0cbf`; all four `n_windows` 7362 / 387 pos / 6975 neg / 20 videos; all `augment: null`, `negative_class_weighting: null`, `resolved_pos_weight` 10.0; `git_commit` null on all four (the `08aa327` snapshot) |
| baseline artefacts exist (no retrain of the model of record needed) | `conv_rotB/f2_s{1,2,3}` = 0.75479 `b9e04d03`, 0.77060 `12aeb901`, 0.75569 `4e0890a2` (git `bc61cab`); `conv_finalB` = 0.75479 `b9e04d03` / ckpt `535141c1` with `eval_E1` **0.87422** and `eval_E2` 0.57469; the reslice of the model of record `07c399be` **0.72916** |
| code drift since the H2h snapshot | `7a3d857` touched the trainer, `augment.py` and `features.py` after `08aa327`, so the regression gate above is required, not ceremonial |
| `uv run ruff check .` | All checks passed |
| guards run | `check_agpl_imports`, `check_face_deps`, `check_train_data_paths`, `check_config_registry`, the camera guard that is now `scripts/guards/check_no_fisheye_primitives.py`, and `check_reid_weight_sources`, all reporting OK. (`check_train_data_paths` and `check_config_registry` were scoped at the time to the Python training configurations, not to `configs/events/`, which is all push 3 touches, so they are a sanity check here, not the binding gate; both now scope to the current training surface, `configs/events/train_*.yaml`, `rot_*.yaml` and `lopo_*.yaml`, `configs/reid/*.yaml` and `third_party/configs/*`. The binding data gates are the permanent exam-side allocation and the diet rule above. The renamed guard is an unconditional ban on the retired camera primitives rather than the singleton check it was then.) |
| torch/numpy on this Mac | torch 2.7.1 / numpy 2.4.6 under plain `uv run` |

**Blockers: none.** The one build step remaining before run 1 is generating the 10 training
configs + 10 val-scoring configs described above (mechanical: at most three fields each), plus the
two exam configs if and only if the bar is cleared.

## Results (appended 2026-09-05, after all ten runs; verdict: not promoted)

All ten runs completed under one code state, HEAD `bfd7839`, on the Mac CPU env declared
above. Every run wrote `report.json` (so every artefact is complete), and every run carries
a populated `git_commit`, so the caveat that H2h ran outside the repository is retired.

| # | run | arm | seed | val_auc_best | onnx sha8 | ckpt sha8 | reproduction |
|---|---|---|---|---:|---|---|---|
| 1 | `p3base_s1` | baseline | 20260826 | **0.75479** | `b9e04d03` | `d9f0e54b` | **reproduced** (pinned 0.75479 / `b9e04d03`) |
| 2 | `p3harm_s1` | harmonized | 20260826 | **0.78876** | `06b43397` | `40829919` | **reproduced** (pinned 0.78876 / `06b43397`) |
| 3 | `p3base_s2` | baseline | 20260901 | **0.77060** | `12aeb901` | `2ecfd96d` | **reproduced** (pinned 0.77060 / `12aeb901`) |
| 4 | `p3harm_s2` | harmonized | 20260901 | **0.78954** | `70afe929` | `79cf8577` | **reproduced** (pinned 0.78954 / `70afe929`) |
| 5 | `p3base_s3` | baseline | 20260905 | **0.75569** | `4e0890a2` | `6d7654df` | **reproduced** (pinned 0.75569 / `4e0890a2`) |
| 6 | `p3harm_s3` | harmonized | 20260905 | **0.77629** | `dafe0cbf` | `ead0cc35` | **reproduced** (pinned 0.77629 / `dafe0cbf`) |
| 7 | `p3base_s4` | baseline | 20260910 | **0.77480** | `25113cf7` | `eedcccfc` | not applicable (new seed) |
| 8 | `p3harm_s4` | harmonized | 20260910 | **0.76840** | `77e804ae` | `2a1acc93` | not applicable (new seed) |
| 9 | `p3base_s5` | baseline | 20260915 | **0.76539** | `ca321b58` | `4b7b9ea3` | not applicable (new seed) |
| 10 | `p3harm_s5` | harmonized | 20260915 | **0.80097** | `f321335c` | `75dca86b` | not applicable (new seed) |

All ten report `n_windows` 7362 / 387 pos / 6975 neg / 20 videos, `augment: null`,
`negative_class_weighting: null`, `geo_augment: null`, `resolved_pos_weight` 10.0, which is
the pinned dataset shape and the three explicitly disabled levers, checked per run before the
next run started.

**Why the ckpt sha8s differ from the recorded parents while the onnx sha8s match exactly.**
`verifier.pt` is `torch.save({"state_dict", "config"})` and the embedded `config` carries
`out_dir`, which by construction differs per run. The ONNX carries the weights alone, so it
is the byte-identity test. The same effect is already visible in the committed record: `conv_finalB`
ckpt `535141c1` and `conv_rotB/f2_s1` ckpt `24b56799` are two different checkpoint files of
the **same** model `b9e04d03`. The regression gate is read on the onnx sha8, as the table
above declares it.

### Regression gate: 6 of 6 exact, gate cleared before run 7 started

Every one of runs 1-6 reproduced its pre-declared `val_auc_best` to all five recorded digits
**and** its onnx sha8 bit-for-bit. Consequence beyond the gate itself: the open question
raised in the regression-gate section above is now closed, because commit `7a3d857`'s
`features.py` addition (`WindowSource`, `with_sources`, `_window_source`) is default-off
byte-identical on the **convulsion** path too, not only on the fall path where it was first
proven.

### P1, margin: **pass**

    mean(p3base, 5) = (0.75479 + 0.77060 + 0.75569 + 0.77480 + 0.76539) / 5
                    = 3.82127 / 5 = 0.764254
    mean(p3harm, 5) = (0.78876 + 0.78954 + 0.77629 + 0.76840 + 0.80097) / 5
                    = 3.92396 / 5 = 0.784792
    delta = 0.784792 - 0.764254 = 0.020538   >  0.016   pass  (headroom +0.004538, 1.28x the bar)

Recorded for completeness: at three seeds the arm sat at a difference of 0.02450 (1.53 times the bar). The two
new seeds cut the margin to 1.28 times without erasing it. The five-seed baseline mean 0.764254
landed 0.00390 above the recorded three-seed 0.76036, so the absolute reference in the
protocol (0.77636) was itself moved by the new baseline seeds; P1 is evaluated on the declared
*difference of five-seed means*, which is the form the bar was written in.

### P2, separation: **fail** (this is the condition that kills the push)

    min(p3harm) = 0.76840  (s4, seed 20260910, onnx 77e804ae)
    max(p3base) = 0.77480  (s4, seed 20260910, onnx 25113cf7)
    0.76840 < 0.77480  ->  not all five harmonized seeds are strictly above all five baseline seeds

Complete separation requires 25 of 25 pairwise comparisons; the observed arm has **23 of 25**.
The two inversions are both `p3harm_s4`:

| inversion | harmonized | baseline |
|---|---:|---:|
| `p3harm_s4` at or below `p3base_s2` | 0.76840 (`77e804ae`) | 0.77060 (`12aeb901`) |
| `p3harm_s4` at or below `p3base_s4` | 0.76840 (`77e804ae`) | 0.77480 (`25113cf7`) |

The declared certificate, exact two-sided p = 2/252 = 0.0079, is available **only** under
complete separation, so no p-value is available for this arm. Per the verdict rule, P1
passing while P2 fails, for example at 4-of-5 separation, is a fail. That is precisely what
happened, and the H2h reading it was designed to guard against, that the worst harmonized
seed beat the best baseline seed three times out of three, did not survive the addition of
seed 20260910.

### P3, stability: **pass**

    spread(p3base, 5) = 0.77480 - 0.75479 = 0.02001
    spread(p3harm, 5) = 0.80097 - 0.76840 = 0.03257
    cap = 2 x 0.02001 = 0.04002 ;  0.03257 <= 0.04002   pass  (ratio 1.63x)

Both arms are wider at five seeds than the three-seed figures the bar was calibrated on
(baseline 0.01581 to 0.02001; harmonized 0.01325 to 0.03257), and the harmonized arm's ratio
of 0.84 became 1.63. It clears the pre-declared cap; it clears it by less than half the room the
three-seed reading suggested.

### P4, the not-louder gate: **fail** at recall 0.50 (passes at 0.10 and 0.25)

Scored with the committed `scripts/events/eval_event_verifier.py`, one config per run, on the
fold-F2 val surface. **Every one of the ten evals reported `n_windows` 902 / `n_window_pos`
136 / 766 negative**, as pinned. Independent confirmation that this surface *is* the trainer's
val split: each eval's `window_auc` equals its run's `val_auc_best` to all five digits, for
all ten runs. Counts are matched-recall, k is the ceiling of r times 136 positives caught, FP = negatives at or
above the k-th positive's score; there were **zero ties at every threshold**, so the strict
and non-strict counts agree everywhere.

| recall | k | p3base FP (s1…s5) | mean | p3harm FP (s1…s5) | mean | harm minus base | gate |
|---:|---:|---|---:|---|---:|---:|---|
| 0.10 (14/136 = 0.10294) | 14 | 18, 16, 17, 10, 10 | **14.2** | 14, 20, 17, 11, 8 | **14.0** | **-0.2** | pass |
| 0.25 (34/136 = 0.25000) | 34 | 41, 39, 49, 48, 33 | **42.0** | 28, 39, 35, 31, 30 | **32.6** | **-9.4** | pass |
| 0.50 (68/136 = 0.50000) | 68 | 106, 97, 114, 108, 93 | **103.6** | 109, 144, 133, 158, 119 | **132.6** | **+29.0** | **fail** |

Per-run onnx sha8s for the rows above, in seed order: baseline `b9e04d03`, `12aeb901`,
`4e0890a2`, `25113cf7`, `ca321b58`; harmonized `06b43397`, `70afe929`, `dafe0cbf`,
`77e804ae`, `f321335c`.

The failure is not a one-seed artefact: at recall 0.50 the harmonized run is louder than its
seed-matched baseline in **5 of 5** pairs (109>106, 144>97, 133>114, 158>108, 119>93), a mean
+28.0% more false positives. The direction reverses with operating point, because the
harmonized arm is *quieter* in the sparse-alarm regime (-22.4% at recall 0.25) and louder once
the threshold is dropped to catch half the positive windows. Under the false-positive-first
priority the gate is the gate: this is exactly the axis push 2's augmentation arm died on, it
was pre-declared as a gate rather than a footnote, and it fails at one of the three declared
points.

### Fixed-threshold context: temperature-confounded, and not the gate

Recorded because the protocol said it would be, and labelled as it was declared. The fitted
temperatures span 0.4878 to 3.7137 across these ten runs, so a fixed 0.5 is a different
operating point for every row and the table below compares nothing:

| run | T | tp @0.5 | fp @0.5 | recall @0.5 |
|---|---:|---:|---:|---:|
| `p3base_s1` `b9e04d03` | 1.0251 | 60 | 80 | 0.4412 |
| `p3base_s2` `12aeb901` | 0.5385 | 16 | 21 | 0.1176 |
| `p3base_s3` `4e0890a2` | 0.9284 | 57 | 86 | 0.4191 |
| `p3base_s4` `25113cf7` | 2.2082 | 108 | 324 | 0.7941 |
| `p3base_s5` `ca321b58` | 0.8836 | 53 | 63 | 0.3897 |
| `p3harm_s1` `06b43397` | 1.7672 | 71 | 128 | 0.5221 |
| `p3harm_s2` `70afe929` | 0.4878 | 20 | 24 | 0.1471 |
| `p3harm_s3` `dafe0cbf` | 1.3459 | 62 | 124 | 0.4559 |
| `p3harm_s4` `77e804ae` | 3.7137 | 51 | 68 | 0.3750 |
| `p3harm_s5` `f321335c` | 3.2011 | 92 | 224 | 0.6765 |

### Verdict

| condition | result | margin |
|---|---|---|
| P1 margin above 0.016 | **pass** | difference 0.020538, +0.004538 of headroom |
| P2 complete separation | **fail** | 23/25 pairs; `p3harm_s4` 0.76840 sits under two baseline seeds |
| P3 spread at or below 2 times baseline | **pass** | 0.03257 at or below 0.04002 (1.63 times) |
| P4 not louder at 0.10/0.25/0.50 | **fail** | +29.0 mean FP at recall 0.50 (132.6 against 103.6) |

**P1 ∧ P2 ∧ P3 ∧ P4 is false, so the candidate is not promoted. No exam look taken. Zero
exam looks spent under push 3.** Two of the four conditions failed, and either one alone is
disqualifying by the pre-declared verdict rule. No ruler was changed after a number was seen,
no tiebreak was sought, and the candidate `p3harm` seed 20260826 (`06b43397`), fixed by the
recipe before any run, was never scored against E1 or E2.

### What this result does and does not say

- It **does** say: on the fold-F2 selection surface, the harmonized AJILE12 negative diet
  trains a better-*ranking* model on average (a difference of 0.020538 over five seeds) than the
  unchanged diet, and it does so more quietly in the sparse-alarm regime.
- It **does not** say the lever works. The five-seed design was declared precisely because
  three seeds cannot certify anything, and the fifth seed pair broke the separation the
  three-seed reading rested on. The gain is real on average and not reliable per seed.
- It **does not** say anything about the AJILE12 dose being shortcut-free: the
  `scene_n_tracks` caveat above is unchanged and unclosed, and it cancels in the difference rather than
  being removed.
- It **does not** say anything about E1, E2, false-alarm rate, or clinical detection
  performance. Nothing here is quoted as any of those. It remains an interim prototype under
  the event-model carve-out in POLICY.md, rule 4, motion signature only.
- The negative-side finding is the substantive one for the false-positive-first priority: the
  harmonized diet buys ranking at the top of the score distribution and **pays for it in the
  middle**, at +28.0% false positives at recall 0.50 in 5 of 5 seed-matched pairs.

### Artifacts and wall clock

- Training runs: `outputs/verifier/conv_p3base/f2_s{1..5}` and
  `outputs/verifier/conv_p3harm/f2_s{1..5}` (each `report.json` + `verifier.pt` +
  `verifier.onnx`), gitignored.
- P4 scoring: `.../eval_valF2/{report.json,windows.jsonl}` under each of those ten run dirs.
- Configs: `configs/events/rot_train_conv_p3{base,harm}_f2_s{1..5}.yaml` and
  `configs/events/rot_valF2_p3{base,harm}_s{1..5}.yaml`. Each training config is
  `rot_train_conv_finalB.yaml` with exactly the declared fields substituted for its arm and
  seed, verified by
  parsed diff; no `augment:`, `negative_class_weights:` or `geo_augment:` key exists in any
  of the twenty. `uv run ruff check .` clean and the guards listed in the preflight record
  green after generation.
- Curves: `final_models/seizure/curves/seizure_p3_{base,harm}_f2_s<seed>.csv`, ten files, 60 epochs
  each, registered in `scripts/analysis/extract_training_curves.py`.
- **No exam config was created.** `rot_eval_E1_p3harm.yaml` / `rot_eval_E2_p3harm.yaml` do not exist,
  by design: the bar was not cleared, so the look was not taken.
- Wall clock: 10 training runs from 18:53:43Z to 19:07:33Z = **13 min 50 s** (~83 s per run,
  well inside the ~3.4 min budgeted cadence); P4 pass of 10 evals 30 s; end to end under
  20 min against a ~50 min worst case.

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

The results table identifies its ten runs by eight hex characters of their ONNX and
checkpoint digests, twenty distinct values in all. Only final B `b9e04d03` resolves to a full
digest in this repository; the other nineteen resolve to nothing longer, as does the
`conv_rotB/f2_s1` checkpoint `24b56799` quoted in the byte-identity note beneath the table.
The candidate was not promoted, so no lane page entry carries them, and the `report.json`
files under `outputs/verifier/conv_p3base/`, `outputs/verifier/conv_p3harm/` and
`outputs/verifier/conv_rotB/` that hold the full values are gitignored build outputs. Stated
once for all of them: those nineteen prefixes and `24b56799` identify runs in gitignored
output directories rather than graphs, and are not verifiable from this tree. That is worth
saying explicitly, because the regression gate above is read on the eight-character ONNX
value. Capturing the full digests from those run directories would close the gap.

The bars the page reads against are complete. The full digests of final B `b9e04d03`, its
checkpoint `535141c1` and the model of record `07c399be` are carried on
`final_models/seizure/README.md`.

## Exam looks ledger (append-only)

This ledger records every look taken on E1 and E2, whichever protocol declared it, not only
looks declared by push 3. Push 3 contributed none; the three rows below were declared under
the coupling protocol on `final_models/seizure/README.md`.

| # | Candidate | Declared under | E1 | E2 | Date |
|---|---|---|---|---|---|
| 1 | Final B recipe refit on the v2c clinical track tree, seed 20260826 (ckpt 90a8734c / onnx 978a1c31; validation 0.88910 on holdout pat02 and pat09) | the coupling protocol on `final_models/seizure/README.md`: one E1 look and one E2 look for the single candidate, recipe unchanged except for the tree | window 0.69787, bar 0.87422 not met; 0 of 12 focal events caught at 0.8 | window 0.88047 / video 0.50000, bar 0.72916 cleared on window AUC; 0 of 4 convulsive events caught at 0.8 at 0.00 false-positive windows per hour, 0 of 621 negative windows over 0.8. **Not promotable**: no event reaches the alarm threshold the model of record catches all four at | 2026-09-07 |
| 2 | Model of record convulsion_v2_real `07c399be`, re-sliced onto the v2c tree with its weights untouched | the same protocol, one look per exam per frozen head, for attribution | window 0.60001; 9 of 12 focal events caught at 0.8 | window 0.65902 / video 0.25000; 3 of 4 caught at 0.8 at 951.23 false-positive windows per hour, 175 of 621 negative windows over 0.8; the same head holds the E2 bar at 0.72916 on the earlier tree | 2026-09-07 |
| 3 | Final B `b9e04d03`, re-sliced onto the v2c tree with its weights untouched | the same protocol, one look per exam per frozen head, for attribution | window 0.72499, against 0.87422 on the earlier tree; 7 of 12 focal events caught at 0.8 | window 0.65681 / video 0.25000; 4 of 4 caught at 0.8 at 1739.39 false-positive windows per hour, 320 of 621 negative windows over 0.8. Attribution of the movement to the v2c tree is in research/eyes-coupling-probe.md | 2026-09-07 |

**Budget status: push 3 is resolved and the candidate is not promoted. Zero exam looks were
taken under push 3's own budget; the 1 candidate slot goes unspent and, by this page's own
precedent for push 2's budget, does not carry over.** No push-3 row was added; the three rows
above were taken under the coupling protocol on `final_models/seizure/README.md`, not under
this push. The model-of-record entries are unchanged and the model record was not edited: the
broad seizure-motion head stays final B `b9e04d03`, and the convulsive head stays the model of
record `07c399be`.
