# Pose v2: licensing and trainability research (RTX 5090 / sm_120)

Compiled 2026-09-01 from primary sources. It is the brief behind the pose-lane
vendoring decision; the processor latency of the resulting path is measured in
research/pose-deployed-path-cpu-proof.md.
Verdict in one line: **stock RTMPose-m stays the plan; the mm stack runs on
sm_120 via maintained third-party wheels; the real counsel item is the `body7`
pretrained-weight ancestry, not any code licence.**

Evidence tags: `[V]` verified by reading the primary source; `[R]` reported
secondhand; `[C]` conjecture or inference

---

## Summary

**The premise is wrong in a useful way.** The claim that mmpose cannot train on a
5090 describes a *packaging* problem, not a code problem. Verified by source
read: **the RTMPose training path imports zero compiled mmcv CUDA ops**, and
there are **two
independent working prebuilt-wheel paths** for mmcv on sm_120 today, one of them
an actively maintained Apache-2.0 fork with sm_120 explicitly in its build matrix.

**Recommendation: stay on RTMPose-m, vendor upstream mmpose at a pinned commit,
and run it on the `onedl` (VBTI) fork of mmcv/mmengine/mmdet.** Fallback: same
mmpose, but the MiroPsota `pt2.11.0cu128` mmcv wheel, which matches the exact
torch build in use. Write a local ~30-line ONNX exporter; do not take a
dependency on MMDeploy.

The genuine licensing exposure is **not** the code (everything in the OpenMMLab
lane is Apache-2.0). It is the **`body7` pretrained weights**, which are trained
on a 7-dataset mixture including MPII, AI Challenger, CrowdPose and PoseTrack.
That is the one item for counsel.

---

## Q1: mmpose viability on sm_120

### 1.1 Official wheels: hard ceiling, verified by probe

The OpenMMLab wheel index was enumerated directly.

| Index path | HTTP | Contents |
|---|---|---|
| `cu121/torch2.4.0` | **200** | mmcv 2.2.0, cp38 to cp312 |
| `cu118/torch2.4.0` | **200** | yes |
| `cpu/torch2.4.0` | **200** | yes |
| `cu121/torch2.5.0` | 404 | none |
| `cu124/torch2.4.0`, `cu126/torch2.6.0`, `cu128/torch2.7.0`, `cu128/torch2.8.0` | **404** | none |

`[V]` **Official prebuilt mmcv stops at mmcv 2.2.0 / torch 2.4.0 / cu121.** There
is no cu124, cu126 or cu128 index at all.
(`https://download.openmmlab.com/mmcv/dist/<plat>/<torch>/index.html`)

`[V]` **mmcv on PyPI is sdist-only.** `mmcv-2.2.0.tar.gz` (479 KB), released
**2024-04-24**, no binary wheels. `pip install mmcv` therefore always compiles.
(`https://pypi.org/pypi/mmcv/json`)

### 1.2 Version gates, the good news

| Package | Gate | Source |
|---|---|---|
| mmpose `v1.3.2` **and** `main` | `mmcv >= 2.0.0rc4, <= 3.0.0`; `mmengine >= 0.6.0, <= 1.0.0` | `[V]` `mmpose/__init__.py` |
| mmdet `v3.3.0` (upstream) | `mmcv >= 2.0.0rc4, < 2.2.0` (strict) | `[V]` the version assert in `mmdet/__init__.py` |
| **onedl-mmdetection `main`** | **no mmcv/mmengine assert at all**, gate removed | `[V]` `mmdet/__init__.py` is 4 lines |

`[V]` mmpose does **not** cap at 2.1.0 (a widely repeated claim, for example the
Zenn RTX 5090 write-up asserts `<=2.1.0`). Both `v1.3.2` and `main` accept mmcv
up to 3.0.0. The only real conflict is **upstream** mmdet's `< 2.2.0`, and the onedl
fork deletes that gate entirely.

### 1.3 Does RTMPose training need compiled mmcv ops? **No.**

GitHub code search over `open-mmlab/mmpose` for `mmcv.ops` returns 11 hits, 4 of
them docs. `[V]`

| File | Status |
|---|---|
| `necks/posewarper_neck.py` | `try: from mmcv.ops import DeformConv2d`, **guarded** `[V]` |
| `heads/hybrid_heads/dekr_head.py` | its `mmcv.ops` import is **guarded** `[V]` |
| `engine/optim_wrappers/force_default_constructor.py` | behind `if mmcv_full_available():`, **guarded** `[V]` |
| `heads/transformer_heads/edpose_head.py` | `from mmcv.ops import MultiScaleDeformableAttention`, **unguarded** `[V]` |
| `.../transformers/deformable_detr_layers.py` | same, **unguarded** `[V]` |

Those last two are reached eagerly via the `from .transformer_heads import EDPoseHead`
line in `mmpose/models/heads/__init__.py`. `[V]` **That single line
is the only thing standing between stock mmpose and `mmcv-lite`.**

Supporting facts:

- `[V]` `mmcv/__init__.py` carries an explicit comment: the `op` and `utils`
  modules "are not imported to this level, so mmcv may be used without PyTorch."
  `import mmcv` never touches `mmcv._ext`.
- `[V]` `mmcv/cnn/__init__.py` and `mmcv/cnn/bricks/__init__.py` contain **no**
  reference to `ops`: `ConvModule`, `DepthwiseSeparableConvModule` and `Scale` are
  pure Python.
- `[V]` `rtmcc_head.py` imports only `torch`, `mmengine.dist`,
  `mmengine.structures`, and mmpose-internal modules. Its `forward()` runs conv,
  then `flatten`, then MLP, then GAU, then two Linears returning
  `(pred_x, pred_y)`. No mmcv at all.
- `[V]` `mmpose/models/backbones/cspnext.py` uses `mmcv.cnn` only.
- `[V]` `mmcv-lite 2.2.0` ships as **`mmcv_lite-2.2.0-py2.py3-none-any.whl`
  (732 KB)**: pure Python, platform-independent, **no torch pin** in
  `requires_dist`. It installs the same `mmcv` import package.

**RTMO** is equally clean on ops but not on dependencies:

- `[V]` `nms_torch` (in `mmpose/evaluation/functional/nms.py`) is pure torch,
  *not* `mmcv.ops.nms`.
- `[V]` `SimOTAAssigner` imports only `torch`, `torch.nn.functional`,
  `mmengine.structures`. Pure torch.
- `[V]` `yoloxpose_head.py` (RTMO's parent) uses `mmcv.cnn` only.
- **One caution.** `[V]` `rtmo_head.py` carries `from mmdet.utils import
  ConfigType, reduce_mean`. **This is the only mmdet coupling in RTMO's head, and both
  symbols already exist in mmpose** (`mmpose.utils.typing.ConfigType`,
  `mmpose.utils.reduce_mean`, which the parent class already uses). A one-line
  swap frees RTMO from mmdet.

### 1.4 The catch: the stock RTMPose *config* needs mmdet, and mmdet needs compiled mmcv

`[V]`
`mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py`
pulls three things from mmdet:

```python
from mmdet.datasets.transforms import YOLOXHSVRandomAug
from mmdet.engine.hooks import PipelineSwitchHook   # the stage-2 switch at epoch 380
from mmdet.models import CSPNeXt                    # note: _scope_='mmdet' in the legacy config
```

And `[V]` `mmdet/models/__init__.py` does `from .layers import *`, which reaches
`mmdet/models/layers/bbox_nms.py` and its `from mmcv.ops.nms import batched_nms`,
**unguarded**. So `import mmdet.models` requires real compiled mmcv.

Both mmdet pieces are trivially vendorable if the zero-compile route is wanted:
`PipelineSwitchHook` is ~40 lines of pure Python over `mmengine.Hook` +
`mmcv.transforms.Compose` `[V]`; `YOLOXHSVRandomAug` is a cv2/numpy
`BaseTransform` `[V]`; and mmpose ships its **own** `CSPNeXt`
(`mmpose/models/backbones/cspnext.py`, registered in mmpose's `MODELS`) that is
mmcv.cnn-only.

### 1.5 Two working prebuilt-wheel paths for sm_120

**Path A, `onedl` / VBTI (Apache-2.0, actively maintained). Preferred.**

`[V]` `VBTI-development/onedl-mmcv` `ci/build-matrix.json` on `main`:

| CUDA | torch | Python | `torch_cuda_arch_list` | sm_120? |
|---|---|---|---|---|
| 11.8 | 2.4.1 | 3.10 to 3.12 | `7.5 8.0 8.6 8.9 9.0` | no |
| 12.6 | 2.8.0, 2.12.0 | 3.10 to 3.13 | `7.5 8.6 8.9 9.0` | no |
| **12.8** | **2.8.0** | **3.10 to 3.13** | **`7.5 8.6 8.9 9.0 12.0`** | **yes** |
| **13.0** | **2.12.0** | **3.10 to 3.13** | **`7.5 8.6 8.9 9.0 12.0`** | **yes** |

- `[V]` Custom PyPI index at `https://mmwheels.onedl.ai/`, grouped by
  CUDA against torch. Published: `cu128-torch280` gives
  `onedl_mmcv-2.3.7.post1-cp310-cp310-manylinux_2_34_x86_64.whl`;
  `cu129-torch2100` gives the same version, **cp312**.
- `[V]` Installs as import module **`mmcv`** (`mmcv/version.py`: `__version__ =
  version('onedl-mmcv')`, resolving to `2.3.7.post1`), so mmpose's gate passes untouched.
- `[V]` CI test command is `import mmcv.ops; assert ext_loader.check_ops_exist()`,
  so compiled ops are verified per wheel.
- `[V]` Genuinely maintained: releases through 2026-06 fixing batched-NMS fp16,
  bf16 MSDeformAttn fallback, numpy overflow; `onedl-mmengine 0.10.12`
  (2026-06-11) adds an explicit `_safe_torch_load` handling PyTorch 2.6+
  `weights_only`; `onedl-mmdeploy 1.5.1` shipped **2026-08-22** with `numpy>2` +
  `onnxscript`.
- `[V]` Apache-2.0 across the family ("Copyright 2018-2023 OpenMMLab. Copyright
  2025 VBTI Products BV.").
- **There is no `onedl-mmpose`.** `[V]` `VBTI-development/mmpose` is an
  unmaintained mirror (last push 2024-08-07), and issue **#1 "[Feature] Any plan
  to upgrade MMPose?"** has been open since **2025-10-28 with zero comments**.
  The pose toolbox is the one gap in the fork family, which is why mmpose is
  *vendored* rather than pip-depended on.

**Path B, MiroPsota `torch_packages_builder` (exact match for the torch build in use)**

- `[V]` Index carries
  **`mmcv-2.2.0+a8073c7pt2.11.0cu128-cp{310,311,312,313,314}-linux_x86_64.whl`**,
  an exact match for torch 2.11.0+cu128.
- `[V]` HEAD request on the cp312 wheel: **55,547,911 bytes**. A real compiled
  artefact (mmcv-lite is 732 KB).
- `[V]` Arch chain closes: the builder sets `TORCH_CUDA_ARCH_LIST` from
  `torch.cuda.get_arch_list()` of the installed torch plus `FORCE_CUDA=1`
  (`.github/workflows/cuda/common_env.sh` and `get_torch_cuda_arch_list.py` in the
  `torch_packages_builder` repository, not in this one); and
  PyTorch **v2.11.0**'s `.ci/manywheel/build_cuda.sh`, in its architecture-list
  block, reads:

  ```
  TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0"
  case ${CUDA_VERSION} in
      12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0" ;;  # +Blackwell support
  ```

  So **torch 2.11.0+cu128 x86_64 ships sm_120, and the derived mmcv wheel does
  too.** (Note: cu**12.6** builds have *no* 12.0, so do not use a cu126 torch on
  this box.)
- `[V]` `a8073c7` = mmcv main @ **2026-01-16** (PR #3320), i.e. newer than the
  v2.2.0 tag. Builder repo is MIT. Caveat stated by the maintainer: wheels are
  built but "not tested to work correctly."

**Path C, source build.** Works, and the community failure reports have
identifiable causes:

- `[V]` mmcv#3310 (RTX 5090, torch 2.9.0+cu130) and mmcv#3319 (torch
  2.9.1+cu128): setup completes, `_ext` never compiles, wheel is 8 KB. Both
  **unanswered by maintainers**.
- `[C to V]` The likely cause is modern setuptools: MiroPsota's build script
  *explicitly downgrades* to `setuptools<82` for mmcv `[V]`, and the Zenn RTX
  5090 report succeeds precisely by using `--no-build-isolation` `[R]`:

  ```
  MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv==2.1.0 --no-cache-dir --no-build-isolation
  ```

  Result: CUDA 12.8 / nvcc 12.8.61 / torch 2.9.1+cu128 / RTX 5090, ~5-10 min,
  ~43 MB wheel.
- `[R]` mmcv#3327, a build report for RTX 5090 on CUDA 12.8: conda
  `cuda-nvcc=12.8` + `cuda-toolkit=12.8`, `CUDA_HOME=$CONDA_PREFIX`,
  `MAX_JOBS=1 python setup.py develop`. Single user report, no maintainer
  confirmation.
- `[V]` `MMCV_WITH_OPS` **defaults to `'1'`** in `setup.py`, so "forgot the env
  var" is *not* the explanation; the arch list comes from torch's own default
  detection (auto-detects the local sm_120 GPU).

### 1.6 Environment landmines (all verified)

| Item | Fact | Fix |
|---|---|---|
| **mmengine `torch.load`** | `[V]` upstream **0.10.7** (last stable) has bare `torch.load(...)` at 4 sites, which **breaks on torch >= 2.6** (`weights_only=True` default flip). | Use `mmengine >= 0.11.0rc0` (adds `weights_only=False` at all 4 sites) or `onedl-mmengine 0.10.12` (`_safe_torch_load`, tries `True` then falls back). |
| **mmdeploy** | `[V]` upstream last release **2023-12-25**, last commit 2024-09-30, pins `protobuf<=3.20.2`. Effectively dead. | Don't use it. See Q4 ONNX. |
| **chumpy** | `[V]` in `mmpose/requirements/runtime.txt`; PyPI 0.70 last released **2020-08-26**, classifiers stop at Py3 generic. SMPL-only. | Delete from requirements. |
| **xtcocotools** | `[V]` 1.14.3 (2023-10-19), wheels only for **cp37 to cp311**; sdist otherwise. | Pin Python **3.10 or 3.11**. |
| **`torch.cuda.amp.GradScaler`** | `[V]` still present at torch **v2.11.0** (`torch/cuda/amp/__init__.py`), so mmengine's `AmpOptimWrapper` import survives. | No action. |
| **mmrazor** (RTMPose distillation/pruning) | `[V]` last push 2024-06-11. | Irrelevant for fine-tuning; skip distillation. |

### 1.7 Upstream maintenance snapshot `[V]` (GitHub API, 2026-08-31)

| Repo | Last push | Last release | License | Archived |
|---|---|---|---|---|
| mmpose | **2025-08-04** (332 open issues) | 1.3.2, 2024-07-12 | Apache-2.0 | no |
| mmcv | 2026-01-29 (recent commits are Huawei NPU only) | 2.2.0, 2024-04-24 | Apache-2.0 | no |
| mmengine | 2026-07-13 | 0.11.0rc3 | Apache-2.0 | no |
| mmdet | none recorded | 3.3.0, 2024-01-05 | Apache-2.0 | no |
| mmdeploy | 2024-09-30 | 1.3.1, 2023-12-25 | Apache-2.0 | no |

`[R]` Community consensus (mmdetection discussion #11815) is that OpenMMLab
redirected staff to LLM work after the death of a senior academic leader; **no
official archival or successor statement exists.** No maintainer ever replied in that
thread.

---

## Q2: pure-PyTorch RTMPose/SimCC training reimplementations

| Option | License | Maintained | Fidelity | Verdict |
|---|---|---|---|---|
| **Stock mmpose on `mmcv-lite`** | Apache-2.0 | mmpose stale | **100% (it *is* the recipe)** | Best "de-mm" option, a ~3-line diff |
| **Intel `getitune`** (`open-edge-platform/geti`) | **Apache-2.0** `[V]` | **pushed 2026-09-01** | model/loss/codec faithful; recipe simplified | Best true reimplementation |
| DeepLabCut 3.x `pose_estimation_pytorch` | **LGPL-3.0-or-later** `[V]` | very active (2026-08-31) | **highest fidelity found anywhere** | Blocked: counsel flag, copyleft |
| `Nota-NetsPresso/netspresso-trainer` | Apache-2.0 `[V]` | stale (2025-09-29) | RTMCC head + loss only; MobileNetV3 + WFLW, no CSPNeXt | partial cross-check source |
| `leeyegy/SimCC` (paper repo) | **no licence** `[V]` | dead (2022-07-17) | Py3.6 Cython/CUDA NMS, `opencv==3.4.1.15` | dead end |
| `rtmlib` | Apache-2.0 `[V]` | **v0.0.16, 2026-08-04** | not applicable | **inference-only**; torch is not even a dependency `[V]` |
| `neesetifa/MMPose_Clean`, `aaroncherian/freemocap_playground`, `hoangvanphi2004/DNP`, +6 | no licence / defective relicense `[V]` | not applicable | not applicable | unusable |

**Key detail on `mmcv-lite`:** `[V]` MMPose's own `docs/en/installation.md`
concedes the point: "some functionalities are missing in this mode, usually
GPU-compiled ops like Deformable Convolution. **Most models in MMPose don't
depend on these ops.**"

**Intel `getitune` detail** (the strongest vendor target should mmpose be left
behind): `[V]`
`library/src/getitune/backend/lightning/models/keypoint_detection/` ships
`rtmpose.py`, `heads/rtmcc_head.py`, `losses/kl_discret_loss.py`,
`utils/{rtmcc_block,scale_norm,simcc_label,keypoint_eval}.py` +
`detection/backbones/cspnext.py`. Zero `import mmcv`/`import mmengine` (the 22
mm-references are all attribution docstrings). Pins `torch==2.12.1+cu130`.
`KLDiscretLoss(beta=10.0, label_softmax=True)`, `simcc_split_ratio=2.0`,
`sigma=(4.9, 5.66)` and `final_layer_kernel_size=7` all match the official config
exactly. Its exporter already emits `output_names=["pred_x","pred_y"]`, opset 18,
**exactly rtmlib's contract**. Gaps: only `rtmpose_tiny` in `CSPNEXT_CFG`
(need to add `deepen 0.67 / widen 0.75` and `in_channels=768`); drops
`RandomHalfBody`, `RandomFlip`, Albumentations, EMA, and the stage-2
`PipelineSwitchHook`. **Keep the optional `[ultralytics]` extra uninstalled.** It
is AGPL-3.0 and only the YOLO recipes touch it; the keypoint tree does not.

---

## Q3: alternative architectures

**Anchor:** `[V]` RTMPose-m @256x192 = **75.8 AP (AIC+COCO) / 74.9 AP (body7),
13.59 M params, 1.93 GFLOPs, 11.06 ms CPU** (i7-11700, ONNXRuntime). RTMPose-s =
72.2 AP / 5.47 M / 0.68 G / 4.48 ms. RTMPose-t = 68.5 AP / 3.34 M / 0.36 G /
3.20 ms.

### Licence-clean and modern-torch-trainable

| Candidate | Code | Weights | torch | Compiled ext? | COCO AP | Cost | ONNX |
|---|---|---|---|---|---|---|---|
| **RF-DETR Keypoint (Preview)** | **Apache-2.0** `[V]` | **Apache-2.0** (repo's own table row) `[V]` | `>=2.2.0`, no upper pin | **No**, pure-PyTorch `ms_deform_attn_core_pytorch`, zero `.cu` in tree `[V]` | 71.8 OKS @576² | 40.7 M | ONNX/TRT/CoreML/ExecuTorch `[V]` |
| **DETRPose** (arXiv 2506.13027) | **Apache-2.0** `[V]` | Apache-2.0 repo-wide `[R]`; init from D-FINE-Objects365 | unpinned (`numpy==1.23.5` pin is friction) | **No** `[V]` | N 57.2 / S 67.0 / L 72.5 | N 4.1 M/9.3 G | official `export_onnx.py` `[V]` |
| **HRNet-W32** | **MIT** `[V]` | GDrive, no statement `[C]` | plain torch | No (the `make` is demo-NMS only) | 74.4 | 28.5 M / 7.1 G | trivial |
| **SimpleBaseline** | **MIT** (Microsoft) `[V]` | no statement `[C]` | plain torch | No | R152 72.0 | not recorded | trivial |
| **torchvision KeypointRCNN** | **BSD-3** `[V]` | same BSD-3 project `[C]` | native | No | 65.0 kp AP | R50-FPN, heavy | native |
| **movenet.pytorch** | **MIT** `[V]` | author's own `[R]` | plain torch | No | below MoveNet | not recorded | yes |

### Killed on licence

| Candidate | Blocker |
|---|---|
| **ED-Pose** | **IDEA License 1.0**: "solely for your non-commercial research purposes"; commercial use "must request a license from IDEA" `[V]`. Also needs a compiled MSDeformAttn ext (unconditional top-level import of the C extension) `[V]`. |
| **Sapiens v1** | **CC-BY-NC 4.0** `[V]`. Also 0.3B to 2B params. |
| **Sapiens2** | Custom licence that **explicitly bans surveillance and biometric processing** `[V]`. A hospital-monitoring camera is squarely inside the prohibition. Hard no. |
| **YOLO-NAS-Pose** | Deci licence defines "Software" to include **model weights**, bars commercial use *and* bars modification, that is, it forbids the fine-tune itself `[V]`. |
| **KAPAO** GPL-3.0; **OpenPifPaf** AGPLv3; **AlphaPose** non-commercial academic `[V]` | Kill-list. |
| **TokenPose**, **SimCC-original** | **No LICENSE file**, so all rights reserved `[V]`. |
| **PCT** | MIT code `[V]` but `mmcv-full==1.7.0`, and mmcv **1.x cannot build against torch >= 2.1** at all (mmcv PR #3247, still unmerged) `[V]`. |
| **PETR/opera** | Apache-2.0 code `[V]` but requires building a vendored **mmcv 1.x** fork `[V]`. Same wall. |
| **MediaPipe BlazePose** | Apache-2.0 but **not retrainable**: Model Maker has no pose task `[V]`; 33 landmarks, not COCO-17. |

### ViTPose: the licensing trap is one level up

`[V]` The `ViTAE-Transformer/ViTPose` LICENSE really is **Apache-2.0**
("Copyright 2018-2020 Open-MMLab"). No S-Lab, no NC clause. **The problem is the
weights' ancestry.**

`[V]` `facebookresearch/mae/LICENSE` is **Creative Commons
Attribution-NonCommercial 4.0 International**. And mmpose's *own* ViTPose
configs initialize from OpenMMLab-mirrored MAE checkpoints:

```
td-hm_ViTPose-{small,base,large,huge}_8xb64-210e_coco-256x192.py
  init: https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_{s,b,l,h}_20230913.pth
```

`[V]` on the URLs; `[C]` on the legal conclusion, which is that re-hosting on an
Apache-2.0 project's CDN does not relicense. This kills the otherwise-attractive "just use
mmpose's own Apache-2.0 ViTPose weights" idea (ViTPose-S 73.9 AP, ViTPose-B 75.7
AP @256x192, OpenMMLab-trained `[V]`). ViTPose-**S** is the least-bad variant
`[C]`, since Meta never released a ViT-S MAE.

Two further points. `[V]` HF transformers **cannot fine-tune ViTPose**:
`modeling_vitpose.py` contains `raise NotImplementedError("Training is not yet
supported")`. And `[V]` ViTPose-S is **22 M / 5.3 GFLOPs for 73.8 AP** vs
RTMPose-m's **13.6 M / 1.93 GFLOPs for 75.8 AP**, so 2.7 times the FLOPs for about
2 AP *less*. This page carries two readings for ViTPose-S, 73.9 AP above and 73.8 AP
here, both tagged `[V]` and neither distinguished from the other by protocol; they are
recorded unreconciled, and the gap is 1.9 or 2.0 AP on either reading.
**There is no CPU win here.**

---

## Q4: ranking and recommendation

### 1st, recommended: stock RTMPose-m on the `onedl` stack

*Why:* keeps the exact architecture already validated in production, preserves
the full paper recipe (RandomHalfBody, Albumentations, EMA, stage-2 pipeline
switch), needs **zero new licences**, drops straight into rtmlib, and puts the
compiled dependency on a **maintained Apache-2.0 fork with sm_120 in its build
matrix.**

```
Python 3.10                                  # xtcocotools wheels; onedl cu128 wheels are cp310
torch==2.8.0+cu128                           # onedl's Blackwell-verified pairing
pip install --extra-index-url https://mmwheels.onedl.ai/cu128-torch280/simple/ \
    onedl-mmcv==2.3.7.post1
pip install onedl-mmengine==0.10.12 onedl-mmdetection==3.5.1
```

Vendor **`open-mmlab/mmpose` @ `759b39c1`** (last `main` commit, 2025-08-04).
Do *not* pip-depend on it; there is no `onedl-mmpose` and none is planned.

- Config, inside that clone:
  `mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py`
- Init: `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth`
  (counsel flag below applies)
- Vendoring patches: drop `chumpy` from `requirements/runtime.txt`. **No mmdet
  gate patch needed**: `[V]` onedl-mmdetection removed the assert, and both
  `CSPNeXt` and `PipelineSwitchHook` are present in the fork.
- Verify once on the pod: `python -c "import mmcv.ops; from mmcv.utils import
  ext_loader; assert ext_loader.check_ops_exist()"`.

### 2nd, fallback A: keep torch 2.11.0+cu128, use the exact-match wheel

If pinning torch down to 2.8 is unacceptable:

```
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    "mmcv==2.2.0+a8073c7pt2.11.0cu128"
pip install "mmengine>=0.11.0rc3" "mmdet==3.3.0"
```

One extra patch: set `mmcv_maximum_version = '2.3.0'` in `mmdet/__init__.py`. Same
mmpose pin, same config, same checkpoint. Slightly higher risk (builder ships
untested wheels), but it is a genuine 55 MB compiled artefact built with sm_120
in the arch list.

### 3rd, fallback B: zero-compile (`mmcv-lite`)

For a pod where nothing may be compiled: `pip install mmcv-lite==2.2.0
"mmengine>=0.11.0rc3"`, **no mmdet at all**, plus a ~4-file diff:

1. Guard/remove `from .transformer_heads import EDPoseHead` in
   `mmpose/models/heads/__init__.py`.
2. Swap the config's `mmdet.CSPNeXt` for mmpose's own `CSPNeXt`.
3. Vendor `YOLOXHSVRandomAug` (~40 lines cv2/numpy) and `PipelineSwitchHook`
   (~40 lines).

Cheapest to stand up, most divergence from upstream to carry.

### ONNX: do not use MMDeploy

`[V]` RTMCCHead's `forward()` runs conv, then `flatten`, then MLP, then GAU, then
two Linears returning `(pred_x, pred_y)`. No NMS, no data-dependent shapes, no dynamic
control flow. At a fixed 256x192 input, a ~30-line `torch.onnx.export` over
`model.backbone + model.head` is sufficient. `[V]` `rtmlib`'s
`RTMPose(onnx_model=<local path>, model_input_size=(192,256), mean=..., std=...)`
accepts an arbitrary local `.onnx` and unpacks `simcc_x, simcc_y = outputs` with
`simcc_split_ratio=2.0`, so **a retrained model is a drop-in with zero
inference-code change.** (If a maintained MMDeploy is wanted anyway:
`onedl-mmdeploy==1.5.1`, not the dead upstream.)

### Explicitly not recommended

- **RTMO**: pure-torch internals `[V]`, but its ONNX export genuinely needs
  MMDeploy rewriters (one-stage + NMS + dynamic shapes) `[V]`, and it drags in
  mmdet. Revisit only if top-view crowding makes the detector, crop and top-down
  chain the bottleneck.
- **ViTPose**: slower than RTMPose-m at equal-or-worse AP, plus a CC-BY-NC
  ancestry.
- **Anything requiring mmcv 1.x** (ViTPose-original, PETR, PCT): structurally
  impossible on torch >= 2.1 `[V]`.

### Worth a spike, not the main line

**RF-DETR Keypoint (Preview)** is the only candidate that is Apache-2.0 for
*both* code and weights by the repo's own explicit labelling, trains on
`torch>=2.2` with no compiled ops, and ships a keypoint fine-tune notebook
`[V]`. Its architectural argument for top-view CCTV is real, because it removes
the detector, crop and top-down chain, which is exactly where perspective
distortion hurts a top-down model. But it is "Preview," and 40.7 M @576x576 is a serious
CPU bill. If spiked, benchmark against **`detector + RTMPose-m × N people` at
actual ward occupancy**, not per-model.

---

## For counsel: the real licensing exposure is the weights, not the code

Every piece of the recommended stack is **Apache-2.0**: mmpose, mmcv, mmengine,
mmdet, rtmlib, and the onedl forks `[V]`. The exposure is elsewhere:

1. **`body7` pretrained weights.** `[V]` `body7` = COCO + **AI Challenger** +
   **CrowdPose** + **MPII** + sub-JHMDB + Halpe + **PoseTrack18**. `[R]` MPII is
   "free to researchers from educational or research institutes for
   non-commercial purposes"; CrowdPose is CC BY 4.0 "for academic and
   non-commercial research purposes"; AI Challenger and PoseTrack carry their
   own custom terms. **Fine-tuning a commercial hospital product from `body7`
   inherits that mixture.** Even the "cleanest" published variant is *AIC+COCO*,
   and the CSPNeXt backbone pretrain is `cspnext-m_udp-aic-coco_...pth`, so AI
   Challenger is baked in at every level. (Consistent with the provenance rule at
   POLICY.md, rule 8: no published RTMPose checkpoint is AIC-free.)
2. **Clean-room fallback exists** and is worth pricing: `[V]`
   `cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth` (mmdet, ImageNet-1k only)
   as backbone init, then train on COCO plus this project's own top-view data.
   Costs some AP,
   buys a much shorter provenance chain.
3. **Apply the standard consistently.** ImageNet-1k terms touch the incumbent
   RTMPose too. Whatever bar is set for ViTPose/MAE, RTMPose's own backbone must
   clear the same one; otherwise alternatives are rejected on a ground the
   incumbent also fails.
4. **`rtmlib` cosmetic note.** `[V]` Its LICENSE is Apache-2.0 but was copied
   unchanged from OpenMMLab, so the copyright line reads "Copyright 2018-2020
   Open-MMLab" rather than the author's name. Harmless; worth one sentence in
   the licence memo.
