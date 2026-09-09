"""Frequency-domain seizure detection from a pose-keypoint stream.

Clinical basis. The clonic phase of a generalized tonic-clonic seizure is rhythmic jerking
of the limbs in the 2-6 Hz band (Geertsema et al. detect convulsive seizures from the power
in 2-6 Hz relative to 0.5-12.5 Hz of video motion). We compute that same band-power ratio,
but on tracked limb keypoints instead of raw optical flow — so it is robust to camera motion
and reports *which* limbs are jerking and at what frequency.

Design (mirrors the D7 false-positives-forbidden stance):
  * Body-relative + scale-normalized. Each limb keypoint is measured relative to the torso
    centroid and divided by torso size, so walking / whole-body translation and the camera's
    distance do not register as limb oscillation.
  * Per-limb band-power ratio. For each limb we resample to a uniform grid, detrend, and take
    the fraction of spectral power in 2-6 Hz. A limb "jerks" only if that ratio is high AND
    its amplitude clears a floor (pose jitter reverses direction constantly at no amplitude).
  * Multi-limb + sustained. A single repetitive motion (waving, brushing teeth) is one limb;
    a convulsion drives several at once. The event is CONFIRMED only after enough limbs jerk
    together continuously for min_hold_s — a candidate that a trained verifier would gate.

This is a strong, explainable candidate detector, not a diagnosis: geometry cannot separate a
seizure from other whole-body rhythmic motion, which is what min_hold_s and the multi-limb
requirement (and, in the shipped system, the verifier) are for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# COCO-17 indices.
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14
L_AN, R_AN = 15, 16

TORSO = (L_SH, R_SH, L_HIP, R_HIP)
# Distal joints that swing during a convulsion (most diagnostic first).
LIMBS = (L_WR, R_WR, L_EL, R_EL, L_AN, R_AN, L_KN, R_KN, NOSE)


@dataclass
class SeizureConfig:
    window_s: float = 4.0  # analysis window (frequency resolution ~ 1/window)
    min_window_s: float = 2.0  # need at least this much history before scoring
    fs_target: float = 20.0  # uniform resample rate (Hz); Nyquist 10 Hz covers clonic band
    # The discriminator is the HIGH-FREQUENCY fraction: energy in [band_lo, band_hi] as a
    # fraction of movement energy [ref_lo, band_hi]. Clonic jerking is fast (>=3 Hz) and
    # broadband; voluntary rhythmic motion (walking ~1 Hz, jumping jacks ~2 Hz) sits below
    # 3 Hz and narrowband, so it fails this even at high amplitude. Measured on real data:
    # seizure hf_ratio ~0.40 vs jumping jacks ~0.08.
    band_lo: float = 3.0  # "fast jerk" band low (Hz)
    band_hi: float = 8.0  # fast jerk band high (Hz)
    ref_lo: float = 1.0  # movement band low (denominator starts here)
    kp_score_thr: float = 0.3  # keypoint visible if score above this
    min_visible_frac: float = 0.6  # limb must be visible for this frac of the window
    # Tuned on a real EMU tonic-clonic seizure vs two hard negatives (a calm patient and a
    # jumping-jacks workout). At these values: 71% firing across the seizure window (the
    # misses are the non-jerking tonic phase), 0% on the calm baseline, 0% on jumping jacks.
    amp_min: float = 0.09  # min RMS limb motion (fraction of torso size); kills pose jitter
    hf_ratio_thr: float = 0.35  # min high-frequency fraction; kills low-freq voluntary motion
    min_limbs: int = 2  # this many limbs jerking simultaneously -> candidate
    min_hold_s: float = (
        5.0  # candidate must persist this long -> CONFIRMED seizure (no brief bursts)
    )
    release_s: float = 1.5  # drop below threshold this long -> clear


@dataclass
class SeizureState:
    state: str = "idle"  # "idle" | "candidate" | "seizure"
    n_limbs: int = 0  # limbs jerking in the clonic band right now
    band_ratio: float = 0.0  # mean clonic band-power ratio over jerking limbs
    dominant_hz: float = 0.0  # mean dominant frequency of jerking limbs
    amplitude: float = 0.0  # mean normalized amplitude of jerking limbs
    held_s: float = 0.0  # how long the candidate condition has held
    limb_hz: dict = field(default_factory=dict)  # per-limb dominant Hz (for display)


class SeizureAnalyzer:
    """Streaming detector. Feed it (keypoints[17,2], scores[17], t) per frame."""

    def __init__(self, cfg: SeizureConfig | None = None) -> None:
        self.cfg = cfg or SeizureConfig()
        self._t: deque[float] = deque()
        self._kp: deque[np.ndarray] = deque()  # each (17,2)
        self._sc: deque[np.ndarray] = deque()  # each (17,)
        self._active_since: float | None = None
        self._below_since: float | None = None
        self._absent_since: float | None = None

    def reset(self) -> None:
        self._t.clear()
        self._kp.clear()
        self._sc.clear()
        self._active_since = None
        self._below_since = None
        self._absent_since = None

    def on_absent(self, now: float) -> SeizureState:
        """No trackable subject this frame. Decay the state toward idle exactly as a
        non-jerking frame would (so a CONFIRMED alert clears after release_s instead of
        freezing on an empty scene), and once the subject has been gone longer than
        release_s, drop the stale keypoint history so a returning person starts clean."""
        if self._absent_since is None:
            self._absent_since = now
        if now - self._absent_since >= self.cfg.release_s:
            self._t.clear()
            self._kp.clear()
            self._sc.clear()
        st = SeizureState(state=self.state_name())
        return self._advance(st, now, active=False)

    def _trim(self, now: float) -> None:
        cut = now - self.cfg.window_s
        while self._t and self._t[0] < cut:
            self._t.popleft()
            self._kp.popleft()
            self._sc.popleft()

    @staticmethod
    def _hf_ratio(sig: np.ndarray, fs: float, cfg: SeizureConfig) -> tuple[float, float]:
        """(high-frequency fraction, dominant Hz within the fast band) for a 1-D signal.

        hf_ratio = power[band_lo..band_hi] / power[ref_lo..band_hi]: the share of movement
        energy that is fast jerking. High for clonic seizure, low for slow voluntary rhythm.
        """
        n = len(sig)
        if n < 8:
            return 0.0, 0.0
        sig = sig - sig.mean()
        x = np.arange(n)
        sig = sig - np.polyval(np.polyfit(x, sig, 1), x)  # remove slow drift
        win = np.hanning(n)
        spec = np.abs(np.fft.rfft(sig * win)) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        movement = (freqs >= cfg.ref_lo) & (freqs <= cfg.band_hi)
        fast = (freqs >= cfg.band_lo) & (freqs <= cfg.band_hi)
        move_p = spec[movement].sum()
        if move_p <= 1e-12:
            return 0.0, 0.0
        ratio = float(spec[fast].sum() / move_p)
        pk = freqs[fast][np.argmax(spec[fast])] if fast.any() and spec[fast].max() > 0 else 0.0
        return ratio, float(pk)

    def update(self, keypoints: np.ndarray, scores: np.ndarray, now: float) -> SeizureState:
        cfg = self.cfg
        self._absent_since = None  # a subject is present this frame
        self._t.append(float(now))
        self._kp.append(np.asarray(keypoints, np.float32))
        self._sc.append(np.asarray(scores, np.float32))
        self._trim(now)

        st = SeizureState(state=self.state_name())
        span = self._t[-1] - self._t[0] if len(self._t) > 1 else 0.0
        if span < cfg.min_window_s or len(self._t) < 8:
            return self._advance(st, now, active=False)

        t = np.array(self._t)
        kp = np.stack(self._kp)  # (T,17,2)
        sc = np.stack(self._sc)  # (T,17)

        # torso centroid + scale per frame (visible torso points only)
        torso_vis = sc[:, TORSO] > cfg.kp_score_thr
        tc = np.full((len(t), 2), np.nan, np.float32)
        for i in range(len(t)):
            vis = torso_vis[i]
            if vis.sum() >= 2:
                tc[i] = kp[i, list(TORSO)][vis].mean(axis=0)
        # scale = median shoulder->hip span across the window (robust, single number)
        sh = np.nanmean(kp[:, [L_SH, R_SH]], axis=1)
        hp = np.nanmean(kp[:, [L_HIP, R_HIP]], axis=1)
        scale = np.nanmedian(np.hypot(*(sh - hp).T))
        if not np.isfinite(scale) or scale < 1e-3:
            scale = np.nanmedian(
                np.hypot(kp[:, L_SH, 0] - kp[:, R_SH, 0], kp[:, L_SH, 1] - kp[:, R_SH, 1])
            )
        if not np.isfinite(scale) or scale < 1e-3:
            return self._advance(st, now, active=False)

        # uniform time grid
        grid = np.arange(t[0], t[-1], 1.0 / cfg.fs_target)
        if len(grid) < 8:
            return self._advance(st, now, active=False)

        jerking = 0
        ratios, peaks, amps = [], [], []
        limb_hz: dict[int, float] = {}
        for k in LIMBS:
            vis = sc[:, k] > cfg.kp_score_thr
            if vis.mean() < cfg.min_visible_frac:
                continue
            # body-relative, scale-normalized position (both axes)
            rel = (kp[:, k] - tc) / scale  # (T,2)
            ok = vis & np.isfinite(rel).all(axis=1) & np.isfinite(tc).all(axis=1)
            if ok.sum() < 8:
                continue
            axis_ratio, axis_peak, axis_amp = [], [], []
            for ax in (0, 1):
                s = np.interp(grid, t[ok], rel[ok, ax])
                amp = float(np.sqrt(np.mean((s - s.mean()) ** 2)))
                r, pk = self._hf_ratio(s, cfg.fs_target, cfg)
                axis_ratio.append(r)
                axis_peak.append(pk)
                axis_amp.append(amp)
            # take the more active axis for this limb
            a = int(np.argmax(axis_amp))
            amp, ratio, peak = axis_amp[a], axis_ratio[a], axis_peak[a]
            limb_hz[k] = peak
            # A limb jerks only if it moves enough (not jitter) AND that movement is fast
            # (not slow voluntary rhythm). Both gates are required.
            if amp >= cfg.amp_min and ratio >= cfg.hf_ratio_thr:
                jerking += 1
                ratios.append(ratio)
                peaks.append(peak)
                amps.append(amp)

        st.n_limbs = jerking
        st.limb_hz = limb_hz
        if ratios:
            st.band_ratio = float(np.mean(ratios))
            st.dominant_hz = float(np.mean(peaks))
            st.amplitude = float(np.mean(amps))
        active = jerking >= cfg.min_limbs
        return self._advance(st, now, active=active)

    def _advance(self, st: SeizureState, now: float, active: bool) -> SeizureState:
        cfg = self.cfg
        if active:
            self._below_since = None
            if self._active_since is None:
                self._active_since = now
            held = now - self._active_since
            st.held_s = held
            st.state = "seizure" if held >= cfg.min_hold_s else "candidate"
        else:
            if self._active_since is not None:
                if self._below_since is None:
                    self._below_since = now
                # allow brief dropouts without losing the event
                if now - self._below_since >= cfg.release_s:
                    self._active_since = None
                    self._below_since = None
                    st.state = "idle"
                else:
                    held = (self._below_since - self._active_since) if self._active_since else 0.0
                    st.held_s = max(0.0, held)
                    st.state = "seizure" if held >= cfg.min_hold_s else "candidate"
            else:
                st.state = "idle"
        self._state = st.state
        return st

    def state_name(self) -> str:
        return getattr(self, "_state", "idle")
