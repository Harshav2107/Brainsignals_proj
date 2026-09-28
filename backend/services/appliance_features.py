"""Fast shared features for appliance-level classification.

Used BOTH by the training script (train_appliance_clf.py) and the live
backend (services/appliance_clf.py) so features always match exactly.

16 features: geometry (3) + trigger/paradigm (4) + amplitude stats (3)
+ spectral shape (6). All cheap: no bandpass/resampling, one Welch PSD
on the channel-averaged signal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

FEATURE_NAMES = [
    "n_channels", "log10_samples", "log10_dur_s",
    "has_trigger", "n_stim_types", "log1p_events", "event_rate_hz",
    "log10_std", "log10_meanabs", "outlier_frac",
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
    "spec_edge95_hz",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(
    eeg_raw: np.ndarray,
    trigger=None,
    orig_fs: float = 500.0,
) -> np.ndarray:
    eeg_raw = np.asarray(eeg_raw, dtype=np.float64)
    if eeg_raw.ndim == 1:
        eeg_raw = eeg_raw[np.newaxis, :]
    ch, n = int(eeg_raw.shape[0]), int(eeg_raw.shape[1])
    dur = max(n / float(orig_fs or 500.0), 1e-6)

    # ---- trigger / paradigm features ----
    has_trig, n_stim, n_ev, rate = 0.0, 0.0, 0.0, 0.0
    if trigger is not None:
        try:
            t = np.asarray(trigger).ravel()
            t = t[np.isfinite(t)].astype(int)
            if t.size:
                has_trig = 1.0
                n_stim = float(len([s for s in range(1, 7) if (t == s).any()]))
                n_ev = float((t > 0).sum())
                rate = float(n_ev) / dur
        except Exception:
            pass

    # ---- amplitude stats (channel-averaged, subsampled for speed) ----
    m = eeg_raw.mean(axis=0)
    step = max(len(m) // 20000, 1)
    ms = m[::step]
    sd = float(np.std(ms))
    ma = float(np.mean(np.abs(ms - np.mean(ms))))
    out = float(np.mean(np.abs(ms) > (4.0 * sd + 1e-9))) if sd > 0 else 0.0

    # ---- spectral shape (decimate to ~125 Hz, one Welch) ----
    d = max(int(round(float(orig_fs or 500.0) / 125.0)), 1)
    seg = m[::d]
    fs_d = float(orig_fs or 500.0) / d
    try:
        nperseg = int(min(256, max(len(seg) // 2, 16)))
        freqs, psd = welch(seg, fs=fs_d, nperseg=nperseg)
        total = float(psd.sum()) + 1e-12
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 60)]

        def rel(lo, hi):
            sel = (freqs >= lo) & (freqs < hi)
            return float(psd[sel].sum()) / total

        rels = [rel(lo, hi) for lo, hi in bands]
        cum = np.cumsum(psd) / total
        edge = float(freqs[int(np.searchsorted(cum, 0.95, side="left").clip(0, len(freqs) - 1))])
    except Exception:
        rels = [0.2] * 5
        edge = 0.0

    return np.array(
        [
            float(ch), float(np.log10(n + 1)), float(np.log10(dur)),
            has_trig, n_stim, float(np.log1p(n_ev)), float(rate),
            float(np.log10(sd + 1e-9)), float(np.log10(ma + 1e-9)), float(out),
            *rels, float(edge),
        ],
        dtype=np.float64,
    )
