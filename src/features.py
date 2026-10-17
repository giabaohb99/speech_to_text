"""Trích đặc trưng theo khung: MFCC + delta + delta-delta, F0, năng lượng, ZCR, spectral centroid."""
import librosa
import numpy as np
import pandas as pd
from scipy.signal import medfilt

N_MFCC = 13
MFCC_COLS = [f"mfcc_{i:02d}" for i in range(1, N_MFCC + 1)]
D_COLS = [f"d_{i:02d}" for i in range(1, N_MFCC + 1)]
DD_COLS = [f"dd_{i:02d}" for i in range(1, N_MFCC + 1)]
EXTRA_COLS = ["f0", "energy", "zcr", "centroid"]
FRAME_COLS = ["frame_idx"] + MFCC_COLS + D_COLS + DD_COLS + EXTRA_COLS
DTW_COLS = MFCC_COLS + D_COLS + DD_COLS + ["f0"]

PYIN_FRAME = 1024   # 64 ms, đủ dài cho giọng trầm 70 Hz
F0_MEDFILT = 5      # lọc trung vị khử lỗi nhảy quãng tám


def contour_f0(f0_hz):
    """F0 Hz (NaN ở khung vô thanh) -> đường log2 đã nội suy, lọc và trừ trung vị của file.

    Trừ trung vị bỏ khác biệt cao độ giữa người, giữ lại hình dạng thanh điệu.
    """
    voiced = np.isfinite(f0_hz)
    if voiced.sum() < 2:
        return np.zeros_like(f0_hz)
    log_f0 = np.log2(f0_hz)
    frames = np.arange(len(f0_hz))
    log_f0 = np.interp(frames, frames[voiced], log_f0[voiced])
    log_f0 = medfilt(log_f0, F0_MEDFILT)
    return log_f0 - np.median(log_f0)


def extract(y, sr, cfg):
    """Trả về (DataFrame một dòng một khung theo FRAME_COLS, thông tin phụ cho tóm tắt và vẽ)."""
    acfg, fcfg = cfg["audio"], cfg["features"]
    hop = int(sr * acfg["hop_ms"] / 1000)
    win = int(sr * acfg["frame_ms"] / 1000)
    n_fft = fcfg["n_fft"]

    emphasized = librosa.effects.preemphasis(y, coef=cfg["preprocess"]["preemphasis"])
    mfcc = librosa.feature.mfcc(y=emphasized, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop,
                                win_length=win, window="hamming", n_mels=fcfg["n_mels"])
    width = min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))
    delta = librosa.feature.delta(mfcc, width=width, order=1)
    delta2 = librosa.feature.delta(mfcc, width=width, order=2)

    f0_hz, _, _ = librosa.pyin(y, fmin=fcfg["f0_min"], fmax=fcfg["f0_max"], sr=sr,
                               frame_length=PYIN_FRAME, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]

    n = min(mfcc.shape[1], len(f0_hz), len(rms), len(zcr), len(centroid))
    f0_hz = f0_hz[:n]
    data = np.column_stack([
        np.arange(n), mfcc[:, :n].T, delta[:, :n].T, delta2[:, :n].T,
        contour_f0(f0_hz), 10 * np.log10(rms[:n] ** 2 + 1e-10), zcr[:n], centroid[:n],
    ])
    frames = pd.DataFrame(data, columns=FRAME_COLS)
    frames["frame_idx"] = frames["frame_idx"].astype(int)
    info = {"f0_hz": f0_hz, "duration_ms": 1000 * len(y) / sr}
    return frames, info


def summarize(frames, info):
    """Gộp đặc trưng theo khung thành một dòng thống kê cho cả mẫu."""
    row = {"duration_ms": round(info["duration_ms"], 1), "n_frames": len(frames)}
    for col in MFCC_COLS:
        row[f"mean_{col}"] = frames[col].mean()
    for col in MFCC_COLS:
        row[f"std_{col}"] = frames[col].std(ddof=0)
    for col in D_COLS:
        row[f"mean_{col}"] = frames[col].mean()
    f0 = frames["f0"].to_numpy()
    voiced_hz = info["f0_hz"][np.isfinite(info["f0_hz"])]
    row["f0_hz_median"] = float(np.median(voiced_hz)) if len(voiced_hz) else 0.0
    row["voiced_ratio"] = len(voiced_hz) / max(1, len(info["f0_hz"]))
    row["f0_std"] = f0.std()
    row["f0_slope"] = np.polyfit(np.arange(len(f0)), f0, 1)[0] if len(f0) > 1 else 0.0
    row["energy_mean"] = frames["energy"].mean()
    row["energy_max"] = frames["energy"].max()
    row["zcr_mean"] = frames["zcr"].mean()
    row["centroid_mean"] = frames["centroid"].mean()
    return row
