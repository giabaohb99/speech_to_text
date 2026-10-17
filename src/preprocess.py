"""Tiền xử lý: nạp, bỏ thành phần một chiều, chuẩn hóa đỉnh, cắt khoảng lặng hai đầu.

Pre-emphasis không làm ở đây mà làm ngay trước khi tính MFCC, vì F0 cần tín hiệu gốc.
Không cắt được (toàn lặng hoặc đoạn có tiếng quá ngắn) thì trả nguyên tín hiệu, info["trimmed"] = False.
"""
import librosa
import numpy as np

PEAK = 0.95
FLOOR_PERCENTILE = 10     # năng lượng nền = phân vị 10 của mọi khung trong file
MAX_DYNAMIC_DB = 35       # khung thấp hơn đỉnh quá mức này coi như lặng, kể cả khi nền rất sạch
FRICATIVE_ZCR = 0.25      # ZCR cao kèm năng lượng gần ngưỡng: giữ phụ âm xát như s, x, ch


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def normalize(y):
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    return y / peak * PEAK if peak > 0 else y


def frame_energy_zcr(y, frame, hop):
    """Năng lượng log (dB) và ZCR từng khung, khung căn giữa giống librosa."""
    padded = np.pad(y, frame // 2)
    if len(padded) < frame:
        padded = np.pad(padded, (0, frame - len(padded)))
    frames = librosa.util.frame(padded, frame_length=frame, hop_length=hop)
    energy_db = 10 * np.log10(np.mean(frames**2, axis=0) + 1e-10)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=0)), axis=0)
    return energy_db, zcr


def trim_silence(y, sr, acfg, pcfg):
    """Cắt lặng hai đầu. Trả về (tín hiệu đã cắt, thông tin chi tiết để vẽ và kiểm tra)."""
    frame = int(sr * acfg["frame_ms"] / 1000)
    hop = int(sr * acfg["hop_ms"] / 1000)
    energy_db, zcr = frame_energy_zcr(y, frame, hop)
    floor_db = np.percentile(energy_db, FLOOR_PERCENTILE)
    threshold = max(floor_db + pcfg["energy_margin_db"], energy_db.max() - MAX_DYNAMIC_DB)
    voiced = (energy_db > threshold) | ((energy_db > threshold - 3) & (zcr > FRICATIVE_ZCR))

    info = {"energy_db": energy_db, "zcr": zcr, "threshold_db": threshold,
            "hop": hop, "start": 0, "end": len(y), "trimmed": False}
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return y, info
    pad = int(sr * pcfg["pad_ms"] / 1000)
    start = max(0, idx[0] * hop - pad)
    end = min(len(y), idx[-1] * hop + frame // 2 + pad)
    if (idx[-1] - idx[0] + 1) * hop < sr * pcfg["min_voiced_ms"] / 1000:
        return y, info  # đoạn có tiếng quá ngắn, nhiều khả năng là tiếng động: giữ nguyên
    info.update(start=start, end=end, trimmed=True)
    return y[start:end], info


def load_and_clean(path, cfg):
    """Nạp file và trả về (tín hiệu sạch đã cắt lặng, thông tin cắt lặng)."""
    sr = cfg["audio"]["sample_rate"]
    y = normalize(load_audio(path, sr))
    return trim_silence(y, sr, cfg["audio"], cfg["preprocess"])
