import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.common import load_config
from src.features import DTW_COLS, FRAME_COLS, extract, summarize
from src.preprocess import normalize, trim_silence
from src.store import build_bank, load_bank

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def tone(ms, freq=200.0, amp=0.5):
    t = np.arange(int(SR * ms / 1000)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def noise(ms, amp, rng):
    return amp * rng.standard_normal(int(SR * ms / 1000))


def test_trim_keeps_voiced_part_plus_padding(cfg):
    rng = np.random.default_rng(0)
    voiced = tone(300)
    y = np.concatenate([noise(300, 0.002, rng), voiced, noise(300, 0.002, rng)])
    y = y + noise(len(y) * 1000 // SR, 0.001, rng)
    out, info = trim_silence(normalize(y), SR, cfg["audio"], cfg["preprocess"])
    voice_start, voice_end = int(0.3 * SR), int(0.6 * SR)
    lost_ms = (max(0, info["start"] - voice_start) + max(0, voice_end - info["end"])) * 1000 / SR
    assert info["trimmed"]
    assert lost_ms <= 20
    assert 300 <= len(out) * 1000 / SR <= 300 + 2 * cfg["preprocess"]["pad_ms"] + 40


def test_trim_keeps_quiet_fricative_onset(cfg):
    rng = np.random.default_rng(1)
    fricative = noise(80, 0.03, rng)          # phụ âm xát nhỏ đứng trước nguyên âm to
    y = np.concatenate([noise(300, 0.001, rng), fricative, tone(250), noise(300, 0.001, rng)])
    _, info = trim_silence(normalize(y), SR, cfg["audio"], cfg["preprocess"])
    assert info["start"] <= int(0.3 * SR) + int(0.02 * SR)


def test_trim_all_silence_does_not_fail(cfg):
    y = np.zeros(SR)
    out, info = trim_silence(normalize(y), SR, cfg["audio"], cfg["preprocess"])
    assert len(out) == SR
    assert not info["trimmed"]


def test_frame_count_matches_length(cfg):
    y = tone(500, freq=150)
    frames, info = extract(y, SR, cfg)
    hop = SR * cfg["audio"]["hop_ms"] // 1000
    assert len(frames) == 1 + len(y) // hop
    assert list(frames.columns) == FRAME_COLS
    assert not frames.isna().any().any()
    assert 140 < np.nanmedian(info["f0_hz"]) < 160


def test_rising_pitch_gives_positive_f0_slope(cfg):
    t = np.arange(int(0.5 * SR)) / SR
    freq = 150 + 100 * t / t[-1]
    y = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    frames, info = extract(y, SR, cfg)
    assert summarize(frames, info)["f0_slope"] > 0


def test_build_and_load_bank_roundtrip(cfg, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(2)
    for word, freq in [("mot", 150), ("hai", 250)]:
        for idx in (1, 2):
            y = np.concatenate([noise(150, 0.002, rng), tone(300, freq), noise(150, 0.002, rng)])
            sf.write(raw / f"{word}_an_{idx:02d}.wav", y, SR)
    test_cfg = {**cfg, "paths": {**cfg["paths"], "raw": raw, "features": tmp_path / "features"}}

    summary = build_bank(test_cfg, {"mot": "một", "hai": "hai"})
    frames = pd.read_csv(tmp_path / "features" / "features_frames.csv", encoding="utf-8-sig")
    bank = load_bank(test_cfg)

    assert len(summary) == 4
    assert frames.groupby("file_id").size().tolist() == [len(s) for s in bank.sequences]
    assert all(s.shape[1] == len(DTW_COLS) for s in bank.sequences)
    assert sorted(bank.meta["word"].unique()) == ["hai", "mot"]
