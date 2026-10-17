import numpy as np
import pandas as pd
import pytest

from src.common import load_config
from src.dtw import cross_distances, dtw_distance, dtw_path
from src.features import DTW_COLS
from src.recognizer import FeatureTransform, Recognizer, is_excluded
from src.store import Bank

NO_BAND = 1000.0


def reference_dtw(a, b):
    """DTW kinh điển không dải, viết thẳng theo định nghĩa để đối chiếu."""
    n, m = len(a), len(b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            acc[i, j] = cost + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    return acc[n, m] / (n + m)


def test_distance_to_itself_is_zero():
    a = np.random.default_rng(0).standard_normal((30, 5))
    assert dtw_distance(a, a) == pytest.approx(0.0)


def test_distance_is_symmetric_for_different_lengths():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((25, 4)), rng.standard_normal((40, 4))
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))


def test_matches_reference_without_band():
    rng = np.random.default_rng(2)
    for _ in range(20):
        a = rng.standard_normal((rng.integers(5, 30), 3))
        b = rng.standard_normal((rng.integers(5, 30), 3))
        assert dtw_distance(a, b, band_ratio=NO_BAND) == pytest.approx(reference_dtw(a, b))


def test_band_never_lowers_distance():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((40, 3)), rng.standard_normal((35, 3))
    assert dtw_distance(a, b) >= dtw_distance(a, b, band_ratio=NO_BAND) - 1e-9


def test_time_stretched_copy_is_closer_than_other_sequence():
    t = np.linspace(0, 1, 40)
    a = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    t_slow = np.linspace(0, 1, 55)
    stretched = np.column_stack([np.sin(2 * np.pi * t_slow), np.cos(2 * np.pi * t_slow)])
    other = np.column_stack([np.cos(6 * np.pi * t), np.sin(6 * np.pi * t)])
    assert dtw_distance(a, stretched) < 0.2 * dtw_distance(a, other)


def test_cross_distances_matches_pairwise_calls():
    rng = np.random.default_rng(6)
    seqs_a = [rng.standard_normal((rng.integers(10, 30), 4)) for _ in range(4)]
    seqs_b = [rng.standard_normal((rng.integers(10, 30), 4)) for _ in range(5)]
    matrix = cross_distances(seqs_a, seqs_b)
    expected = [[dtw_distance(a, b) for b in seqs_b] for a in seqs_a]
    assert matrix.shape == (4, 5)
    assert np.allclose(matrix, expected)


def test_feature_transform_cmn_removes_mfcc_offset():
    cfg = load_config()
    rng = np.random.default_rng(7)
    base = rng.standard_normal((30, len(DTW_COLS)))
    shifted = base.copy()
    shifted[:, :13] += 5.0  # lệch hằng số trên MFCC, như đổi micro hoặc đổi giọng
    transform = FeatureTransform(cfg, cmn=True).fit([base, shifted])
    assert np.allclose(transform.apply(base), transform.apply(shifted), atol=1e-5)
    assert not np.allclose(FeatureTransform(cfg, cmn=False).fit([base, shifted]).apply(base),
                           FeatureTransform(cfg, cmn=False).fit([base, shifted]).apply(shifted))


def test_path_is_monotonic_from_start_to_end():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((20, 3)), rng.standard_normal((28, 3))
    path = dtw_path(a, b)
    assert path[0] == (0, 0) and path[-1] == (19, 27)
    steps = np.diff(np.array(path), axis=0)
    assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(axis=1) >= 1).all()


def test_is_excluded_by_name_or_group():
    assert is_excluded("an", ["an"])
    assert is_excluded("tts_f1_p3", ["tts_f1"])
    assert not is_excluded("tts_f10_p0", ["tts_f1"])
    assert not is_excluded("binh", ["an"])


@pytest.fixture(scope="module")
def toy_recognizer():
    """Kho giả 3 từ, mỗi từ một khuôn mẫu riêng, 2 người x 3 lần đọc có nhiễu và co giãn."""
    rng = np.random.default_rng(5)
    dim = len(DTW_COLS)
    templates = {w: rng.standard_normal((30, dim)) for w in ["mot", "hai", "ba"]}
    meta, sequences = [], []
    for word, base in templates.items():
        for speaker in ["an", "binh"]:
            for idx in range(1, 4):
                length = rng.integers(25, 36)
                warped = base[np.linspace(0, len(base) - 1, length).round().astype(int)]
                sequences.append(warped + 0.3 * rng.standard_normal(warped.shape))
                meta.append({"file_id": f"{word}_{speaker}_{idx:02d}", "word": word,
                             "speaker": speaker, "sample_idx": idx})
    bank = Bank(pd.DataFrame(meta), sequences)
    return Recognizer(bank, load_config()), templates, rng


def test_recognizer_predicts_correct_word(toy_recognizer):
    recognizer, templates, rng = toy_recognizer
    for word, base in templates.items():
        query = pd.DataFrame(base + 0.3 * rng.standard_normal(base.shape), columns=DTW_COLS)
        pred = recognizer.predict(query, exclude_speakers=["an"])
        assert pred.word == word
        assert pred.confidence > 0.5
        assert all(not fid.split("_")[1] == "an" for fid, _, _ in pred.neighbors)
        assert len(pred.top3) == 3


def test_recognizer_excludes_given_file(toy_recognizer):
    recognizer, _, _ = toy_recognizer
    query = pd.DataFrame(recognizer.sequences[0], columns=DTW_COLS)  # gần như trùng mẫu mot_an_01
    pred = recognizer.predict(query, exclude_files={"mot_an_01"})
    assert pred.neighbors[0][0] != "mot_an_01"
