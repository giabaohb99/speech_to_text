import numpy as np
import pandas as pd

from src.classifier import load_summary_matrix, predict_classifier
from src.common import load_config
from src.evaluate import (accuracy_table, classifier_predictions, compute_folds, fold_indices, predict_folds,
                          speaker_group)
from src.features import DTW_COLS
from src.store import Bank


def test_speaker_group():
    assert speaker_group("tts_f1_p3") == "tts_f1"
    assert speaker_group("tts_m1_p0") == "tts_m1"
    assert speaker_group("an") == "an"
    assert speaker_group("tts_f1") == "tts_f1"


def toy_bank():
    """3 từ, 3 người thật, mỗi người 2 lần đọc; mỗi người lệch một khoảng riêng như khác giọng."""
    rng = np.random.default_rng(0)
    dim = len(DTW_COLS)
    templates = {w: rng.standard_normal((25, dim)) for w in ["mot", "hai", "ba"]}
    offsets = {s: 0.3 * rng.standard_normal(dim) for s in ["an", "binh", "chi"]}
    meta, seqs = [], []
    for word, base in templates.items():
        for speaker, offset in offsets.items():
            for idx in (1, 2):
                seqs.append(base + offset + 0.2 * rng.standard_normal(base.shape))
                meta.append({"file_id": f"{word}_{speaker}_{idx:02d}", "word": word,
                             "speaker": speaker, "sample_idx": idx})
    return Bank(pd.DataFrame(meta), seqs)


def test_folds_never_match_self_or_own_speaker():
    bank, cfg = toy_bank(), load_config()
    folds = compute_folds(bank, cfg, f0_weight=2.0, cmn=False)
    assert len(folds["B"]) == 3
    preds = predict_folds(bank, folds, k=3)
    a, b = preds[preds["scenario"] == "A"], preds[preds["scenario"] == "B"]
    assert len(a) == len(b) == 18
    assert (a["nearest_file"] != a["file_id"]).all()
    nearest_speaker = b["nearest_file"].str.split("_").str[1]
    assert (nearest_speaker != b["speaker"]).all()
    assert preds["correct"].mean() > 0.9


def test_fold_indices_cover_every_sample_once_per_scenario():
    bank = toy_bank()
    folds = fold_indices(bank)
    for scenario in ("A", "B"):
        tested = np.concatenate([test for test, _ in folds[scenario]])
        assert sorted(tested) == list(range(len(bank.meta)))
        for test, train in folds[scenario]:
            assert not set(test) & set(train)


def test_classifiers_run_on_same_folds():
    bank = toy_bank()
    rng = np.random.default_rng(1)
    centers = {"mot": 0.0, "hai": 3.0, "ba": 6.0}
    y = bank.meta["word"].to_numpy()
    X = np.array([centers[w] + 0.3 * rng.standard_normal(6) for w in y])
    for name in ("SVM", "Random Forest"):
        predicted, fit_ms = predict_classifier(name, X, y, fold_indices(bank))
        preds = classifier_predictions(bank, predicted)
        assert (preds.groupby("scenario").size() == 18).all()
        assert preds["correct"].mean() > 0.9
        assert fit_ms > 0


def test_summary_matrix_drops_speaker_pitch(tmp_path):
    cfg = load_config()
    pd.DataFrame({"file_id": ["b", "a"], "word": ["mot", "hai"], "speaker": ["an", "an"], "sample_idx": [1, 1],
                  "text": ["một", "hai"], "f0_hz_median": [200.0, 210.0], "mean_mfcc_01": [1.0, 2.0]}
                 ).to_csv(tmp_path / "features_summary.csv", index=False, encoding="utf-8-sig")
    test_cfg = {**cfg, "paths": {**cfg["paths"], "features": tmp_path}}
    X, cols = load_summary_matrix(test_cfg, ["a", "b"])
    assert cols == ["mean_mfcc_01"]
    assert X[:, 0].tolist() == [2.0, 1.0]


def test_accuracy_table_has_overall_and_per_speaker_rows():
    bank, cfg = toy_bank(), load_config()
    preds = predict_folds(bank, compute_folds(bank, cfg, 2.0, False), k=3).assign(method="DTW")
    acc = accuracy_table(preds)
    overall = acc[acc["speaker"] == "TẤT CẢ"]
    assert set(overall["scenario"]) == {"A", "B"}
    assert (overall["n"] == 18).all()
    assert len(acc) == 2 * (1 + 3)
