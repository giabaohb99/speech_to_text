"""Nhận dạng bằng DTW + k láng giềng gần nhất trên kho đặc trưng."""
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.dtw import cross_distances
from src.features import DTW_COLS, MFCC_COLS

EPS = 1e-6
F0_INDEX = DTW_COLS.index("f0")
N_MFCC = len(MFCC_COLS)


class FeatureTransform:
    """Biến chuỗi đặc trưng thô thành chuỗi dùng cho DTW.

    1. CMN (tùy chọn): trừ trung bình theo thời gian của từng hệ số MFCC trong chính file đó,
       giảm khác biệt kênh thu và giọng giữa người nói.
    2. Chuẩn hóa từng chiều theo trung bình, độ lệch chuẩn học từ tập train (fit).
    3. Nhân trọng số cho chiều F0.
    """

    def __init__(self, cfg, f0_weight=None, cmn=None):
        fcfg = cfg["features"]
        self.f0_weight = fcfg["f0_weight"] if f0_weight is None else f0_weight
        self.cmn = fcfg.get("cmn", False) if cmn is None else cmn
        self.weights = np.ones(len(DTW_COLS), dtype=np.float32)
        self.weights[F0_INDEX] = self.f0_weight
        self.mean = self.std = None

    def _pre(self, seq):
        seq = np.array(seq, dtype=np.float32)
        if self.cmn:
            seq[:, :N_MFCC] -= seq[:, :N_MFCC].mean(axis=0)
        return seq

    def fit(self, seqs):
        stacked = np.vstack([self._pre(s) for s in seqs])
        self.mean = stacked.mean(axis=0)
        self.std = stacked.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def apply(self, seq):
        return (self._pre(seq) - self.mean) / self.std * self.weights


@dataclass
class Prediction:
    word: str
    confidence: float
    top3: list                    # [(mã từ, điểm, khoảng cách nhỏ nhất)]
    neighbors: list               # [(file_id, mã từ, khoảng cách)] của k mẫu gần nhất
    word_distances: dict          # mã từ -> khoảng cách DTW nhỏ nhất tới mẫu của từ đó
    elapsed_ms: float = 0.0


def vote(dists, words, file_ids, k):
    """Bỏ phiếu k láng giềng gần nhất. dists, words, file_ids cùng độ dài; khoảng cách vô cùng bị bỏ qua."""
    order = np.argsort(dists, kind="stable")
    order = order[np.isfinite(dists[order])]
    scores = {}
    for idx in order[:k]:
        scores[words[idx]] = scores.get(words[idx], 0.0) + 1.0 / (dists[idx] + EPS)
    min_dist = {}
    for idx in order:
        min_dist.setdefault(words[idx], float(dists[idx]))
    ranked = sorted(min_dist, key=lambda w: (-scores.get(w, 0.0), min_dist[w]))
    best = ranked[0]
    return Prediction(
        word=best,
        confidence=scores[best] / sum(scores.values()),
        top3=[(w, scores.get(w, 0.0), min_dist[w]) for w in ranked[:3]],
        neighbors=[(file_ids[i], words[i], float(dists[i])) for i in order[:k]],
        word_distances=min_dist,
    )


def is_excluded(speaker, exclude_speakers):
    """Loại người trùng tên, hoặc cả nhóm: 'tts_f1' loại mọi 'tts_f1_p*'."""
    return any(speaker == x or speaker.startswith(x + "_") for x in exclude_speakers)


class Recognizer:
    def __init__(self, bank, cfg, transform=None):
        self.cfg = cfg
        self.meta = bank.meta.reset_index(drop=True)
        self.transform = transform or FeatureTransform(cfg).fit(bank.sequences)
        self.sequences = [self.transform.apply(seq) for seq in bank.sequences]

    def normalize(self, seq):
        return self.transform.apply(seq)

    def sequence_of(self, file_id):
        return self.sequences[int(np.flatnonzero(self.meta["file_id"].to_numpy() == file_id)[0])]

    def candidates(self, exclude_speakers=(), exclude_files=()):
        keep = [
            i for i, row in enumerate(self.meta.itertuples())
            if row.file_id not in exclude_files and not is_excluded(row.speaker, exclude_speakers)
        ]
        if not keep:
            raise ValueError("Kho trống sau khi loại người hoặc file")
        return keep

    def predict(self, frames, exclude_speakers=(), exclude_files=(), k=None):
        """frames: DataFrame theo khung từ features.extract. Trả về Prediction."""
        return self.predict_sequence(frames[DTW_COLS].to_numpy(dtype=np.float32),
                                     exclude_speakers, exclude_files, k)

    def predict_sequence(self, raw_seq, exclude_speakers=(), exclude_files=(), k=None):
        """Như predict nhưng nhận thẳng ma trận đặc trưng thô (số khung x len(DTW_COLS))."""
        started = time.perf_counter()
        dcfg = self.cfg["dtw"]
        query = self.normalize(raw_seq)
        keep = self.candidates(exclude_speakers, exclude_files)
        dists = cross_distances([query], [self.sequences[i] for i in keep],
                                dcfg["band_ratio"], dcfg["band_min_frames"])[0]
        pred = vote(dists, self.meta["word"].to_numpy()[keep], self.meta["file_id"].to_numpy()[keep],
                    k or self.cfg["knn"]["k"])
        pred.elapsed_ms = 1000 * (time.perf_counter() - started)
        return pred


LOG_FIELDS = ["time", "file", "predicted", "text", "confidence", "top2", "top3",
              "nearest_file", "excluded_speakers", "elapsed_ms", "figure"]


def append_log(log_path, row):
    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8-sig") as log_file:
        log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        if new_log:
            log.writeheader()
        log.writerow(row)


def recognize_file(path, cfg, words, recognizer, out_dir, exclude_speakers=(), k=None, save_figure=True):
    """Nhận dạng một file, lưu hình kết quả và ghi một dòng vào out_dir/log.csv.

    Tự loại chính file khỏi kho nếu nó đã có trong kho. Dùng chung cho lệnh recognize và giao diện demo.
    Trả về (Prediction, đường dẫn hình hoặc None, có loại chính file không).
    """
    from src.plots import plot_recognition
    from src.store import extract_file

    path = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded_self = path.stem in set(recognizer.meta["file_id"])
    frames, _ = extract_file(path, cfg)
    pred = recognizer.predict(frames, exclude_speakers, {path.stem} if excluded_self else set(), k)
    figure_path = None
    if save_figure:
        figure_path = out_dir / f"{path.stem}_{datetime.now():%Y%m%d-%H%M%S}.png"
        plot_recognition(path, cfg, words, recognizer, frames, pred, figure_path)
    append_log(out_dir / "log.csv", {
        "time": datetime.now().isoformat(timespec="seconds"), "file": str(path),
        "predicted": pred.word, "text": words.get(pred.word, pred.word), "confidence": round(pred.confidence, 3),
        "top2": pred.top3[1][0] if len(pred.top3) > 1 else "",
        "top3": pred.top3[2][0] if len(pred.top3) > 2 else "",
        "nearest_file": pred.neighbors[0][0], "excluded_speakers": " ".join(exclude_speakers),
        "elapsed_ms": round(pred.elapsed_ms), "figure": figure_path.name if figure_path else "",
    })
    return pred, figure_path, excluded_self


def run_recognize(cfg, words, paths, exclude_speakers=(), k=None, save_figure=True):
    """Lệnh recognize: nhận dạng từng file, in kết quả, lưu hình và nhật ký vào results/recognize/."""
    from src.store import load_bank

    recognizer = Recognizer(load_bank(cfg), cfg)
    out_dir = cfg["paths"]["results"] / "recognize"
    low = cfg["knn"]["low_confidence"]
    for path in map(Path, paths):
        if not path.exists():
            print(f"Không thấy file {path}")
            continue
        pred, figure_path, excluded_self = recognize_file(path, cfg, words, recognizer, out_dir,
                                                          exclude_speakers, k, save_figure)
        print()
        print(path.name)
        if excluded_self:
            print("  (file này có sẵn trong kho, đã loại chính nó khỏi kho khi so khớp)")
        if exclude_speakers:
            print(f"  (đã loại khỏi kho: {', '.join(exclude_speakers)})")
        flag = "  KHÔNG CHẮC" if pred.confidence < low else ""
        print(f"  Từ nhận dạng: {words.get(pred.word, pred.word)}   độ tin cậy {pred.confidence:.2f}{flag}")
        for rank, (w, score, dist) in enumerate(pred.top3, 1):
            print(f"    {rank}. {words.get(w, w):<6} điểm {score:7.2f}   khoảng cách gần nhất {dist:.3f}")
        print(f"  Mẫu gần nhất: {pred.neighbors[0][0]}   thời gian so khớp {pred.elapsed_ms:.0f} ms")
        if figure_path:
            print(f"  Đã lưu hình {figure_path}")
