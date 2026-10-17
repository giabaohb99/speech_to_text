"""Kho đặc trưng: dựng từ data/raw, ghi hai file csv, đọc lại vào bộ nhớ.

- features_frames.csv  : một dòng một khung, dùng cho DTW
- features_summary.csv : một dòng một mẫu, xem theo từ và theo người, dùng cho SVM

Thống kê chuẩn hóa không lưu ở đây: bộ nhận dạng tự tính từ đúng tập train nó dùng
(recognizer.FeatureTransform), để khi đánh giá không rò rỉ dữ liệu của tập test.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import parse_filename
from src.features import DTW_COLS, FRAME_COLS, extract, summarize
from src.preprocess import load_and_clean

KEY_COLS = ["file_id", "word", "speaker", "sample_idx"]
FRAMES_FILE = "features_frames.csv"
SUMMARY_FILE = "features_summary.csv"
CSV_ENCODING = "utf-8-sig"  # Excel nhận đúng tiếng Việt


def extract_file(path, cfg):
    """Tiền xử lý và trích đặc trưng một file. Dùng chung cho dựng kho và nhận dạng.

    Trả về (DataFrame theo khung, info); info["trimmed"] cho biết có cắt được lặng không.
    """
    y, cut = load_and_clean(path, cfg)
    frames, info = extract(y, cfg["audio"]["sample_rate"], cfg)
    return frames, {**info, "trimmed": cut["trimmed"]}


def build_bank(cfg, words):
    raw_dir, out_dir = cfg["paths"]["raw"], cfg["paths"]["features"]
    paths = sorted(raw_dir.glob("*.wav"))
    if not paths:
        raise SystemExit(f"Không có file wav trong {raw_dir}. Chạy 'python stt.py gen-data' trước.")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_parts, summary_rows, skipped, untrimmed = [], [], [], []
    started = time.perf_counter()
    for i, path in enumerate(paths, 1):
        parsed = parse_filename(path.stem)
        if parsed is None or parsed[0] not in words:
            skipped.append(path.name)
            continue
        word, speaker, sample_idx = parsed
        frames, info = extract_file(path, cfg)
        if not info["trimmed"]:
            untrimmed.append(path.name)
        keys = {"file_id": path.stem, "word": word, "speaker": speaker, "sample_idx": sample_idx}
        frame_parts.append(frames.assign(**keys))
        summary_rows.append({**keys, "text": words[word], **summarize(frames, info)})
        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} file, {time.perf_counter() - started:.0f} s")

    frames_df = pd.concat(frame_parts, ignore_index=True)[KEY_COLS + FRAME_COLS]
    summary_df = pd.DataFrame(summary_rows)
    frames_df.to_csv(out_dir / FRAMES_FILE, index=False, float_format="%.5f", encoding=CSV_ENCODING)
    summary_df.to_csv(out_dir / SUMMARY_FILE, index=False, float_format="%.5f", encoding=CSV_ENCODING)

    for name in skipped:
        print("Bỏ qua (sai quy ước tên hoặc từ lạ):", name)
    for name in untrimmed:
        print("Không cắt được lặng, dùng nguyên file (nên nghe lại):", name)
    print(f"Đã ghi {len(summary_df)} mẫu, {len(frames_df)} khung vào {out_dir}")
    return summary_df


@dataclass
class Bank:
    """Kho đặc trưng trong bộ nhớ. sequences[i] là ma trận (số khung x len(DTW_COLS)) chưa chuẩn hóa."""
    meta: pd.DataFrame          # một dòng một mẫu: file_id, word, speaker, sample_idx
    sequences: list


def load_bank(cfg):
    out_dir = cfg["paths"]["features"]
    if not (out_dir / FRAMES_FILE).exists():
        raise SystemExit("Chưa có kho đặc trưng. Chạy 'python stt.py build-features' trước.")
    frames = pd.read_csv(out_dir / FRAMES_FILE, encoding=CSV_ENCODING)
    meta, sequences = [], []
    for file_id, group in frames.groupby("file_id", sort=False):
        first = group.iloc[0]
        meta.append({"file_id": file_id, "word": first["word"], "speaker": first["speaker"],
                     "sample_idx": int(first["sample_idx"])})
        sequences.append(group.sort_values("frame_idx")[DTW_COLS].to_numpy(dtype=np.float32))
    return Bank(pd.DataFrame(meta), sequences)
