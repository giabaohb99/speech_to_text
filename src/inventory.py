"""Kiểm kê kho âm thanh: đếm theo từ và theo người, báo file sai quy ước hoặc sai định dạng."""
import pandas as pd
import soundfile as sf

from src.common import parse_filename


def run_inventory(cfg, words):
    acfg = cfg["audio"]
    raw_dir = cfg["paths"]["raw"]
    if not raw_dir.exists():
        print(f"Chưa có thư mục {raw_dir}. Chạy 'python stt.py gen-data' hoặc 'record' trước.")
        return 1

    rows, errors, warnings = [], [], []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() != ".wav":
            errors.append(f"{path.name}: không phải file .wav")
            continue
        parsed = parse_filename(path.stem)
        if parsed is None:
            errors.append(f"{path.name}: sai quy ước <từ>_<người>_<số>.wav")
            continue
        word, speaker, index = parsed
        if word not in words:
            errors.append(f"{path.name}: mã từ '{word}' không có trong words.txt")
            continue
        try:
            info = sf.info(path)
        except RuntimeError as exc:
            errors.append(f"{path.name}: không đọc được ({exc})")
            continue
        if info.samplerate != acfg["sample_rate"] or info.channels != 1:
            warnings.append(f"{path.name}: {info.samplerate} Hz, {info.channels} kênh (sẽ được đổi khi nạp)")
        if not acfg["min_duration_s"] <= info.duration <= acfg["max_duration_s"]:
            warnings.append(f"{path.name}: dài {info.duration:.2f} giây, ngoài khoảng cho phép")
        rows.append({"word": word, "speaker": speaker, "duration": info.duration})

    if not rows:
        print("Kho rỗng.")
        return 1

    df = pd.DataFrame(rows)
    table = pd.crosstab(df["speaker"], df["word"]).reindex(columns=list(words), fill_value=0)
    print(f"Tổng {len(df)} file, {df['word'].nunique()}/{len(words)} từ, {df['speaker'].nunique()} người.")
    print(f"Độ dài: nhỏ nhất {df['duration'].min():.2f} s, trung bình {df['duration'].mean():.2f} s, "
          f"lớn nhất {df['duration'].max():.2f} s.\n")
    print("Số file theo người (hàng) và từ (cột):")
    print(table.to_string())

    missing_words = [code for code in words if table[code].sum() == 0]
    empty_cells = [(s, w) for s in table.index for w in words if table.at[s, w] == 0]
    print()
    if missing_words:
        print("Từ chưa có file nào:", ", ".join(missing_words))
    if empty_cells:
        print(f"{len(empty_cells)} cặp người-từ chưa có file, ví dụ:",
              ", ".join(f"{s}/{w}" for s, w in empty_cells[:10]))
    for message in warnings[:20]:
        print("Cảnh báo:", message)
    if len(warnings) > 20:
        print(f"... và {len(warnings) - 20} cảnh báo khác")
    for message in errors:
        print("Lỗi:", message)
    if not (missing_words or empty_cells or warnings or errors):
        print("Kho đầy đủ, không có lỗi.")
    return 1 if (missing_words or errors) else 0
