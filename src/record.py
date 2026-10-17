"""Thu âm thật từ micro, lưu đúng quy ước <từ>_<người>_<số>.wav."""
import numpy as np
import sounddevice as sd
import soundfile as sf

from src.common import CODE_PATTERN, make_filename, parse_filename

QUIET_PEAK = 0.05


def next_index(raw_dir, word, speaker):
    used = [0]
    for path in raw_dir.glob(f"{word}_{speaker}_*.wav"):
        parsed = parse_filename(path.stem)
        if parsed and parsed[0] == word and parsed[1] == speaker:
            used.append(parsed[2])
    return max(used) + 1


def record_once(seconds, sr):
    input("  Nhấn Enter rồi đọc ngay...")
    print("  Đang ghi...", end="", flush=True)
    y = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print(" xong.")
    return y[:, 0]


def record_session(cfg, words, speaker, repeats, only_words, seconds):
    if not CODE_PATTERN.match(speaker):
        raise SystemExit("Mã người chỉ gồm chữ thường không dấu và số, ví dụ: an, binh2")
    if only_words:
        unknown = set(only_words) - set(words)
        if unknown:
            raise SystemExit(f"Mã từ không có trong words.txt: {', '.join(sorted(unknown))}")
        words = {code: text for code, text in words.items() if code in only_words}

    sr = cfg["audio"]["sample_rate"]
    raw_dir = cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = sd.query_devices(kind="input")
    print(f"Micro: {device['name']}. Người đọc: {speaker}. {len(words)} từ x {repeats} lần.")
    print("Mỗi lần: nhấn Enter, đọc rõ một từ trong khoảng", seconds, "giây.\n")

    for code, text in words.items():
        for _ in range(repeats):
            index = next_index(raw_dir, code, speaker)
            print(f'Từ: "{text}"  (lần {index})')
            while True:
                y = record_once(seconds, sr)
                peak = float(np.max(np.abs(y)))
                if peak < QUIET_PEAK:
                    print(f"  Cảnh báo: tiếng quá nhỏ (đỉnh {peak:.3f}), nên thu lại.")
                answer = input("  [Enter] lưu, [n] nghe lại, [r] thu lại, [q] thoát: ").strip().lower()
                while answer == "n":
                    sd.play(y, sr)
                    sd.wait()
                    answer = input("  [Enter] lưu, [n] nghe lại, [r] thu lại, [q] thoát: ").strip().lower()
                if answer == "q":
                    print("Dừng thu âm.")
                    return
                if answer == "r":
                    continue
                path = raw_dir / make_filename(code, speaker, index)
                sf.write(path, y, sr, subtype="PCM_16")
                print(f"  Đã lưu {path.name}\n")
                break
    print("Thu âm xong. Chạy 'python stt.py inventory' để kiểm tra.")
