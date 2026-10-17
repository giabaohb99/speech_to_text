"""Cấu hình, danh sách từ và quy ước tên file dùng chung."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CODE_PATTERN = re.compile(r"^[a-z0-9]+$")


def load_config(path=None):
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: ROOT / value for key, value in cfg["paths"].items()}
    return cfg


def load_words(path=None):
    """Đọc words.txt, trả về dict mã không dấu -> từ có dấu, giữ đúng thứ tự."""
    words = {}
    lines = Path(path or ROOT / "words.txt").read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        code, _, text = (part.strip() for part in line.partition(","))
        if not CODE_PATTERN.match(code) or not text:
            raise ValueError(f"words.txt dòng {line_no} sai định dạng 'ma,từ': {line!r}")
        if code in words:
            raise ValueError(f"words.txt dòng {line_no}: mã {code!r} bị trùng")
        words[code] = text
    return words


def parse_filename(stem):
    """'mot_tts_f1_p2_03' -> ('mot', 'tts_f1_p2', 3). Sai quy ước thì trả None."""
    parts = stem.split("_")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    return parts[0], "_".join(parts[1:-1]), int(parts[-1])


def make_filename(word, speaker, index):
    return f"{word}_{speaker}_{index:02d}.wav"
