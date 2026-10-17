"""Đóng gói nộp: nén toàn bộ dự án thành một file zip, bỏ môi trường ảo và thư mục đệm."""
import zipfile
from datetime import datetime

from src.common import ROOT

SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", "dist", ".git", ".streamlit"}


def iter_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if path.is_file() and not SKIP_DIRS & set(rel.parts):
            yield path, rel


def make_package(out_dir=None):
    out_dir = out_dir or ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{ROOT.name}_{datetime.now():%Y%m%d-%H%M}.zip"
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, rel in iter_files():
            zf.write(path, f"{ROOT.name}/{rel.as_posix()}")
            n_files += 1
    print(f"Đã tạo {zip_path}")
    print(f"{n_files} tệp, {zip_path.stat().st_size / 1e6:.1f} MB. Không gồm: {', '.join(sorted(SKIP_DIRS))}.")
    return zip_path
