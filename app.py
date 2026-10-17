"""Giao diện demo: ghi âm, tải file hoặc chọn file mẫu, nhận dạng và hiện kết quả.

Chạy: python stt.py demo   (hoặc: python -m streamlit run app.py)
Mỗi lần nhận dạng lưu file đầu vào, hình kết quả và một dòng log.csv vào results/demo/.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import load_config, load_words  # noqa: E402
from src.evaluate import speaker_group  # noqa: E402
from src.recognizer import FeatureTransform, Recognizer, recognize_file  # noqa: E402
from src.store import FRAMES_FILE, load_bank  # noqa: E402

SOURCES = ["Ghi âm từ micro", "Tải file lên", "Chọn file mẫu trong kho"]
F0_CHOICES = [0.0, 1.0, 2.0, 4.0]

st.set_page_config(page_title="Nhận dạng từ đơn tiếng Việt", layout="wide")


@st.cache_resource(show_spinner="Đang nạp kho đặc trưng...")
def load_resources():
    cfg = load_config()
    if not (cfg["paths"]["features"] / FRAMES_FILE).exists():
        return cfg, load_words(), None
    return cfg, load_words(), load_bank(cfg)


@st.cache_resource(show_spinner="Đang chuẩn bị bộ nhận dạng...")
def get_recognizer(f0_weight):
    cfg, _, bank = load_resources()
    return Recognizer(bank, cfg, FeatureTransform(cfg, f0_weight).fit(bank.sequences))


cfg, words, bank = load_resources()
out_dir = Path(os.environ.get("STT_DEMO_DIR", cfg["paths"]["results"] / "demo"))

st.title("Nhận dạng từ đơn tiếng Việt")
st.caption("MFCC + F0, so khớp DTW với kho mẫu, bỏ phiếu k láng giềng gần nhất.")

if bank is None:
    st.error("Chưa có kho đặc trưng. Chạy lần lượt: `python stt.py gen-data` rồi `python stt.py build-features`.")
    st.stop()

speakers = sorted(bank.meta["speaker"].unique())
with st.sidebar:
    st.header("Kho mẫu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Từ", bank.meta["word"].nunique())
    col2.metric("Người", len(speakers))
    col3.metric("Mẫu", len(bank.meta))
    st.caption("Danh sách từ: " + ", ".join(words.values()))

    st.header("Tham số")
    k = st.slider("Số láng giềng k", 1, 9, cfg["knn"]["k"], step=2)
    f0_weight = st.select_slider("Trọng số F0 (thanh điệu)", F0_CHOICES, value=float(cfg["features"]["f0_weight"]))

    st.header("Giả lập giọng mới")
    groups = sorted({speaker_group(s) for s in speakers} | set(speakers))
    exclude = st.multiselect(
        "Loại khỏi kho khi so khớp", groups,
        help="Chọn một người hoặc cả nhóm giọng (ví dụ tts_f1) để thử như người chưa từng có trong kho.")

source = st.radio("Nguồn âm thanh", SOURCES, horizontal=True)
audio_path, audio_bytes, suffix = None, None, ".wav"
if source == SOURCES[0]:
    recorded = st.audio_input("Bấm nút micro, đọc rõ một từ trong danh sách, rồi bấm dừng")
    if recorded is not None:
        audio_bytes = recorded.getvalue()
elif source == SOURCES[1]:
    uploaded = st.file_uploader("Chọn file âm thanh chứa một từ", type=["wav", "flac", "ogg", "mp3"])
    if uploaded is not None:
        audio_bytes, suffix = uploaded.getvalue(), Path(uploaded.name).suffix.lower()
        st.audio(audio_bytes)
else:
    raw_files = sorted(p.stem for p in cfg["paths"]["raw"].glob("*.wav"))
    stem = st.selectbox("File mẫu", raw_files,
                        help="File trong kho được tự loại khỏi kho khi so khớp, để không tự khớp với chính nó.")
    if stem:
        audio_path = cfg["paths"]["raw"] / f"{stem}.wav"
        st.audio(str(audio_path))

if st.button("Nhận dạng", type="primary", disabled=audio_path is None and audio_bytes is None):
    if audio_bytes is not None:
        inputs = out_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        kind = "micro" if source == SOURCES[0] else "tai_len"
        audio_path = inputs / f"{kind}_{datetime.now():%Y%m%d-%H%M%S}{suffix}"
        audio_path.write_bytes(audio_bytes)
    with st.spinner("Đang trích đặc trưng và so khớp..."):
        pred, figure, excluded_self = recognize_file(audio_path, cfg, words, get_recognizer(f0_weight),
                                                     out_dir, tuple(exclude), k)
    st.session_state["result"] = {"pred": pred, "figure": figure, "excluded_self": excluded_self,
                                  "path": audio_path, "exclude": list(exclude), "k": k, "f0": f0_weight}

result = st.session_state.get("result")
if result:
    pred = result["pred"]
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.markdown("Từ nhận dạng")
        st.markdown(f"<h1 style='margin-top:0'>{words.get(pred.word, pred.word)}</h1>", unsafe_allow_html=True)
        st.metric("Độ tin cậy", f"{pred.confidence:.0%}")
        if pred.confidence < cfg["knn"]["low_confidence"]:
            st.warning("Độ tin cậy thấp, kết quả không chắc chắn.")
        st.dataframe(pd.DataFrame(
            [{"Hạng": i, "Từ": words.get(w, w), "Điểm": round(score, 2), "Khoảng cách gần nhất": round(dist, 3)}
             for i, (w, score, dist) in enumerate(pred.top3, 1)]), hide_index=True)
        st.caption("Láng giềng gần nhất: " + ", ".join(f"{fid} ({d:.2f})" for fid, _, d in pred.neighbors))
        notes = [f"k = {result['k']}, trọng số F0 = {result['f0']:g}, so khớp {pred.elapsed_ms:.0f} ms."]
        if result["excluded_self"]:
            notes.append("File có sẵn trong kho nên đã loại chính nó khi so khớp.")
        if result["exclude"]:
            notes.append("Đã loại khỏi kho: " + ", ".join(result["exclude"]) + ".")
        st.caption(" ".join(notes))
    with right:
        if result["figure"]:
            st.image(str(result["figure"]))
            st.caption(f"Đã lưu hình: {result['figure']}  |  Đầu vào: {result['path']}")
