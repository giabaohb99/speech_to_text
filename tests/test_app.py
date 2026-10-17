"""Kiểm thử giao diện bằng streamlit AppTest trên kho thật; bỏ qua nếu chưa dựng kho đặc trưng."""
import pytest
from streamlit.testing.v1 import AppTest

from src.common import ROOT, load_config
from src.store import FRAMES_FILE

pytestmark = pytest.mark.skipif(not (load_config()["paths"]["features"] / FRAMES_FILE).exists(),
                                reason="chưa có kho đặc trưng")


def run_app(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_DEMO_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_recognize_sample_file_saves_figure_and_log(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path)
    at.radio[0].set_value("Chọn file mẫu trong kho").run()
    at.selectbox[0].set_value("tam_tts_f1_p0_01").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("tám" in m.value for m in at.markdown)
    assert any("đã loại chính nó" in c.value for c in at.caption)
    assert len(list(tmp_path.glob("tam_tts_f1_p0_01_*.png"))) == 1
    assert (tmp_path / "log.csv").exists()


def test_exclude_group_in_sidebar(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path)
    at.radio[0].set_value("Chọn file mẫu trong kho").run()
    at.selectbox[0].set_value("mot_tts_f1_p0_01").run()
    at.sidebar.multiselect[0].set_value(["tts_f1"]).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Đã loại khỏi kho: tts_f1" in c.value for c in at.caption)
    neighbors = next(c.value for c in at.caption if c.value.startswith("Láng giềng"))
    assert "tts_f1" not in neighbors
