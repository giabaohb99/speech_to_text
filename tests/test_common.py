import pytest

from src.common import load_words, make_filename, parse_filename
from src.record import next_index


def test_parse_filename_with_underscored_speaker():
    assert parse_filename("mot_tts_f1_p2_03") == ("mot", "tts_f1_p2", 3)
    assert parse_filename("xuong_an_12") == ("xuong", "an", 12)


@pytest.mark.parametrize("stem", ["mot_an", "mot_an_x1", "mot"])
def test_parse_filename_rejects_bad_names(stem):
    assert parse_filename(stem) is None


def test_make_and_parse_roundtrip():
    name = make_filename("bon", "binh", 7)
    assert name == "bon_binh_07.wav"
    assert parse_filename(name[:-4]) == ("bon", "binh", 7)


def test_load_words_project_file_has_20_words():
    words = load_words()
    assert len(words) == 20
    assert words["mot"] == "một"


def test_load_words_rejects_bad_code(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("mot_1,một\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_words(path)


def test_next_index_continues_after_existing(tmp_path):
    for name in ["mot_an_01.wav", "mot_an_03.wav", "mot_binh_09.wav", "hai_an_05.wav"]:
        (tmp_path / name).touch()
    assert next_index(tmp_path, "mot", "an") == 4
    assert next_index(tmp_path, "ba", "an") == 1
