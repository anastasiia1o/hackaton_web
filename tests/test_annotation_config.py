"""Тесты конфигурируемых классов разметки (src/annotation_config.py)."""

import json

from src import annotation_config as ac


def test_load_default_classes_from_repo_config():
    classes = ac.load_classes()
    names = {c.name for c in classes}
    assert names == {"unlabeled", "talc", "ordinary_intergrowth", "fine_intergrowth", "uncertain"}
    by_id = {c.id: c for c in classes}
    assert by_id[0].color == (0, 0, 0, 0)          # неразмеченная — прозрачная
    assert by_id[1].name_ru == "Тальк"


def test_load_classes_falls_back_on_missing_file(tmp_path):
    classes = ac.load_classes(path=tmp_path / "missing.json")
    assert len(classes) == 5  # дефолтные встроенные классы


def test_load_classes_custom_file(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({"classes": [
        {"id": 0, "name": "bg", "name_ru": "Фон", "color": [0, 0, 0, 0]},
        {"id": 1, "name": "ore", "name_ru": "Руда", "color": [255, 0, 0, 200]},
    ]}), encoding="utf-8")
    classes = ac.load_classes(path=custom)
    assert len(classes) == 2
    assert classes[1].name == "ore"


def test_classes_json_dict_shape():
    d = ac.classes_json_dict()
    assert set(d.keys()) == {"0", "1", "2", "3", "4"}
    assert d["1"]["name"] == "talc"
    assert "color" in d["1"]
