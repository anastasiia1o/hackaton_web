"""Тесты конфигурируемых классов разметки (src/annotation_config.py)."""

import json

from src import annotation_config as ac


def test_load_default_classes_from_repo_config():
    from src import config
    classes = ac.load_classes()
    names = {c.name for c in classes}
    assert names == {"unlabeled", "talc", "ordinary_intergrowth", "fine_intergrowth", "uncertain"}
    by_id = {c.id: c for c in classes}
    assert by_id[0].color == (0, 0, 0, 0)          # неразмеченная — прозрачная
    # patch-AL: id разметки = коды контракта (config.CLASS_*).
    assert by_id[config.CLASS_TALC].name == "talc"
    assert by_id[config.CLASS_ORDINARY].name == "ordinary_intergrowth"
    assert by_id[config.CLASS_FINE].name == "fine_intergrowth"


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
    assert d["3"]["name"] == "talc"          # patch-AL: тальк = код 3 (контракт)
    assert "color" in d["3"]


def test_ids_match_contract_codes():
    """id разметки совпадают с кодами контракта (единое пространство, §3)."""
    from src import config
    by_id = {c.id: c for c in ac.load_classes()}
    assert by_id[config.CLASS_ORDINARY].name == "ordinary_intergrowth"
    assert by_id[config.CLASS_FINE].name == "fine_intergrowth"
    assert by_id[config.CLASS_TALC].name == "talc"
    assert ac.TRAINABLE_CLASS_IDS == (
        config.CLASS_ORDINARY, config.CLASS_FINE, config.CLASS_TALC
    )
    assert ac.UNCERTAIN_ID == config.CLASS_ARTIFACT
