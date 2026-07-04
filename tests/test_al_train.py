"""
Тесты модуля дообучения active learning (ml_service/train.py) — без torch.

Проверяем «холодную» часть: разбор ImageFolder-экспорта патчей и манифеста,
маппинг кодов классов контракта в индексы выхода модели, стратифицированное
разбиение. Собственно обучение (finetune) требует torch и здесь не гоняется.

Запуск:  pytest -q tests/test_al_train.py
"""

import csv

from PIL import Image

from ml_service import train as t
from ml_service.model import MODEL_TO_CONTRACT


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 120, 120)).save(path)


def test_contract_to_model_is_inverse_of_model_to_contract():
    # 0 talc->3, 1 ordinary->1, 2 fine->2 (см. ml_service/model.py)
    for model_idx, contract in enumerate(MODEL_TO_CONTRACT.tolist()):
        assert t.CONTRACT_TO_MODEL[int(contract)] == model_idx
    # фон(0) и артефакт(4) не тренируемы
    assert 0 not in t.CONTRACT_TO_MODEL
    assert 4 not in t.CONTRACT_TO_MODEL


def _write_manifest(export_dir, rows):
    fields = ["path", "label", "label_name", "weight"]
    with open(export_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_read_patch_export_maps_labels_and_weights(tmp_path):
    export = tmp_path / "exp1"
    (export / "imgs").mkdir(parents=True)
    _img(export / "imgs" / "talc" / "a.jpg")
    _img(export / "imgs" / "fine_intergrowth" / "b.jpg")
    _img(export / "imgs" / "ordinary_intergrowth" / "c.jpg")
    _write_manifest(export, [
        {"path": "imgs/talc/a.jpg", "label": 3, "label_name": "talc", "weight": 0.9},
        {"path": "imgs/fine_intergrowth/b.jpg", "label": 2, "label_name": "fine_intergrowth", "weight": 0.5},
        {"path": "imgs/ordinary_intergrowth/c.jpg", "label": 1, "label_name": "ordinary_intergrowth", "weight": 1.0},
        # фоновый/артефактный класс должен быть пропущен
        {"path": "imgs/bg/x.jpg", "label": 0, "label_name": "unlabeled", "weight": 1.0},
    ])

    samples = t.read_patch_export(export)
    assert len(samples) == 3  # фон пропущен
    by_contract = {s.contract_label: s for s in samples}
    assert by_contract[3].model_idx == 0  # talc
    assert by_contract[1].model_idx == 1  # ordinary
    assert by_contract[2].model_idx == 2  # fine
    assert abs(by_contract[2].weight - 0.5) < 1e-6
    assert t.class_histogram(samples) == {"talc": 1, "ordinary": 1, "fine": 1}


def test_read_patch_export_fallback_without_manifest(tmp_path):
    export = tmp_path / "exp2"
    _img(export / "imgs" / "talc" / "a.jpg")
    _img(export / "imgs" / "talc" / "b.jpg")
    _img(export / "imgs" / "fine" / "c.jpg")
    # манифеста нет — читаем структуру папок
    samples = t.read_patch_export(export)
    assert t.class_histogram(samples) == {"talc": 2, "ordinary": 0, "fine": 1}
    assert all(s.weight == 1.0 for s in samples)


def test_read_base_dataset_imagefolder(tmp_path):
    root = tmp_path / "base"
    _img(root / "ordinary" / "a.jpg")
    _img(root / "fine" / "b.jpg")
    _img(root / "talc" / "c.jpg")
    _img(root / "junk_folder" / "d.jpg")  # неизвестный класс — игнор
    samples = t.read_base_dataset(root)
    assert t.class_histogram(samples) == {"talc": 1, "ordinary": 1, "fine": 1}


def test_stratified_split_keeps_at_least_one_per_class_in_train():
    samples = (
        [t.Sample("p", 0, 1.0, 3)]                       # ровно 1 talc
        + [t.Sample(f"o{i}", 1, 1.0, 1) for i in range(10)]
        + [t.Sample(f"f{i}", 2, 1.0, 2) for i in range(10)]
    )
    train, val = t.stratified_split(samples, val_frac=0.5, seed=1)
    # единственный talc остаётся в train (val не может его забрать)
    assert t.class_histogram(train)["talc"] == 1
    assert t.class_histogram(val)["talc"] == 0
    # прочие классы разбиты
    assert t.class_histogram(train)["ordinary"] >= 1
    assert t.class_histogram(val)["ordinary"] >= 1
    assert len(train) + len(val) == len(samples)
