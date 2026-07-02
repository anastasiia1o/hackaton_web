"""
Тесты валидатора контракта (src/contract.py).

Проверяем, что: (1) корректный mock-ответ проходит без ошибок; (2) на каждое
типовое нарушение контракта валидатор выдаёт понятную ошибку.

Запуск:  pytest -q tests/test_contract.py
"""

import copy

from src import config
from src.contract import validate_ml_response, assert_valid, ContractError
from mock_ml.generator import generate


def _good_response(tmp_path) -> dict:
    """Реальный mock-ответ по контракту (с настоящим файлом маски)."""
    # исходник не нужен реально, generate создаёт маску сам
    img = tmp_path / "slide.png"
    from PIL import Image
    Image.new("RGB", (300, 200), (50, 50, 50)).save(img)
    return generate(str(img), out_dir=tmp_path / "out", size=(300, 200))


def test_good_response_passes(tmp_path):
    resp = _good_response(tmp_path)
    errors = [e for e in validate_ml_response(resp) if not e.startswith("[warning]")]
    assert errors == []
    assert_valid(resp)  # не должно бросить


def test_missing_field_detected(tmp_path):
    resp = _good_response(tmp_path)
    del resp["confidence_map"]           # желательное поле → только warning
    del resp["mask"]                     # обязательное → жёсткая ошибка
    errors = validate_ml_response(resp, check_mask_file=False)
    assert any("mask" in e and not e.startswith("[warning]") for e in errors)


def test_bad_class_code_detected(tmp_path):
    resp = _good_response(tmp_path)
    resp["objects"][0]["class"] = 9      # недопустимый класс
    errors = validate_ml_response(resp, check_mask_file=False)
    assert any("class" in e for e in errors)


def test_size_mismatch_detected(tmp_path):
    resp = _good_response(tmp_path)
    resp["image_size"] = {"width": 999, "height": 999}  # не совпадает с маской
    errors = validate_ml_response(resp)  # с чтением файла
    assert any("не совпадает" in e for e in errors)


def test_assert_valid_raises(tmp_path):
    resp = _good_response(tmp_path)
    del resp["model_version"]
    try:
        assert_valid(resp, check_mask_file=False)
        assert False, "ожидалась ContractError"
    except ContractError as e:
        assert "model_version" in str(e)
