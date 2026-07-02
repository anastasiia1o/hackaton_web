"""
Тесты геологической логики. Это ГЛАВНАЯ проверка корректности:
геологические правила должны срабатывать ровно как в постановке задачи.

Запуск:  pytest -q
"""

import numpy as np

from src import config
from src.metrics import compute_metrics
from src.classification import classify
from src.schemas import (
    MLResponse, MLObject,
    ORE_TALC, ORE_ORDINARY, ORE_REFRACTORY, ORE_REVIEW,
)


def _mask_from_counts(counts: dict[int, int], total: int = 10000) -> np.ndarray:
    """Собрать 1D-маску с заданным числом пикселей на класс."""
    pixels = []
    for cls, n in counts.items():
        pixels += [cls] * n
    pixels += [config.CLASS_BACKGROUND] * (total - len(pixels))
    return np.array(pixels, dtype=np.uint8)


def _ml(conf: float = 0.9) -> MLResponse:
    return MLResponse(
        model_version="test", inference_time_ms=1, inference_params={},
        image_size={"width": 100, "height": 100}, mask_path="",
        class_legend=config.CLASS_NAMES, confidence_map_path=None,
        objects=[MLObject(0, config.CLASS_ORDINARY, [0, 0, 1, 1], 100, conf)],
        warnings=[],
    )


def test_talc_priority():
    # Тальк 15% валидной площади -> оталькованная, даже если тонких мало.
    mask = _mask_from_counts({
        config.CLASS_TALC: 1500,
        config.CLASS_ORDINARY: 800,
        config.CLASS_FINE: 200,
    })
    m = compute_metrics(mask, _ml())
    c = classify(m)
    assert c.ore_class == ORE_TALC
    assert m.talc_fraction > config.TALC_THRESHOLD


def test_refractory_when_fine_dominates():
    # Тальк мало (2%), тонких больше -> труднообогатимая.
    mask = _mask_from_counts({
        config.CLASS_TALC: 200,
        config.CLASS_ORDINARY: 1000,
        config.CLASS_FINE: 2000,
    })
    m = compute_metrics(mask, _ml())
    c = classify(m)
    assert c.ore_class == ORE_REFRACTORY
    assert m.fine_of_sulphides > 0.5


def test_ordinary_when_ordinary_dominates():
    mask = _mask_from_counts({
        config.CLASS_TALC: 200,
        config.CLASS_ORDINARY: 2500,
        config.CLASS_FINE: 500,
    })
    m = compute_metrics(mask, _ml())
    c = classify(m)
    assert c.ore_class == ORE_ORDINARY


def test_borderline_talc_needs_review():
    # Тальк ~10% (пограничная зона 9-11%) -> экспертная проверка.
    mask = _mask_from_counts({
        config.CLASS_TALC: 1000,
        config.CLASS_ORDINARY: 1500,
        config.CLASS_FINE: 500,
    })
    m = compute_metrics(mask, _ml())
    c = classify(m)
    assert c.ore_class == ORE_REVIEW
    assert c.needs_review


def test_low_confidence_needs_review():
    mask = _mask_from_counts({
        config.CLASS_TALC: 100,
        config.CLASS_ORDINARY: 2000,
        config.CLASS_FINE: 500,
    })
    m = compute_metrics(mask, _ml(conf=0.4))  # низкая уверенность
    c = classify(m)
    assert c.ore_class == ORE_REVIEW


def test_valid_area_excludes_artifacts():
    # Артефакты не входят в валидную площадь.
    mask = _mask_from_counts({
        config.CLASS_ARTIFACT: 5000,
        config.CLASS_TALC: 100,
        config.CLASS_ORDINARY: 900,
    }, total=10000)
    m = compute_metrics(mask, _ml())
    assert m.valid_px == 10000 - 5000
    assert m.artifact_fraction == 0.5
