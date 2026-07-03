"""
Тесты тайлового расчёта метрик для больших панорам (>MAX_DIMENSION_WARN px).

Идея: compute_metrics_from_mask_path не должна грузить всю маску в RAM разом,
но обязана давать РОВНО тот же результат, что и compute_metrics(load_mask(...)).

Запуск:  pytest -q
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.metrics import compute_metrics, compute_metrics_from_mask_path
from src.schemas import MLResponse, MLObject


def _ml(conf: float = 0.9) -> MLResponse:
    return MLResponse(
        model_version="test", inference_time_ms=1, inference_params={},
        image_size={"width": 1, "height": 1}, mask_path="",
        class_legend=config.CLASS_NAMES, confidence_map_path=None,
        objects=[MLObject(0, config.CLASS_ORDINARY, [0, 0, 1, 1], 100, conf)],
        warnings=[],
    )


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask, mode="L").save(path)


def test_tiled_matches_full_load_on_irregular_size(tmp_path):
    # Размер НЕ кратен тайлу (проверяем краевые тайлы) и превышает tile_size.
    rng = np.random.default_rng(42)
    h, w = 5300, 4100
    mask = rng.integers(0, 5, size=(h, w), dtype=np.uint8)
    mask_path = tmp_path / "panorama__mask.png"
    _write_mask(mask_path, mask)

    ml = _ml()
    expected = compute_metrics(mask, ml)
    actual = compute_metrics_from_mask_path(str(mask_path), ml, tile_size=2048)

    assert actual.total_px == expected.total_px
    assert actual.valid_px == expected.valid_px
    assert actual.class_area_px == expected.class_area_px
    assert actual.talc_fraction == expected.talc_fraction
    assert actual.sulphide_fraction == expected.sulphide_fraction
    assert actual.fine_of_sulphides == expected.fine_of_sulphides
    assert actual.artifact_fraction == expected.artifact_fraction


def test_tiled_handles_tile_size_larger_than_image(tmp_path):
    # tile_size больше самого изображения — должен просто прочесть один тайл.
    mask = np.full((50, 40), config.CLASS_TALC, dtype=np.uint8)
    mask_path = tmp_path / "small__mask.png"
    _write_mask(mask_path, mask)

    ml = _ml()
    actual = compute_metrics_from_mask_path(str(mask_path), ml, tile_size=2048)
    assert actual.total_px == 50 * 40
    assert actual.class_area_px[config.CLASS_TALC] == 50 * 40


def test_pipeline_routes_large_panorama_through_tiled_path(tmp_path, monkeypatch):
    # run_analysis должен САМ выбрать тайловый путь для панорамы >10000px,
    # не трогая пиксельный результат по сравнению с обычным путём.
    from src import pipeline, ml_client

    h, w = 11000, 300  # одна сторона > MAX_DIMENSION_WARN
    rng = np.random.default_rng(7)
    mask = rng.integers(0, 5, size=(h, w), dtype=np.uint8)
    mask_path = tmp_path / "big__mask.png"
    _write_mask(mask_path, mask)

    fake_ml = MLResponse(
        model_version="test", inference_time_ms=1, inference_params={},
        image_size={"width": w, "height": h}, mask_path=str(mask_path),
        class_legend=config.CLASS_NAMES, confidence_map_path=None,
        objects=[], warnings=[],
    )
    monkeypatch.setattr(ml_client, "analyze", lambda *a, **kw: fake_ml)

    result = pipeline.run_analysis(str(tmp_path / "big.png"))
    expected = compute_metrics(mask, fake_ml)
    assert result.metrics.class_area_px == expected.class_area_px
    assert result.metrics.talc_fraction == expected.talc_fraction
