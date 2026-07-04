"""
Тесты косметической талько-подсветки (talc_cosmetic).

Слой чисто ВИЗУАЛЬНЫЙ (палитровая сегментация + density-сборка «области
оталькования»): проверяем, что pipeline не падает, возвращает overlay/region
корректной формы, детектит явную талько-зону и переживает вырожденные входы.

Запуск:  pytest -q tests/test_talc_cosmetic.py
"""

import numpy as np
import pytest

from talc_cosmetic import compute_talc_overlay, density_heatmap, TalcOverlayResult


def _talc_image(w=320, h=240):
    """Синтетика: светлый фон + крупная тёмная (талько-подобная) клякса слева."""
    rng = np.random.default_rng(0)
    img = rng.integers(150, 210, (h, w, 3), dtype=np.uint8)  # светлая матрица
    img[40:200, 30:150] = rng.integers(5, 35, (160, 120, 3), dtype=np.uint8)  # тёмное пятно
    return img


def test_overlay_shape_and_types():
    rgb = _talc_image()
    res = compute_talc_overlay(rgb, seg_max_side=200)
    assert isinstance(res, TalcOverlayResult)
    assert res.overlay.shape == rgb.shape
    assert res.overlay.dtype == np.uint8
    assert res.region.shape == rgb.shape[:2]
    assert res.region.dtype == bool
    assert 0.0 <= res.region_pct <= 100.0
    assert 0.0 <= res.talc_raw_pct <= 100.0
    assert res.palette_name in {"панорамная", "жёлтая"}


def test_detects_dark_region():
    """На картинке с крупной тёмной кляксой красная зона должна появиться."""
    rgb = _talc_image()
    res = compute_talc_overlay(rgb, seg_max_side=220)
    assert res.region.any(), "ожидали непустую талько-зону на тёмном пятне"
    # красный оверлей отличается от оригинала внутри зоны
    assert not np.array_equal(res.overlay[res.region], rgb[res.region])


def test_forced_palette():
    rgb = _talc_image()
    res = compute_talc_overlay(rgb, seg_max_side=180, palette="жёлтая")
    assert res.palette_name == "жёлтая"


def test_uniform_image_no_crash():
    """Сплошной кадр без структуры не должен ронять pipeline."""
    rgb = np.full((128, 160, 3), 180, dtype=np.uint8)
    res = compute_talc_overlay(rgb, seg_max_side=120)
    assert res.overlay.shape == rgb.shape
    # на равномерном светлом фоне талька быть не должно
    assert res.talc_raw_pct == pytest.approx(0.0, abs=1.0)


def test_density_heatmap():
    rgb = _talc_image()
    res = compute_talc_overlay(rgb, seg_max_side=180)
    hm = density_heatmap(res.density)
    assert hm.dtype == np.uint8
    assert hm.shape[-1] == 3


def test_cosmetic_does_not_touch_input():
    rgb = _talc_image()
    before = rgb.copy()
    compute_talc_overlay(rgb, seg_max_side=180)
    assert np.array_equal(rgb, before), "входной массив не должен меняться"
