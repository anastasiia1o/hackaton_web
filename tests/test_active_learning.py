"""
Тесты интерактивного активного обучения (src/active_learning.py) — без torch.

Проверяем «холодную» оркестрацию: маппинг кодов класса в индексы выхода модели
и сборку якорных патчей (случайные тайлы с текущим предсказанием, исключение
исправленных зон, пропуск фона). Само дообучение/переинференс требуют torch и
здесь не гоняются.

Запуск:  pytest -q tests/test_active_learning.py
"""

import numpy as np
from PIL import Image

from src import active_learning as al
from src import config


def test_contract_to_model_mapping():
    c2m = al._contract_to_model()
    assert c2m[config.CLASS_TALC] == 0
    assert c2m[config.CLASS_ORDINARY] == 1
    assert c2m[config.CLASS_FINE] == 2
    assert config.CLASS_BACKGROUND not in c2m
    assert config.CLASS_ARTIFACT not in c2m


def test_build_anchor_items_labels_and_weight():
    W, H = 200, 200
    base = Image.new("RGB", (W, H), (100, 100, 100))
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:, : W // 2] = config.CLASS_TALC   # левая половина — тальк
    mask[:, W // 2:] = config.CLASS_FINE    # правая половина — тонкие

    items = al.build_anchor_items(base, mask, corrections=[], n=6, tile_frac=0.15, seed=1)
    assert 0 < len(items) <= 6
    for crop, idx, wt in items:
        assert idx in (0, 2)          # talc(0) или fine(2), не ordinary
        assert wt == 0.3
        assert isinstance(crop, Image.Image)


def test_build_anchor_items_skips_background():
    base = Image.new("RGB", (128, 128), (0, 0, 0))
    mask = np.zeros((128, 128), dtype=np.uint8)  # весь фон — нетренируемо
    items = al.build_anchor_items(base, mask, corrections=[], n=8, seed=0)
    assert items == []


def test_build_anchor_items_excludes_correction_region():
    W, H = 160, 160
    base = Image.new("RGB", (W, H), (100, 100, 100))
    mask = np.full((H, W), config.CLASS_TALC, dtype=np.uint8)
    # исправление накрывает весь кадр → все центры тайлов внутри excl → нет якорей
    corrections = [{"region_fraction": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}}]
    items = al.build_anchor_items(base, mask, corrections, n=8, tile_frac=0.2, seed=0)
    assert items == []
