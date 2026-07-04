"""
Тесты квантизатора области → патчи (src/quantizer.py).

Ключевой инвариант: quantize_region НИКОГДА не падает и ВСЕГДА возвращает
(list, reason). Плюс проверяем крайние случаи из docs/PATCH_AL_REDESIGN.md §5.

Запуск:  python scratchpad/runtests.py tests.test_quantizer
"""

import numpy as np
from PIL import Image

from src import quantizer as qz
from src import config


def _image(w=200, h=200):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), mode="RGB")


def test_empty_region_returns_empty():
    img = _image()
    M = np.zeros((200, 200), dtype=bool)
    patches, reason = qz.quantize_region(M, img, config.CLASS_TALC, S=48)
    assert patches == []
    assert reason == "empty_region"


def test_large_region_yields_multiple_labeled_patches():
    img = _image(200, 200)
    M = np.zeros((200, 200), dtype=bool)
    M[30:170, 30:170] = True   # крупный квадрат, много патчей
    patches, reason = qz.quantize_region(
        M, img, config.CLASS_FINE, S=48, tau=0.65, overlap=0.5, seed=42
    )
    assert reason == "ok"
    assert len(patches) >= 4
    for p in patches:
        assert p.label == config.CLASS_FINE
        assert p.image.size == (48, 48)      # все патчи ровно S×S
        assert p.inside >= 0.65              # порог покрытия соблюдён
        assert not p.upsampled


def test_region_smaller_than_patch_is_upsampled():
    img = _image(200, 200)
    M = np.zeros((200, 200), dtype=bool)
    M[90:110, 90:110] = True   # 20x20 — меньше патча S=48
    patches, reason = qz.quantize_region(M, img, config.CLASS_ORDINARY, S=48, seed=1)
    assert reason in ("ok", "thin_region")
    assert len(patches) >= 1
    assert all(p.image.size == (48, 48) for p in patches)
    assert any(p.upsampled for p in patches)


def test_thin_vein_returns_at_least_one_patch():
    img = _image(200, 200)
    M = np.zeros((200, 200), dtype=bool)
    M[100:104, 20:180] = True   # тонкая горизонтальная жила
    patches, reason = qz.quantize_region(M, img, config.CLASS_TALC, S=48, seed=7)
    assert reason in ("ok", "thin_region")
    assert len(patches) >= 1
    assert all(p.label == config.CLASS_TALC for p in patches)


def test_tiny_region_is_too_small():
    img = _image(200, 200)
    M = np.zeros((200, 200), dtype=bool)
    M[100:102, 100:102] = True   # 2x2 — шум
    patches, reason = qz.quantize_region(M, img, config.CLASS_TALC, S=200, seed=0)
    assert patches == []
    assert reason == "too_small"


def test_determinism_same_seed_same_positions():
    img = _image(200, 200)
    M = np.zeros((200, 200), dtype=bool)
    M[30:170, 30:170] = True
    a, _ = qz.quantize_region(M, img, config.CLASS_FINE, S=48, seed=123)
    b, _ = qz.quantize_region(M, img, config.CLASS_FINE, S=48, seed=123)
    assert [(p.x, p.y) for p in a] == [(p.x, p.y) for p in b]


def test_region_touching_edge_pads_to_S():
    img = _image(200, 200)
    M = np.zeros((200, 200), dtype=bool)
    M[0:60, 0:60] = True   # прижата к углу — патчи вылезают за край, добор паддингом
    patches, reason = qz.quantize_region(M, img, config.CLASS_FINE, S=48, seed=3)
    assert reason in ("ok", "thin_region")
    assert all(p.image.size == (48, 48) for p in patches)


def test_cap_limits_number_of_patches():
    img = _image(400, 400)
    M = np.ones((400, 400), dtype=bool)   # вся картинка — область
    patches, reason = qz.quantize_region(
        M, img, config.CLASS_FINE, S=32, overlap=0.75, N=10, seed=5
    )
    assert reason == "ok"
    assert len(patches) <= 10


def test_never_crashes_on_garbage():
    # Не 2D / странная форма — инвариант «не падаем», возвращаем error-reason.
    img = _image(50, 50)
    patches, reason = qz.quantize_region(np.ones((3, 3, 3)), img, 1, S=16)
    assert patches == []
    assert reason.startswith("error")
