"""
Тесты обогащённого mock-генератора: регулируемый % талька, шум скана,
неравномерное освещение (см. HANDOFF 2026-07-03 — A).

Запуск:  pytest -q
"""

import numpy as np
from PIL import Image

from src import config
from mock_ml.generator import (
    generate,
    _apply_talc_target,
    _apply_noise,
    _build_confidence_map,
)


def test_apply_talc_target_reaches_approx_fraction():
    rng = np.random.default_rng(1)
    mask = np.zeros((700, 900), dtype=np.uint8)
    objects_raw: list[dict] = []
    achieved = _apply_talc_target(mask, rng, objects_raw, 0.30)

    actual = np.count_nonzero(mask == config.CLASS_TALC) / mask.size
    assert abs(actual - 0.30) < 0.05
    assert abs(achieved - actual) < 1e-9
    # Все добавленные объекты — тальк (сценарные пятна стёрты).
    assert all(o["cls"] == config.CLASS_TALC for o in objects_raw)


def test_apply_talc_target_erases_existing_talc():
    rng = np.random.default_rng(2)
    mask = np.full((200, 200), config.CLASS_TALC, dtype=np.uint8)  # 100% тальк
    objects_raw = [{"cls": config.CLASS_TALC, "bbox": [0, 0, 1, 1], "area_px": 1, "confidence": 0.9}]
    _apply_talc_target(mask, rng, objects_raw, 0.10)

    actual = np.count_nonzero(mask == config.CLASS_TALC) / mask.size
    assert actual < 0.20  # старый 100%-й тальк стёрт и заменён под новую цель


def test_apply_noise_adds_artifacts_proportional_to_level():
    h, w = 700, 900
    rng = np.random.default_rng(3)
    mask_low = np.zeros((h, w), dtype=np.uint8)
    added_low = _apply_noise(mask_low, rng, 0.05)

    rng2 = np.random.default_rng(3)
    mask_high = np.zeros((h, w), dtype=np.uint8)
    added_high = _apply_noise(mask_high, rng2, 0.5)

    assert added_low >= 0
    assert added_high > added_low  # больше уровень шума -> больше вкраплений


def test_apply_noise_zero_level_is_noop():
    mask = np.zeros((100, 100), dtype=np.uint8)
    rng = np.random.default_rng(4)
    added = _apply_noise(mask, rng, 0.0)
    assert added == 0.0
    assert np.count_nonzero(mask) == 0


def test_uneven_illumination_darkens_corners_vs_center():
    h, w = 400, 500
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng(5)
    conf = _build_confidence_map(h, w, mask, rng, illumination="uneven")

    center = int(conf[h // 2, w // 2])
    corner = int(conf[0, 0])
    assert center > corner  # виньетка: центр увереннее краёв


def test_flat_illumination_has_no_vignette():
    h, w = 400, 500
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng(6)
    conf = _build_confidence_map(h, w, mask, rng, illumination="flat")
    # Без виньетки фон однороден (единственное значение, кроме fine/artifact,
    # которых на пустой маске нет).
    assert len(np.unique(conf)) == 1


def test_generate_end_to_end_with_new_params(tmp_path):
    out_dir = tmp_path / "out"
    result = generate(
        "sample.png",
        out_dir=out_dir,
        params={"scenario": "ordinary", "talc_fraction": 0.25, "noise_level": 0.2, "illumination": "uneven"},
        size=(300, 250),
    )

    mask = np.array(Image.open(result["mask"]).convert("L"), dtype=np.uint8)
    talc_actual = np.count_nonzero(mask == config.CLASS_TALC) / mask.size
    assert abs(talc_actual - 0.25) < 0.08

    joined_warnings = " ".join(result["warnings"])
    assert "talc_fraction" in joined_warnings
    assert "шум" in joined_warnings
    assert "освещен" in joined_warnings.lower() or "Неравномерное" in joined_warnings

    # Контракт формы ответа не сломан.
    assert set(result["image_size"]) == {"width", "height"}
    assert result["inference_params"]["talc_fraction"] == 0.25
