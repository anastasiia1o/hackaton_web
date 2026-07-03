"""Тесты экспорта в формате «как S2_v2» (src/dataset_export.py)."""

import zipfile

import numpy as np
from PIL import Image

from src import dataset_export as de

CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (30, 90, 230, 190),
    2: (0, 200, 0, 190),
}
CLASS_NAMES = {0: "фон", 1: "тальк", 2: "обычные срастания"}


def _sample_mask(w=40, h=30):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    mask[10:20, 10:20] = 2
    return mask


def test_majority_class_in_polygon_picks_dominant_class():
    mask = _sample_mask(40, 30)  # 10x10 блок класса 1 в углу (0..10,0..10)
    # полигон, полностью накрывающий блок класса 1
    points = [(0.0, 0.0), (0.25, 0.0), (0.25, 1 / 3), (0.0, 1 / 3)]
    cls = de.majority_class_in_polygon(mask, points, width=40, height=30)
    assert cls == 1


def test_majority_class_in_polygon_background():
    mask = _sample_mask(40, 30)
    points = [(0.7, 0.7), (0.95, 0.7), (0.95, 0.95), (0.7, 0.95)]
    cls = de.majority_class_in_polygon(mask, points, width=40, height=30)
    assert cls == 0


def test_majority_class_in_polygon_too_few_points_returns_none():
    mask = _sample_mask(40, 30)
    assert de.majority_class_in_polygon(mask, [(0.1, 0.1), (0.2, 0.2)], 40, 30) is None
    assert de.majority_class_in_polygon(mask, [], 40, 30) is None


def test_mask_to_id_image_round_trip():
    mask = _sample_mask()
    img = de.mask_to_id_image(mask)
    arr = np.array(img)
    assert arr.shape == (30, 40, 3)
    assert np.array_equal(arr[:, :, 0], mask)
    assert np.array_equal(arr[:, :, 0], arr[:, :, 1])
    assert np.array_equal(arr[:, :, 1], arr[:, :, 2])


def test_colorize_opaque_uses_class_colors():
    mask = _sample_mask()
    img = de.colorize_opaque(mask, CLASS_COLORS)
    arr = np.array(img)
    assert tuple(arr[5, 5]) == (30, 90, 230)   # класс 1
    assert tuple(arr[15, 15]) == (0, 200, 0)   # класс 2
    assert tuple(arr[25, 35]) == (0, 0, 0)     # фон


def test_build_human_triptych_has_three_panels_and_legend():
    mask = _sample_mask()
    image = Image.new("RGB", (40, 30), (80, 80, 80))
    triptych = de.build_human_triptych(image, mask, CLASS_COLORS, CLASS_NAMES, panel_height=60)
    assert triptych.height > 60  # заголовок + панель + легенда
    assert triptych.width > 60 * 3 * 0.5  # три панели рядом


def test_export_s2_bundle_writes_expected_files(tmp_path):
    mask = _sample_mask()
    image = Image.new("RGB", (40, 30), (10, 10, 10))
    items = [{"name": "sample_01", "image": image, "mask": mask}]

    result = de.export_s2_bundle(tmp_path, items, CLASS_COLORS, CLASS_NAMES)
    assert result["num_items"] == 1
    assert (tmp_path / "imgs" / "sample_01.jpg").exists()
    assert (tmp_path / "masks" / "sample_01.png").exists()
    assert (tmp_path / "masks_colored" / "sample_01.png").exists()
    assert (tmp_path / "masks_human" / "sample_01.jpg").exists()


def test_export_s2_bundle_accepts_generator(tmp_path):
    def gen():
        for i in range(3):
            yield {"name": f"g{i}", "image": Image.new("RGB", (20, 20), (i, i, i)), "mask": _sample_mask(20, 20)}

    result = de.export_s2_bundle(tmp_path, gen(), CLASS_COLORS, CLASS_NAMES, include_human=False)
    assert result["num_items"] == 3
    assert len(list((tmp_path / "imgs").iterdir())) == 3


def test_export_s2_bundle_with_split_subfolder(tmp_path):
    mask = _sample_mask()
    image = Image.new("RGB", (40, 30), (10, 10, 10))
    items = [{"name": "a", "image": image, "mask": mask}]

    de.export_s2_bundle(tmp_path, items, CLASS_COLORS, CLASS_NAMES, split="train", include_human=False)
    assert (tmp_path / "imgs" / "train" / "a.jpg").exists()
    assert (tmp_path / "masks" / "train" / "a.png").exists()
    assert not (tmp_path / "masks_human").exists()


def test_zip_directory_produces_valid_zip(tmp_path):
    mask = _sample_mask()
    image = Image.new("RGB", (40, 30), (10, 10, 10))
    items = [{"name": "a", "image": image, "mask": mask}]
    de.export_s2_bundle(tmp_path / "bundle", items, CLASS_COLORS, CLASS_NAMES)

    data = de.zip_directory(tmp_path / "bundle")
    zpath = tmp_path / "out.zip"
    zpath.write_bytes(data)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert any(n.endswith("imgs/a.jpg") for n in names)
    assert any(n.endswith("masks/a.png") for n in names)
