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


def test_write_imagefolder_layout_and_manifest(tmp_path):
    import csv
    recs = [
        {"image": Image.new("RGB", (48, 48), (10, 20, 30)), "label_name": "talc",
         "stem": "img_r_3_000", "label": 3, "weight": 1.0, "source_image": "img",
         "region": "r", "x": 5, "y": 7, "inside": 0.9, "upsampled": 0, "source_size": 48},
        {"image": Image.new("RGB", (48, 48), (0, 0, 0)), "label_name": "fine_intergrowth",
         "stem": "img_r_2_000", "label": 2, "weight": 0.5, "source_image": "img",
         "region": "r", "x": 1, "y": 1, "inside": 0.7, "upsampled": 1, "source_size": 20},
    ]
    classes_json = {"2": {"name": "fine_intergrowth"}, "3": {"name": "talc"}}
    res = de.write_imagefolder(tmp_path, recs, classes_json)

    assert res["num_patches"] == 2
    assert (tmp_path / "imgs" / "talc" / "img_r_3_000.jpg").exists()
    assert (tmp_path / "imgs" / "fine_intergrowth" / "img_r_2_000.jpg").exists()
    assert (tmp_path / "classes.json").exists()

    with open(tmp_path / "manifest.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["path"] for r in rows} == {
        "imgs/talc/img_r_3_000.jpg", "imgs/fine_intergrowth/img_r_2_000.jpg",
    }
    assert set(rows[0].keys()) == set(de.PATCH_MANIFEST_FIELDS)


def test_quantizer_to_imagefolder_pipeline(tmp_path):
    """Область эксперта → патчи (quantizer) → ImageFolder (write_imagefolder)."""
    from src import quantizer as qz
    from src import config

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8), "RGB")
    M = np.zeros((200, 200), dtype=bool)
    M[30:170, 30:170] = True
    patches, reason = qz.quantize_region(M, img, config.CLASS_TALC, S=48, seed=1)
    assert reason == "ok" and patches

    recs = [{
        "image": p.image, "label_name": "talc", "stem": f"s_{k:03d}",
        "label": p.label, "weight": p.weight, "source_image": "img", "region": "r",
        "x": p.x, "y": p.y, "inside": p.inside, "upsampled": int(p.upsampled),
        "source_size": p.source_size,
    } for k, p in enumerate(patches)]
    res = de.write_imagefolder(tmp_path, recs, {"3": {"name": "talc"}})
    assert res["num_patches"] == len(patches)
    assert len(list((tmp_path / "imgs" / "talc").glob("*.jpg"))) == len(patches)


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
