"""
Тесты хранилища датасетов/ROI/разметки (src/dataset_storage.py).

Изолируем от реальной data/ через monkeypatch DATASETS_DIR -> tmp_path,
чтобы тесты не оставляли мусор в рабочем пространстве репозитория.
"""

import numpy as np
import pytest
from PIL import Image

from src import annotation_config as ac
from src import dataset_storage as ds


@pytest.fixture(autouse=True)
def isolated_datasets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATASETS_DIR", tmp_path / "datasets")
    yield


def test_register_image_manual_path_no_copy(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), (10, 20, 30)).save(img_path)

    row = ds.register_image(
        "ds1", filename="sample.png", source="manual_path", source_path=str(img_path),
    )
    assert row["valid"] is True
    assert row["width"] == 100 and row["height"] == 80
    assert row["stored_path"] is None                 # без копии
    assert row["original_path"] == str(img_path)

    resolved = ds.resolve_image_path("ds1", row["image_id"])
    assert resolved == img_path
    assert not (ds.images_dir("ds1") / "sample.png").exists()  # не копировали


def test_register_image_picker_copies_bytes(tmp_path):
    buf_path = tmp_path / "src.png"
    Image.new("RGB", (60, 40), (1, 2, 3)).save(buf_path)
    data = buf_path.read_bytes()

    row = ds.register_image(
        "ds1", filename="picked.png", source="file_picker", file_bytes=data,
    )
    assert row["stored_path"] is not None
    stored = ds.resolve_image_path("ds1", row["image_id"])
    assert stored.exists()
    assert stored.read_bytes() == data                 # копия побитово идентична


def test_list_images_dedupes_by_id(tmp_path):
    img_path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(img_path)
    ds.register_image("ds1", filename="a.png", source="manual_path", source_path=str(img_path))
    ds.register_image("ds1", filename="a.png", source="manual_path", source_path=str(img_path))
    images = ds.list_images("ds1")
    assert len(images) == 1


def test_create_roi_and_list(tmp_path):
    img_path = tmp_path / "pano.png"
    Image.new("RGB", (500, 400), (5, 5, 5)).save(img_path)
    row = ds.register_image("ds1", filename="pano.png", source="manual_path", source_path=str(img_path))
    image_id = row["image_id"]

    roi_crop = Image.new("RGB", (120, 90), (7, 7, 7))
    roi = ds.create_roi(
        "ds1", image_id, x=10, y=20, width=120, height=90,
        source_image_width=500, source_image_height=400, roi_image=roi_crop,
    )
    assert roi["status"] == ac.STATUS_DRAFT
    assert roi["revision"] == 0
    assert (roi["x"], roi["y"], roi["width"], roi["height"]) == (10, 20, 120, 90)

    rois = ds.list_rois("ds1", image_id)
    assert len(rois) == 1
    assert rois[0]["region_id"] == roi["region_id"]

    roi_img = ds.load_roi_image("ds1", image_id, roi["region_id"])
    assert roi_img.size == (120, 90)

    mask, state, shapes = ds.load_annotation("ds1", image_id, roi["region_id"])
    assert mask.shape == (90, 120)
    assert np.all(mask == 0)                            # изначально неразмечено
    assert shapes == {"type": "FeatureCollection", "features": []}


def _setup_roi(tmp_path):
    img_path = tmp_path / "pano.png"
    Image.new("RGB", (300, 200)).save(img_path)
    row = ds.register_image("ds1", filename="pano.png", source="manual_path", source_path=str(img_path))
    image_id = row["image_id"]
    roi = ds.create_roi(
        "ds1", image_id, x=0, y=0, width=50, height=40,
        source_image_width=300, source_image_height=200,
        roi_image=Image.new("RGB", (50, 40)),
    )
    return image_id, roi["region_id"]


def test_save_annotation_flow(tmp_path):
    image_id, region_id = _setup_roi(tmp_path)

    mask1 = np.zeros((40, 50), dtype=np.uint8)
    mask1[0:10, 0:10] = 1  # тальк
    state1 = ds.save_annotation("ds1", image_id, region_id, mask=mask1, status=ac.STATUS_DRAFT)
    assert state1["revision"] == 1
    assert state1["class_pixel_counts"]["1"] == 100

    mask2 = mask1.copy()
    mask2[10:20, 0:10] = 2  # обычные срастания
    state2 = ds.save_annotation("ds1", image_id, region_id, mask=mask2, status=ac.STATUS_ACCEPTED)
    assert state2["revision"] == 2
    assert state2["status"] == ac.STATUS_ACCEPTED

    # ничего не потеряно: и стартовая пустая маска (rev 0), и mask1 (rev 1) архивированы
    rdir = ds.region_dir("ds1", image_id, region_id)
    revisions = sorted((rdir / "revisions").iterdir())
    assert len(revisions) == 2
    blank_rev_mask = np.array(Image.open(revisions[0] / "semantic_mask.png").convert("L"))
    assert np.all(blank_rev_mask == 0)
    prev_mask = np.array(Image.open(revisions[1] / "semantic_mask.png").convert("L"))
    assert np.array_equal(prev_mask, mask1)

    # roi.json синхронизирован
    roi = ds.load_roi("ds1", image_id, region_id)
    assert roi["status"] == ac.STATUS_ACCEPTED
    assert roi["revision"] == 2


def test_save_annotation_rejects_wrong_size(tmp_path):
    image_id, region_id = _setup_roi(tmp_path)
    bad_mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        ds.save_annotation("ds1", image_id, region_id, mask=bad_mask)


def test_get_or_create_whole_image_roi_is_idempotent(tmp_path):
    img_path = tmp_path / "pano.png"
    Image.new("RGB", (200, 150)).save(img_path)
    row = ds.register_image("ds1", filename="pano.png", source="manual_path", source_path=str(img_path))
    image_id = row["image_id"]
    disp = Image.new("RGB", (200, 150), (9, 9, 9))

    roi1 = ds.get_or_create_whole_image_roi("ds1", image_id, disp)
    assert roi1["kind"] == "whole_image"
    assert (roi1["x"], roi1["y"], roi1["width"], roi1["height"]) == (0, 0, 200, 150)

    roi2 = ds.get_or_create_whole_image_roi("ds1", image_id, disp)
    assert roi2["region_id"] == roi1["region_id"]  # переиспользован, не создан заново
    assert len(ds.list_rois("ds1", image_id)) == 1


def test_export_active_learning_filters_by_status(tmp_path):
    image_id, region_id = _setup_roi(tmp_path)
    mask = np.zeros((40, 50), dtype=np.uint8)
    ds.save_annotation("ds1", image_id, region_id, mask=mask, status=ac.STATUS_DRAFT)

    # ещё один ROI, который будет принят
    img_path = tmp_path / "pano.png"
    roi2 = ds.create_roi(
        "ds1", image_id, x=50, y=0, width=30, height=20,
        source_image_width=300, source_image_height=200,
        roi_image=Image.new("RGB", (30, 20)),
    )
    mask2 = np.ones((20, 30), dtype=np.uint8) * 2
    ds.save_annotation("ds1", image_id, roi2["region_id"], mask=mask2, status=ac.STATUS_ACCEPTED)

    result = ds.export_active_learning("ds1")
    assert result["num_samples"] == 1  # только accepted_for_training

    export_dir = tmp_path / "datasets" / "ds1" / "exports" / "active_learning" / result["export_id"]
    assert (export_dir / "classes.json").exists()
    assert (export_dir / "manifest.csv").exists()
    images = list((export_dir / "images").iterdir())
    masks = list((export_dir / "masks").iterdir())
    assert len(images) == 1 and len(masks) == 1
