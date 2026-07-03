"""Тесты построения очереди импорта (src/batch_import.py)."""

from PIL import Image

from src import batch_import as bi


def test_scan_folder_recursive_finds_nested_and_ignores_unsupported(tmp_path):
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "a.png")
    Image.new("RGB", (10, 10)).save(tmp_path / "sub" / "b.jpg")
    (tmp_path / "notes.txt").write_text("x")

    found = bi.scan_folder_recursive(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.png", "b.jpg"]


def test_probe_path_valid_image(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (33, 22)).save(p)
    meta = bi.probe_path(p)
    assert meta["valid"] is True
    assert (meta["width"], meta["height"]) == (33, 22)
    assert meta["format"] == ".png"


def test_probe_path_unsupported_format(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello")
    meta = bi.probe_path(p)
    assert meta["valid"] is False
    assert "формат" in meta["validation_error"].lower()


def test_probe_path_corrupt_image(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"not a real png")
    meta = bi.probe_path(p)
    assert meta["valid"] is False


def test_probe_bytes_matches_probe_path(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (50, 60)).save(p, format="JPEG")
    data = p.read_bytes()
    meta = bi.probe_bytes(data, "img.jpg")
    assert meta["valid"] is True
    assert (meta["width"], meta["height"]) == (50, 60)


def test_build_queue_from_folder(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "one.png")
    Image.new("RGB", (10, 10)).save(tmp_path / "two.tiff")
    items = bi.build_queue_from_folder(tmp_path, source=bi.SOURCE_FOLDER_PICKER)
    assert len(items) == 2
    assert all(i.source == bi.SOURCE_FOLDER_PICKER for i in items)
    row = items[0].to_row()
    assert "Файл" in row and "Статус" in row
