"""
Тесты ГИС-экспорта (src/gis_export.py).

Проверяем структуру GeoJSON: FeatureCollection, полигоны из bbox, атрибуты.
Shapefile тестируем мягко: если pyshp не установлен — просто пропускаем.

Запуск:  pytest -q tests/test_gis_export.py
"""

from PIL import Image

from mock_ml.generator import generate
from src import classification as clf, metrics as metrics_mod
from src.gis_export import build_geojson, export_shapefile
from src.pipeline import load_mask
from src.schemas import AnalysisResult, MLResponse


def _result(tmp_path):
    # MOCK-режима в приложении больше нет — генератор остаётся как ТЕСТ-ФИКСТУРА,
    # чтобы гонять логику (метрики/классификация/экспорт) без torch и весов.
    img = tmp_path / "slide.png"
    Image.new("RGB", (400, 300), (50, 50, 50)).save(img)
    raw = generate(str(img), out_dir=tmp_path / "out",
                   params={"scenario": "refractory"}, size=(400, 300))
    ml = MLResponse.from_json(raw)
    m = metrics_mod.compute_metrics(load_mask(ml.mask_path), ml)
    c = clf.classify(m)
    return AnalysisResult(
        image_name=img.name, image_path=str(img), ml=ml, metrics=m, classification=c,
    )


def test_geojson_structure(tmp_path):
    result = _result(tmp_path)
    gj = build_geojson(result)
    assert gj["type"] == "FeatureCollection"
    assert isinstance(gj["features"], list)
    assert len(gj["features"]) == len(result.ml.objects)
    if gj["features"]:
        f = gj["features"][0]
        assert f["geometry"]["type"] == "Polygon"
        ring = f["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]           # кольцо замкнуто
        assert "class_name" in f["properties"]


def test_geojson_polygon_from_bbox(tmp_path):
    result = _result(tmp_path)
    if not result.ml.objects:
        return
    obj = result.ml.objects[0]
    x, y, w, h = obj.bbox
    gj = build_geojson(result)
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    # первая точка bbox — левый верхний угол
    assert ring[0] == [float(x), float(y)]
    assert [float(x + w), float(y + h)] in ring


def test_shapefile_optional(tmp_path):
    result = _result(tmp_path)
    path = export_shapefile(result)
    # pyshp может отсутствовать — тогда None, и это допустимо
    if path is not None:
        assert path.suffix == ".shp"
        assert path.exists()
