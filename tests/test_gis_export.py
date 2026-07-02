"""
Тесты ГИС-экспорта (src/gis_export.py).

Проверяем структуру GeoJSON: FeatureCollection, полигоны из bbox, атрибуты.
Shapefile тестируем мягко: если pyshp не установлен — просто пропускаем.

Запуск:  pytest -q tests/test_gis_export.py
"""

from src.gis_export import build_geojson, export_shapefile
from src.pipeline import run_analysis
from PIL import Image


def _result(tmp_path):
    img = tmp_path / "slide.png"
    Image.new("RGB", (400, 300), (50, 50, 50)).save(img)
    # используем mock-режим напрямую
    return run_analysis(str(img), params={"scenario": "refractory"}, mode="mock")


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
