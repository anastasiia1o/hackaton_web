"""
Экспорт найденных объектов в ГИС-форматы: GeoJSON и Shapefile.

Зачем: доп. пожелание ТЗ — интеграция с геологическими информационными
системами (ГИС). Каждый найденный сульфид/тальк-объект превращаем в полигон
с атрибутами (класс, площадь, уверенность).

ВАЖНО про координаты. Здесь координаты — В ПИКСЕЛЯХ исходного изображения
(начало отсчёта — левый верхний угол, ось Y вниз). Это НЕ географические
широта/долгота. Для настоящей геопривязки нужен масштаб снимка (мкм/пиксель)
и world-file/CRS — их можно передать позже через метаданные. Пока экспортируем
в «пиксельной» системе координат: этого достаточно, чтобы открыть объекты в
QGIS/ArcGIS как слой и сверить с изображением.

Форматы:
- GeoJSON — обычный текстовый JSON (RFC 7946-подобный), пишется на чистом
  Python, без зависимостей. Основной и всегда доступный формат.
- Shapefile — бинарный формат ESRI (.shp/.shx/.dbf). Требует библиотеку
  `pyshp` (pip install pyshp). Если её нет — функция честно скажет об этом.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import config, storage
from .schemas import AnalysisResult, MLObject


def _bbox_ring(bbox: list[int]) -> list[list[float]]:
    """Прямоугольник [x,y,w,h] → замкнутое кольцо полигона (первая=последняя точка)."""
    x, y, w, h = bbox
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
        [float(x), float(y)],  # замыкаем кольцо
    ]


def _feature(obj: MLObject) -> dict[str, Any]:
    """Один объект → GeoJSON Feature (полигон + атрибуты)."""
    return {
        "type": "Feature",
        "properties": {
            "id": obj.id,
            "class": obj.cls,
            "class_name": config.CLASS_NAMES.get(obj.cls, str(obj.cls)),
            "area_px": obj.area_px,
            "confidence": round(obj.confidence, 3),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [_bbox_ring(obj.bbox)],
        },
    }


def build_geojson(result: AnalysisResult) -> dict[str, Any]:
    """Собрать GeoJSON FeatureCollection из объектов результата."""
    return {
        "type": "FeatureCollection",
        "name": Path(result.image_path).stem,
        # Помечаем систему координат как пиксельную (не географическую).
        "properties": {
            "coordinate_system": "image_pixels",
            "note": "Координаты в пикселях изображения; ось Y направлена вниз.",
            "image_size": result.ml.image_size,
            "model_version": result.ml.model_version,
        },
        "features": [_feature(o) for o in result.ml.objects],
    }


def export_geojson(result: AnalysisResult) -> Path:
    """Записать objects.geojson в папку результатов. Возвращает путь."""
    d = storage.result_dir(result.image_path)
    path = d / "objects.geojson"
    path.write_text(
        json.dumps(build_geojson(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def geojson_bytes(result: AnalysisResult) -> bytes:
    """GeoJSON в память — для кнопки скачивания в Streamlit."""
    return json.dumps(
        build_geojson(result), ensure_ascii=False, indent=2
    ).encode("utf-8")


def export_shapefile(result: AnalysisResult) -> Optional[Path]:
    """
    Записать Shapefile (набор .shp/.shx/.dbf) в папку результатов.
    Требует библиотеку pyshp. Если её нет — вернёт None и не упадёт
    (GeoJSON при этом остаётся основным форматом).
    Возвращает путь к .shp либо None.
    """
    try:
        import shapefile  # pyshp
    except ImportError:
        return None

    d = storage.result_dir(result.image_path)
    base = d / "objects"  # pyshp сам добавит расширения .shp/.shx/.dbf

    writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON)
    # Поля атрибутов (.dbf). 'C'=строка, 'N'=целое, 'F'=дробное.
    writer.field("id", "N")
    writer.field("class", "N")
    writer.field("cls_name", "C", size=40)
    writer.field("area_px", "N")
    writer.field("conf", "F", decimal=3)

    for o in result.ml.objects:
        writer.poly([_bbox_ring(o.bbox)])
        writer.record(
            o.id, o.cls, config.CLASS_NAMES.get(o.cls, str(o.cls)),
            o.area_px, round(o.confidence, 3),
        )
    writer.close()
    return base.with_suffix(".shp")


def export_gis(result: AnalysisResult) -> dict[str, Optional[Path]]:
    """Экспортировать оба формата сразу. Shapefile может быть None (нет pyshp)."""
    return {
        "geojson": export_geojson(result),
        "shapefile": export_shapefile(result),
    }
