"""
ANNOTATION EDITOR — Python-обёртка над кастомным canvas-редактором разметки
(ui/annotation_editor_frontend). Кисть/полигон/ластик/undo-redo/присвоение
класса всему участку живут в JS (мгновенная реакция без rerun'ов Streamlit);
Python лишь передаёт участок+маску+классы и забирает готовую PNG-маску и
геометрию многоугольников для сохранения через src/dataset_storage.py.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

from src.annotation_config import AnnotationClass

_DIR = Path(__file__).resolve().parent / "annotation_editor_frontend"
_component = components.declare_component("orevision_annotation_editor", path=str(_DIR))


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_data_uri(mask: np.ndarray) -> str:
    """Кодируем маску как R=G=B=classId, чтобы JS мог точно восстановить id по R-каналу."""
    img = Image.fromarray(mask.astype(np.uint8), mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def annotation_canvas(
    image: Image.Image,
    mask: Optional[np.ndarray],
    classes: list[AnnotationClass],
    region_key: str,
    shapes_geojson: Optional[dict] = None,
    show_image: bool = True,
    show_mask: bool = True,
    mask_opacity: float = 0.7,
    key: Optional[str] = None,
) -> Optional[dict]:
    """
    Отрисовать редактор и вернуть последнее известное состояние:
    {"mask_png_b64": str, "shapes": geojson dict, "revision": int, "width", "height"}
    или None, если компонент ещё не успел проинициализироваться.
    """
    w, h = image.size
    value = _component(
        image_src=_image_to_data_uri(image),
        mask_src=_mask_to_data_uri(mask) if mask is not None else None,
        width=w, height=h,
        classes=[{"id": c.id, "name_ru": c.name_ru, "color": list(c.color)} for c in classes],
        region_key=region_key,
        shapes=shapes_geojson,
        show_image=show_image,
        show_mask=show_mask,
        mask_opacity=mask_opacity,
        key=key or region_key,
        default=None,
    )
    return value if isinstance(value, dict) else None


def decode_mask_from_value(value: dict) -> Optional[np.ndarray]:
    """Достать маску (2D uint8, значение пикселя = id класса) из значения компонента."""
    b64 = value.get("mask_png_b64") if value else None
    if not b64:
        return None
    data = base64.b64decode(b64)
    arr = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    return arr[:, :, 0]  # R=G=B=classId — читаем R-канал как точный id
