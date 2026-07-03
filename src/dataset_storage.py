"""
DATASET STORAGE — хранение панорам, ROI и разметки для active learning.

Раскладка на диске (см. docs/coordination/HANDOFF.md и README для описания):

  data/datasets/<dataset_id>/
    images/<image_id>.<ext>                       — управляемая копия исходника
    manifest.jsonl                                  — реестр зарегистрированных изображений
    annotations/<image_id>/
      image_meta.json
      regions/<region_id>/
        roi.json                — координаты ROI относительно исходной панорамы
        roi_image.png            — вырезанный участок (копия пикселей, lossless)
        semantic_mask.png        — 8-bit одноканальная маска классов (для обучения)
        annotation_state.json    — статус/ревизия/статистика по классам
        shapes.geojson           — многоугольники (если использовались)
        revisions/<rev>/         — снимки предыдущих версий маски+состояния
    exports/active_learning/<export_id>/
      images/ masks/ manifest.csv manifest.jsonl classes.json

Принципы:
  - исходные изображения НИКОГДА не модифицируются на месте;
  - для источников "path" разрешена ссылка без копии (см. register_image);
    для "folder_picker"/"file_picker" копия ОБЯЗАТЕЛЬНА (иного пути к файлу нет);
  - каждое явное сохранение разметки создаёт ревизию предыдущей версии —
    принятая (accepted_for_training) разметка никогда не исчезает молча.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from . import annotation_config as ac
from . import config

DATASETS_DIR = config.DATA_DIR / "datasets"

COORD_SYSTEM_DESC = (
    "Пиксели исходной панорамы (source_image_id); начало координат (0,0) — "
    "левый верхний угол, ось X вправо, ось Y вниз. x,y,width,height задают "
    "прямоугольник ROI в этой системе координат."
)


# --------------------------------------------------------------------------- #
# Пути
# --------------------------------------------------------------------------- #

def dataset_dir(dataset_id: str) -> Path:
    d = DATASETS_DIR / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def images_dir(dataset_id: str) -> Path:
    d = dataset_dir(dataset_id) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "manifest.jsonl"


def annotations_dir(dataset_id: str) -> Path:
    d = dataset_dir(dataset_id) / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_annotations_dir(dataset_id: str, image_id: str) -> Path:
    d = annotations_dir(dataset_id) / image_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def regions_dir(dataset_id: str, image_id: str) -> Path:
    d = image_annotations_dir(dataset_id, image_id) / "regions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def region_dir(dataset_id: str, image_id: str, region_id: str) -> Path:
    return regions_dir(dataset_id, image_id) / region_id


def exports_root(dataset_id: str) -> Path:
    d = dataset_dir(dataset_id) / "exports" / "active_learning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_dataset_ids() -> list[str]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(p.name for p in DATASETS_DIR.iterdir() if p.is_dir())


def _slugify(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name]
    return "".join(keep).strip("_") or "item"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# Регистрация изображений (manifest.jsonl)
# --------------------------------------------------------------------------- #

def _make_image_id(filename: str, disambiguator: str) -> str:
    stem = _slugify(Path(filename).stem)[:40]
    h = uuid.uuid5(uuid.NAMESPACE_URL, disambiguator).hex[:8]
    return f"{stem}_{h}"


def register_image(
    dataset_id: str,
    *,
    filename: str,
    source: str,
    file_bytes: Optional[bytes] = None,
    source_path: Optional[str] = None,
    probe: Optional[dict] = None,
) -> dict:
    """
    Зарегистрировать изображение в датасете (запись в manifest.jsonl).

    source="manual_path"  -> без копии, храним ссылку на source_path (как есть
                              в текущей архитектуре, где batch читает файлы
                              напрямую с диска — воспроизводимо, без лишнего I/O).
    source in ("folder_picker","file_picker") -> ОБЯЗАТЕЛЬНО копируем байты в
                              images/, т.к. у браузерного File нет пути на диске.
    """
    from . import batch_import

    if source == batch_import.SOURCE_MANUAL_PATH:
        if not source_path:
            raise ValueError("register_image(manual_path) требует source_path")
        p = probe or batch_import.probe_path(Path(source_path))
        image_id = _make_image_id(filename, source_path)
        stored_path = None
    else:
        if file_bytes is None:
            raise ValueError("register_image(picker) требует file_bytes")
        p = probe or batch_import.probe_bytes(file_bytes, filename)
        image_id = _make_image_id(filename, f"{filename}:{len(file_bytes)}:{uuid.uuid4().hex[:6]}")
        ext = Path(filename).suffix.lower() or ".bin"
        dest = images_dir(dataset_id) / f"{image_id}{ext}"
        dest.write_bytes(file_bytes)
        stored_path = str(dest)

    row = {
        "image_id": image_id,
        "dataset_id": dataset_id,
        "original_filename": filename,
        "source": source,
        "original_path": source_path,
        "stored_path": stored_path,
        "format": p["format"],
        "file_size_bytes": p["file_size_bytes"],
        "width": p["width"],
        "height": p["height"],
        "valid": p["valid"],
        "validation_error": p["validation_error"],
        "added_at": _now(),
    }
    with open(manifest_path(dataset_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if p["valid"]:
        save_image_meta(dataset_id, image_id, {
            "image_id": image_id,
            "original_filename": filename,
            "width": p["width"],
            "height": p["height"],
            "format": p["format"],
        })
    return row


def list_images(dataset_id: str) -> list[dict]:
    """Прочитать реестр изображений (при повторной регистрации — последняя запись)."""
    path = manifest_path(dataset_id)
    if not path.exists():
        return []
    by_id: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_id[row["image_id"]] = row
    return sorted(by_id.values(), key=lambda r: r.get("added_at", ""))


def get_image(dataset_id: str, image_id: str) -> Optional[dict]:
    for row in list_images(dataset_id):
        if row["image_id"] == image_id:
            return row
    return None


def resolve_image_path(dataset_id: str, image_id: str) -> Optional[Path]:
    """Путь для реального чтения пикселей: копия в датасете либо исходная ссылка."""
    row = get_image(dataset_id, image_id)
    if row is None:
        return None
    if row.get("stored_path"):
        return Path(row["stored_path"])
    if row.get("original_path"):
        return Path(row["original_path"])
    return None


def save_image_meta(dataset_id: str, image_id: str, meta: dict) -> Path:
    p = image_annotations_dir(dataset_id, image_id) / "image_meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_image_meta(dataset_id: str, image_id: str) -> Optional[dict]:
    p = image_annotations_dir(dataset_id, image_id) / "image_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# ROI (участки панорамы)
# --------------------------------------------------------------------------- #

def _next_region_id(dataset_id: str, image_id: str) -> str:
    existing = list(regions_dir(dataset_id, image_id).iterdir()) if regions_dir(dataset_id, image_id).exists() else []
    n = len(existing) + 1
    return f"roi_{n:04d}_{uuid.uuid4().hex[:6]}"


def create_roi(
    dataset_id: str,
    image_id: str,
    *,
    x: int, y: int, width: int, height: int,
    source_image_width: int, source_image_height: int,
    roi_image: Image.Image,
    extra: Optional[dict] = None,
) -> dict:
    """
    Создать новый сохранённый ROI: координаты относительно оригинала + копия
    пикселей участка (roi_image.png) + пустая (неразмеченная) стартовая маска.

    extra — необязательные дополнительные поля в roi.json (например,
    {"kind": "whole_image"} для страницы «Разметка эксперта», где ROI покрывает
    весь показанный кадр целиком, а не вырезанный пользователем прямоугольник).
    """
    region_id = _next_region_id(dataset_id, image_id)
    rdir = region_dir(dataset_id, image_id, region_id)
    rdir.mkdir(parents=True, exist_ok=True)

    roi_image.convert("RGB").save(rdir / "roi_image.png")
    blank_mask = np.zeros((roi_image.height, roi_image.width), dtype=np.uint8)
    Image.fromarray(blank_mask, mode="L").save(rdir / "semantic_mask.png")
    (rdir / "shapes.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    now = _now()
    roi = {
        "region_id": region_id,
        "dataset_id": dataset_id,
        "source_image_id": image_id,
        "source_image_width": int(source_image_width),
        "source_image_height": int(source_image_height),
        "x": int(x), "y": int(y), "width": int(width), "height": int(height),
        "coordinate_system": COORD_SYSTEM_DESC,
        "created_at": now,
        "updated_at": now,
        "status": ac.STATUS_DRAFT,
        "revision": 0,
        **(extra or {}),
    }
    (rdir / "roi.json").write_text(json.dumps(roi, ensure_ascii=False, indent=2), encoding="utf-8")

    state = {
        "region_id": region_id, "image_id": image_id, "dataset_id": dataset_id,
        "status": ac.STATUS_DRAFT, "revision": 0, "updated_at": now, "author": "",
        "class_pixel_counts": _class_pixel_counts(blank_mask),
    }
    (rdir / "annotation_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return roi


def get_or_create_whole_image_roi(
    dataset_id: str, image_id: str, display_image: Image.Image,
) -> dict:
    """
    Вернуть (создав при необходимости) единственный ROI, покрывающий весь
    показанный кадр целиком — используется страницей «Разметка эксперта»,
    где эксперт размечает изображение напрямую, без отдельного шага
    вырезания прямоугольного участка. Кадр — уже уменьшенное для показа
    изображение (как и весь остальной UI, гигапиксельные панорамы целиком в
    память не грузим); source_image_width/height здесь равны размеру ИМЕННО
    этого кадра, т.к. это и есть пиксельная сетка, которую эксперт размечает.
    """
    for r in list_rois(dataset_id, image_id):
        if r.get("kind") == "whole_image":
            return r
    return create_roi(
        dataset_id, image_id,
        x=0, y=0, width=display_image.width, height=display_image.height,
        source_image_width=display_image.width, source_image_height=display_image.height,
        roi_image=display_image, extra={"kind": "whole_image"},
    )


def list_rois(dataset_id: str, image_id: str) -> list[dict]:
    """Все сохранённые ROI для панорамы, с актуальным статусом/ревизией."""
    rd = regions_dir(dataset_id, image_id)
    if not rd.exists():
        return []
    out = []
    for sub in sorted(rd.iterdir()):
        roi_path = sub / "roi.json"
        if not roi_path.exists():
            continue
        try:
            roi = json.loads(roi_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        state_path = sub / "annotation_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                roi["status"] = state.get("status", roi.get("status"))
                roi["revision"] = state.get("revision", roi.get("revision"))
                roi["updated_at"] = state.get("updated_at", roi.get("updated_at"))
            except Exception:  # noqa: BLE001
                pass
        out.append(roi)
    return sorted(out, key=lambda r: r.get("created_at", ""))


def load_roi(dataset_id: str, image_id: str, region_id: str) -> Optional[dict]:
    p = region_dir(dataset_id, image_id, region_id) / "roi.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_roi_image(dataset_id: str, image_id: str, region_id: str) -> Optional[Image.Image]:
    p = region_dir(dataset_id, image_id, region_id) / "roi_image.png"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


# --------------------------------------------------------------------------- #
# Разметка (semantic_mask.png + shapes.geojson + annotation_state.json)
# --------------------------------------------------------------------------- #

def _class_pixel_counts(mask: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(mask, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def load_annotation(
    dataset_id: str, image_id: str, region_id: str,
) -> tuple[Optional[np.ndarray], Optional[dict], Optional[dict]]:
    """Загрузить (mask, annotation_state, shapes) для повторного редактирования."""
    rdir = region_dir(dataset_id, image_id, region_id)
    mask_path = rdir / "semantic_mask.png"
    state_path = rdir / "annotation_state.json"
    shapes_path = rdir / "shapes.geojson"

    mask = None
    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    state = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            state = None
    shapes = None
    if shapes_path.exists():
        try:
            shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            shapes = None
    return mask, state, shapes


def save_annotation(
    dataset_id: str,
    image_id: str,
    region_id: str,
    *,
    mask: np.ndarray,
    shapes: Optional[dict] = None,
    status: str = ac.STATUS_DRAFT,
    author: str = "",
) -> dict:
    """
    Явно сохранить разметку региона. Каждое сохранение:
      1) архивирует ПРЕДЫДУЩУЮ версию маски+состояния в revisions/ (если была);
      2) пишет новую semantic_mask.png (8-bit, один канал, точный размер ROI);
      3) пишет shapes.geojson (если переданы многоугольники);
      4) обновляет annotation_state.json (статус, номер ревизии, счётчики пикселей);
      5) синхронизирует статус/ревизию в roi.json.
    """
    rdir = region_dir(dataset_id, image_id, region_id)
    if not rdir.exists():
        raise FileNotFoundError(f"ROI не найден: {dataset_id}/{image_id}/{region_id}")

    roi_image_path = rdir / "roi_image.png"
    if roi_image_path.exists():
        with Image.open(roi_image_path) as roi_im:
            expected_size = roi_im.size
        if (mask.shape[1], mask.shape[0]) != expected_size:
            raise ValueError(
                f"Размер маски {mask.shape[1]}x{mask.shape[0]} не совпадает с "
                f"roi_image.png {expected_size[0]}x{expected_size[1]}"
            )

    mask_path = rdir / "semantic_mask.png"
    state_path = rdir / "annotation_state.json"
    shapes_path = rdir / "shapes.geojson"

    prev_revision = 0
    if state_path.exists():
        try:
            prev_state = json.loads(state_path.read_text(encoding="utf-8"))
            prev_revision = int(prev_state.get("revision", 0))
        except Exception:  # noqa: BLE001
            prev_revision = 0
        if mask_path.exists():
            rev_dir = rdir / "revisions" / f"rev_{prev_revision:04d}_{time.strftime('%Y%m%dT%H%M%S')}"
            rev_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_path, rev_dir / "semantic_mask.png")
            if state_path.exists():
                shutil.copy2(state_path, rev_dir / "annotation_state.json")
            if shapes_path.exists():
                shutil.copy2(shapes_path, rev_dir / "shapes.geojson")

    new_revision = prev_revision + 1
    Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
    if shapes is not None:
        shapes_path.write_text(json.dumps(shapes, ensure_ascii=False, indent=2), encoding="utf-8")

    now = _now()
    state = {
        "region_id": region_id, "image_id": image_id, "dataset_id": dataset_id,
        "status": status, "revision": new_revision, "updated_at": now, "author": author,
        "class_pixel_counts": _class_pixel_counts(mask),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    roi = load_roi(dataset_id, image_id, region_id)
    if roi is not None:
        roi["status"] = status
        roi["revision"] = new_revision
        roi["updated_at"] = now
        (rdir / "roi.json").write_text(json.dumps(roi, ensure_ascii=False, indent=2), encoding="utf-8")

    return state


# --------------------------------------------------------------------------- #
# Экспорт для дообучения (active learning)
# --------------------------------------------------------------------------- #

def export_active_learning(
    dataset_id: str,
    *,
    statuses: tuple[str, ...] = ac.EXPORTABLE_STATUSES,
    model_version: Optional[str] = None,
    export_id: Optional[str] = None,
) -> dict:
    """
    Собрать пары images/masks для всех регионов датасета со статусом из
    `statuses` (по умолчанию — только accepted_for_training) + manifest.csv,
    manifest.jsonl, classes.json.
    """
    export_id = export_id or f"export_{time.strftime('%Y%m%dT%H%M%S')}"
    edir = exports_root(dataset_id) / export_id
    (edir / "images").mkdir(parents=True, exist_ok=True)
    (edir / "masks").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ann_root = annotations_dir(dataset_id)
    if ann_root.exists():
        for image_dir in sorted(ann_root.iterdir()):
            if not image_dir.is_dir():
                continue
            image_id = image_dir.name
            img_row = get_image(dataset_id, image_id) or {}
            for roi in list_rois(dataset_id, image_id):
                region_id = roi["region_id"]
                status = roi.get("status", ac.STATUS_DRAFT)
                if status not in statuses:
                    continue
                rdir = region_dir(dataset_id, image_id, region_id)
                roi_img_path = rdir / "roi_image.png"
                mask_path = rdir / "semantic_mask.png"
                if not (roi_img_path.exists() and mask_path.exists()):
                    continue

                sample_id = f"{image_id}__{region_id}"
                shutil.copy2(roi_img_path, edir / "images" / f"{sample_id}.png")
                shutil.copy2(mask_path, edir / "masks" / f"{sample_id}.png")

                state = json.loads((rdir / "annotation_state.json").read_text(encoding="utf-8")) \
                    if (rdir / "annotation_state.json").exists() else {}

                rows.append({
                    "sample_id": sample_id,
                    "dataset_id": dataset_id,
                    "source_image_id": image_id,
                    "source_image_path": img_row.get("stored_path") or img_row.get("original_path"),
                    "roi_id": region_id,
                    "roi_coordinates": json.dumps({
                        "x": roi["x"], "y": roi["y"],
                        "width": roi["width"], "height": roi["height"],
                    }, ensure_ascii=False),
                    "image_path": f"images/{sample_id}.png",
                    "mask_path": f"masks/{sample_id}.png",
                    "annotation_status": status,
                    "class_pixel_counts": json.dumps(state.get("class_pixel_counts", {}), ensure_ascii=False),
                    "original_dimensions": json.dumps({
                        "width": roi.get("source_image_width"),
                        "height": roi.get("source_image_height"),
                    }, ensure_ascii=False),
                    "model_version": model_version or "",
                    "annotation_revision": state.get("revision", roi.get("revision", 0)),
                })

    # manifest.csv
    csv_path = edir / "manifest.csv"
    fieldnames = [
        "sample_id", "dataset_id", "source_image_id", "source_image_path", "roi_id",
        "roi_coordinates", "image_path", "mask_path", "annotation_status",
        "class_pixel_counts", "original_dimensions", "model_version", "annotation_revision",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # manifest.jsonl
    jsonl_path = edir / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # classes.json
    (edir / "classes.json").write_text(
        json.dumps(ac.classes_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"export_id": export_id, "dir": str(edir), "num_samples": len(rows)}


def list_exports(dataset_id: str) -> list[str]:
    root = exports_root(dataset_id)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
