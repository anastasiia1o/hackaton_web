"""
Валидатор контракта ML → сайт.

Задача: проверить ЛЮБОЙ ответ ML-сервиса (mock или real) на соответствие
`API_CONTRACT.md` / `docs/ML_INTEGRATION_GUIDE.md` и выдать ПОНЯТНЫЕ ошибки
до того, как этот ответ дойдёт до логики. Тогда в день интеграции мы увидим
«отсутствует поле confidence_map», а не загадочный краш где-то в metrics.

Использование:
    from src.contract import validate_ml_response, ContractError, assert_valid

    errors = validate_ml_response(raw_json_dict)   # список строк-ошибок
    if errors:
        ...                                         # показать/залогировать

    assert_valid(raw_json_dict)                     # либо бросит ContractError
"""

from __future__ import annotations

import os
from typing import Any

from . import config

# Обязательные поля верхнего уровня и их типы (для понятных сообщений).
_REQUIRED_TOP: dict[str, type | tuple[type, ...]] = {
    "model_version": str,
    "inference_time_ms": (int, float),
    "image_size": dict,
    "mask": str,
    "class_legend": dict,
    "objects": list,
}
# Поля, которые желательны, но не блокируют работу (только предупреждение).
# patch_grid — сырой квантованный вывод patch-classification модели (contract
# v2). Для обратной совместимости его отсутствие — только предупреждение
# (старый пиксельный ML без patch_grid остаётся валидным).
_RECOMMENDED_TOP = ("confidence_map", "inference_params", "warnings", "patch_grid")

# Максимальное число ячеек сетки, при котором ещё проверяем согласованность
# mask == nearest-upsample(labels) целиком (для гигапиксельных панорам этот
# полный разбор пропускаем, чтобы валидатор не грузил всё в RAM).
_PATCH_CONSISTENCY_CELL_CAP = 20000

# Допустимые коды классов (см. API_CONTRACT.md).
_VALID_CLASSES = {
    config.CLASS_BACKGROUND,
    config.CLASS_ORDINARY,
    config.CLASS_FINE,
    config.CLASS_TALC,
    config.CLASS_ARTIFACT,
}


class ContractError(ValueError):
    """Ответ ML не соответствует контракту. Текст содержит все нарушения."""


def validate_ml_response(
    raw: Any,
    check_mask_file: bool = True,
) -> list[str]:
    """
    Проверить сырой JSON-ответ ML. Возвращает СПИСОК ошибок (пустой = всё ок).
    Ничего не бросает — удобно, чтобы собрать все проблемы разом и показать.

    check_mask_file=False отключает чтение файла маски с диска (полезно, когда
    маска ещё не на общей ФС, напр. при юнит-проверке чистого JSON).
    """
    errors: list[str] = []

    # 0. Ответ вообще словарь?
    if not isinstance(raw, dict):
        return [f"Ответ ML должен быть JSON-объектом, а получен {type(raw).__name__}."]

    # 1. Обязательные поля и их типы.
    for field, expected_type in _REQUIRED_TOP.items():
        if field not in raw:
            errors.append(f"Отсутствует обязательное поле '{field}'.")
            continue
        if not isinstance(raw[field], expected_type):
            tname = _type_name(expected_type)
            errors.append(
                f"Поле '{field}' должно быть типа {tname}, "
                f"а получено {type(raw[field]).__name__}."
            )

    # Рекомендованные поля — только мягкое предупреждение (не ошибка).
    for field in _RECOMMENDED_TOP:
        if field not in raw:
            errors.append(f"[warning] Желательное поле '{field}' отсутствует.")

    # Если базовых полей нет — дальше проверять смысла мало.
    if any(not s.startswith("[warning]") for s in errors) and (
        "image_size" not in raw or "class_legend" not in raw
    ):
        return errors

    # 2. image_size: width/height — положительные целые.
    size = raw.get("image_size", {})
    w = _as_int(size.get("width"))
    h = _as_int(size.get("height"))
    if w is None or h is None or w <= 0 or h <= 0:
        errors.append(
            "Поле 'image_size' должно содержать положительные целые "
            "'width' и 'height'."
        )

    # 3. class_legend: должны присутствовать коды 0..4.
    legend_keys = {_as_int(k) for k in raw.get("class_legend", {}).keys()}
    missing_codes = sorted(c for c in _VALID_CLASSES if c not in legend_keys)
    if missing_codes:
        errors.append(
            f"В 'class_legend' не хватает кодов классов: {missing_codes} "
            f"(ожидаются 0..4)."
        )

    # 4. objects: структура каждого элемента.
    objects = raw.get("objects", [])
    if isinstance(objects, list):
        for i, o in enumerate(objects):
            errors.extend(_validate_object(i, o))

    # 5. Файл маски: существует, читается, значения пикселей 0..4, размер совпадает.
    if check_mask_file and isinstance(raw.get("mask"), str):
        errors.extend(_validate_mask_file(raw["mask"], w, h))

    # 6. patch_grid (contract v2): согласованность блочной маски с сеткой патчей.
    if "patch_grid" in raw and raw["patch_grid"] is not None:
        errors.extend(
            _validate_patch_grid(raw["patch_grid"], raw.get("mask"), w, h, check_mask_file)
        )

    return errors


def assert_valid(raw: Any, check_mask_file: bool = True) -> None:
    """Как validate_ml_response, но бросает ContractError при жёстких ошибках."""
    errors = validate_ml_response(raw, check_mask_file=check_mask_file)
    hard = [e for e in errors if not e.startswith("[warning]")]
    if hard:
        bullet = "\n  - ".join(hard)
        raise ContractError(
            "Ответ ML не соответствует контракту (см. docs/ML_INTEGRATION_GUIDE.md):"
            f"\n  - {bullet}"
        )


# --------------------------------------------------------------------------- #
# Вспомогательные проверки
# --------------------------------------------------------------------------- #

def _validate_object(i: int, o: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(o, dict):
        return [f"objects[{i}] должен быть объектом, а получен {type(o).__name__}."]
    for f in ("id", "class", "bbox", "area_px", "confidence"):
        if f not in o:
            errs.append(f"objects[{i}]: отсутствует поле '{f}'.")
    if "class" in o and _as_int(o["class"]) not in (
        config.CLASS_ORDINARY, config.CLASS_FINE, config.CLASS_TALC, config.CLASS_ARTIFACT
    ):
        errs.append(
            f"objects[{i}]: 'class'={o['class']} вне диапазона 1..4."
        )
    bbox = o.get("bbox")
    if bbox is not None and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4):
        errs.append(f"objects[{i}]: 'bbox' должен быть списком из 4 чисел [x,y,w,h].")
    conf = o.get("confidence")
    if conf is not None and not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
        errs.append(f"objects[{i}]: 'confidence'={conf} должно быть в диапазоне [0,1].")
    return errs


def _validate_patch_grid(
    pg: Any,
    mask_path: Any,
    w: int | None,
    h: int | None,
    check_files: bool,
) -> list[str]:
    """
    Проверить блок `patch_grid` (contract v2): типы, размер сетки, коды классов,
    и — если сетка небольшая — согласованность `mask` == nearest-upsample(labels).
    """
    errs: list[str] = []
    if not isinstance(pg, dict):
        return [f"'patch_grid' должен быть объектом, а получен {type(pg).__name__}."]

    for f in ("tile", "rows", "cols", "labels"):
        if f not in pg:
            errs.append(f"patch_grid: отсутствует поле '{f}'.")
    rows = _as_int(pg.get("rows"))
    cols = _as_int(pg.get("cols"))
    tile = _as_int(pg.get("tile"))
    if rows is None or cols is None or rows <= 0 or cols <= 0:
        errs.append("patch_grid: 'rows' и 'cols' должны быть положительными целыми.")
    if tile is None or tile <= 0:
        errs.append("patch_grid: 'tile' должен быть положительным целым.")

    origin = pg.get("origin", [0, 0])
    if not (isinstance(origin, (list, tuple)) and len(origin) == 2):
        errs.append("patch_grid: 'origin' должен быть [x, y].")

    labels_path = pg.get("labels")
    if not check_files or not isinstance(labels_path, str):
        return errs
    if not os.path.exists(labels_path):
        errs.append(f"patch_grid: файл 'labels' не найден: {labels_path}")
        return errs
    try:
        import numpy as np
        from PIL import Image

        with Image.open(labels_path) as im:
            labels = np.array(im.convert("L"))
    except Exception as e:  # noqa: BLE001
        return errs + [f"patch_grid: не удалось прочитать 'labels' ({labels_path}): {e}"]

    lh, lw = labels.shape[:2]
    if rows is not None and cols is not None and (lh != rows or lw != cols):
        errs.append(
            f"patch_grid: размер labels ({lw}x{lh}) не совпадает с rows×cols "
            f"({cols}x{rows})."
        )
    import numpy as np

    bad = sorted(v for v in set(np.unique(labels).tolist()) if v not in _VALID_CLASSES)
    if bad:
        errs.append(
            f"patch_grid.labels содержит недопустимые коды классов {bad[:8]} "
            f"(ожидаются только 0..4)."
        )

    # Согласованность mask == nearest-upsample(labels) — только для небольших сеток.
    if (
        isinstance(mask_path, str)
        and os.path.exists(mask_path)
        and rows is not None and cols is not None
        and rows * cols <= _PATCH_CONSISTENCY_CELL_CAP
        and w and h
    ):
        try:
            up = np.array(
                Image.fromarray(labels.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
            )
            with Image.open(mask_path) as im:
                mask = np.array(im.convert("L"))
            if mask.shape == up.shape and not np.array_equal(mask, up):
                mismatch = float(np.mean(mask != up))
                errs.append(
                    "patch_grid: 'mask' не совпадает с nearest-апскейлом 'labels' "
                    f"(расхождение {mismatch * 100:.1f}% пикселей)."
                )
        except Exception:  # noqa: BLE001 — сверка необязательна, не валим на ней
            pass
    return errs


def _validate_mask_file(path: str, w: int | None, h: int | None) -> list[str]:
    errs: list[str] = []
    if not os.path.exists(path):
        return [f"Файл маски не найден по пути 'mask': {path}"]
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            mask = np.array(im.convert("L"))
    except Exception as e:  # noqa: BLE001
        return [f"Не удалось прочитать файл маски '{path}': {e}"]

    # Размер совпадает с image_size?
    mh, mw = mask.shape[:2]
    if w is not None and h is not None and (mw != w or mh != h):
        errs.append(
            f"Размер маски ({mw}x{mh}) не совпадает с image_size ({w}x{h})."
        )
    # Значения пикселей строго 0..4?
    import numpy as np

    unique = set(np.unique(mask).tolist())
    bad = sorted(v for v in unique if v not in _VALID_CLASSES)
    if bad:
        errs.append(
            f"Маска содержит недопустимые значения пикселей {bad[:8]} "
            f"(ожидаются только 0..4). Проверьте, что PNG не сглажен."
        )
    return errs


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _type_name(t: type | tuple[type, ...]) -> str:
    if isinstance(t, tuple):
        return " или ".join(x.__name__ for x in t)
    return t.__name__
