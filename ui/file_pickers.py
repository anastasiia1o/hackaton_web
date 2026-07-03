"""
FILE PICKERS — выбор папки или нескольких файлов через системный проводник.

Streamlit из коробки умеет только "несколько файлов" (`st.file_uploader`
`accept_multiple_files=True`). Выбора ПАПКИ (рекурсивно) в нём нет — поэтому
здесь самодостаточный vanilla-JS компонент (`webkitdirectory`), без npm/CDN,
тот же подход, что и `ui/viewer.py:region_picker`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

_DIR = Path(__file__).resolve().parent / "folder_picker_frontend"
_component = components.declare_component("orevision_folder_picker", path=str(_DIR))


def folder_or_files_picker(
    key: str,
    mode: str = "folder",
    label: Optional[str] = None,
    max_total_mb: int = 800,
) -> Optional[dict]:
    """
    Отрисовать кнопку выбора (папка/файлы). Возвращает последнее значение,
    присланное браузером: {"files": [{"name", "relative_path", "size",
    "data_b64"}], "nonce": ...} или None, если выбор ещё не сделан.
    """
    value = _component(
        mode=mode, label=label, max_total_mb=max_total_mb, key=key, default=None,
    )
    if not isinstance(value, dict) or "files" not in value:
        return None
    return value


def decode_picked_files(value: dict) -> list[tuple[str, bytes]]:
    """Превратить сырое значение компонента в список (имя_файла, байты)."""
    out = []
    for f in value.get("files", []):
        try:
            data = base64.b64decode(f["data_b64"])
        except Exception:  # noqa: BLE001
            continue
        name = f.get("relative_path") or f.get("name")
        out.append((name, data))
    return out
