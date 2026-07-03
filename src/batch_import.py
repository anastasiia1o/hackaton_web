"""
BATCH IMPORT — построение очереди импорта изображений ДО запуска обработки.

Три источника наполнения очереди:
  - "manual_path"   — ручной путь к папке (как раньше в batch_process.py)
  - "folder_picker" — выбор папки через проводник (рекурсивный обход)
  - "file_picker"   — выбор нескольких отдельных файлов через проводник

Важно: здесь мы читаем только ЗАГОЛОВОК изображения (PIL Image.open открывает
файл лениво — размеры доступны без декодирования всех пикселей), поэтому
построение очереди для сотен панорам не грузит их в память целиком.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from . import config

SOURCE_MANUAL_PATH = "manual_path"
SOURCE_FOLDER_PICKER = "folder_picker"
SOURCE_FILE_PICKER = "file_picker"

SOURCE_LABELS_RU = {
    SOURCE_MANUAL_PATH: "Ручной путь к папке",
    SOURCE_FOLDER_PICKER: "Выбор папки (проводник)",
    SOURCE_FILE_PICKER: "Выбор файлов (проводник)",
}


@dataclass
class QueueItem:
    """Один элемент очереди импорта — до старта обработки."""
    filename: str
    source: str
    format: str
    file_size_bytes: int
    width: Optional[int]
    height: Optional[int]
    valid: bool
    validation_error: Optional[str] = None
    source_path: Optional[str] = None   # абсолютный путь — для manual_path
    file_bytes: Optional[bytes] = field(default=None, repr=False)  # для picker-источников

    def to_row(self) -> dict:
        """Плоский dict для показа в таблице очереди (без сырых байт)."""
        return {
            "Файл": self.filename,
            "Формат": self.format,
            "Размер файла": _human_size(self.file_size_bytes),
            "Размер изображения": (
                f"{self.width}×{self.height}" if self.width and self.height else "—"
            ),
            "Источник": SOURCE_LABELS_RU.get(self.source, self.source),
            "Статус": "OK" if self.valid else f"Ошибка: {self.validation_error}",
        }


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}" if unit != "Б" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


def _is_supported(suffix: str) -> bool:
    return suffix.lower() in {e.lower() for e in config.SUPPORTED_FORMATS}


def scan_folder_recursive(folder: Path) -> list[Path]:
    """Рекурсивно найти все поддерживаемые изображения в папке (и подпапках)."""
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and _is_supported(p.suffix):
            out.append(p)
    return out


def probe_path(path: Path) -> dict:
    """Считать метаданные файла по пути: формат/размер/разрешение/валидность."""
    fmt = path.suffix.lower()
    try:
        size_bytes = path.stat().st_size
    except OSError as e:
        return {"format": fmt, "file_size_bytes": 0, "width": None, "height": None,
                "valid": False, "validation_error": str(e)}

    if not _is_supported(fmt):
        return {"format": fmt, "file_size_bytes": size_bytes, "width": None, "height": None,
                "valid": False, "validation_error": f"Неподдерживаемый формат: {fmt or '(нет)'}"}
    try:
        with Image.open(path) as im:
            width, height = im.size
        return {"format": fmt, "file_size_bytes": size_bytes, "width": width, "height": height,
                "valid": True, "validation_error": None}
    except Exception as e:  # noqa: BLE001
        return {"format": fmt, "file_size_bytes": size_bytes, "width": None, "height": None,
                "valid": False, "validation_error": f"Не удалось прочитать изображение: {e}"}


def probe_bytes(file_bytes: bytes, filename: str) -> dict:
    """Как probe_path, но для байт в памяти (загрузка через file/folder picker)."""
    fmt = Path(filename).suffix.lower()
    size_bytes = len(file_bytes)
    if not _is_supported(fmt):
        return {"format": fmt, "file_size_bytes": size_bytes, "width": None, "height": None,
                "valid": False, "validation_error": f"Неподдерживаемый формат: {fmt or '(нет)'}"}
    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            width, height = im.size
        return {"format": fmt, "file_size_bytes": size_bytes, "width": width, "height": height,
                "valid": True, "validation_error": None}
    except Exception as e:  # noqa: BLE001
        return {"format": fmt, "file_size_bytes": size_bytes, "width": None, "height": None,
                "valid": False, "validation_error": f"Не удалось прочитать изображение: {e}"}


def queue_item_from_path(path: Path, source: str = SOURCE_MANUAL_PATH) -> QueueItem:
    meta = probe_path(path)
    return QueueItem(
        filename=path.name, source=source, source_path=str(path.resolve()),
        **meta,
    )


def queue_item_from_bytes(
    filename: str, file_bytes: bytes, source: str = SOURCE_FILE_PICKER,
) -> QueueItem:
    meta = probe_bytes(file_bytes, filename)
    return QueueItem(filename=filename, source=source, file_bytes=file_bytes, **meta)


def build_queue_from_folder(folder: Path, source: str = SOURCE_MANUAL_PATH) -> list[QueueItem]:
    """Построить очередь для целой папки (рекурсивно), не читая пиксели."""
    return [queue_item_from_path(p, source=source) for p in scan_folder_recursive(folder)]
