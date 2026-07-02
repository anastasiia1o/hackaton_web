"""
Централизованная конфигурация OreVision.

Здесь собраны ВСЕ "магические числа" и настройки в одном месте, чтобы:
- геолог мог поменять порог талька, не лазая по коду;
- оба агента (A и B) ссылались на одни и те же значения и цвета.

Значения можно переопределить переменными окружения (удобно для Docker),
но по умолчанию всё работает локально без какой-либо настройки.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Корневые пути проекта --------------------------------------------------
# BASE_DIR указывает на корень репозитория (папка orevision-app).
BASE_DIR = Path(__file__).resolve().parent.parent

# Папка с локальными данными. Всё хранится ЛОКАЛЬНО, ничего не уходит в интернет.
DATA_DIR = Path(os.getenv("OREVISION_DATA_DIR", BASE_DIR / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"     # исходные изображения шлифов
RESULTS_DIR = DATA_DIR / "results"     # маски, confidence, отчёты, JSON
SAMPLES_DIR = DATA_DIR / "samples"     # демонстрационные образцы

# --- Версии (для воспроизводимости и логов) --------------------------------
APP_VERSION = "0.2.0"          # версия сайта OreVision
CONTRACT_VERSION = "v1"        # версия API-контракта с ML

# --- Настройки ML-сервиса ---------------------------------------------------
# По умолчанию работаем в MOCK-режиме, чтобы сайт можно было запустить
# и показать ДО того, как ML-команда поднимет свой сервис.
ML_MODE = os.getenv("OREVISION_ML_MODE", "mock")          # "mock" | "real"
ML_SERVICE_URL = os.getenv("OREVISION_ML_URL", "http://localhost:8001")
ML_ANALYZE_ENDPOINT = f"{ML_SERVICE_URL}/analyze"
ML_TIMEOUT_SEC = int(os.getenv("OREVISION_ML_TIMEOUT", "300"))  # до 5 минут на панораму
# Проверять каждый ответ ML валидатором контракта (src/contract.py).
VALIDATE_ML_RESPONSE = os.getenv("OREVISION_VALIDATE_ML", "1") != "0"

# --- Геологические пороги классификации ------------------------------------
# ЕДИНЫЙ источник правды для правил классификации (см. src/classification.py).
TALC_THRESHOLD = 0.10          # >10% талька -> "Оталькованная руда"
# Зона неопределённости вокруг порога талька (пограничный случай).
TALC_BORDERLINE_LOW = 0.09     # 9%
TALC_BORDERLINE_HIGH = 0.11    # 11%
# Порог низкой уверенности модели -> "Требуется экспертная проверка".
LOW_CONFIDENCE_THRESHOLD = 0.55
# Доля артефактов, выше которой изображение считается "грязным".
ARTIFACT_WARN_FRACTION = 0.30  # >30% площади — артефакты
# Зона "ничья" вокруг 50/50 обычные/тонкие -> нет явного преобладания -> проверка.
TIE_MARGIN = 0.02              # |доля тонких среди сульфидов − 0.5| < 0.02 → ничья
# Минимальная доля сульфидов, ниже которой тип срастаний определять нельзя.
MIN_SULPHIDE_FRACTION = 0.005  # 0.5% валидной площади

# --- Коды классов (ДОЛЖНЫ совпадать с API_CONTRACT.md и ML-командой) --------
CLASS_BACKGROUND = 0   # фон / нерудная матрица
CLASS_ORDINARY = 1     # обычные (рядовые) срастания
CLASS_FINE = 2         # тонкие (труднообогатимые) срастания
CLASS_TALC = 3         # тальк
CLASS_ARTIFACT = 4     # артефакт / исключённая область

CLASS_NAMES = {
    CLASS_BACKGROUND: "Фон / нерудная матрица",
    CLASS_ORDINARY: "Обычные срастания",
    CLASS_FINE: "Тонкие срастания",
    CLASS_TALC: "Тальк",
    CLASS_ARTIFACT: "Артефакт / исключено",
}

# --- Цвета overlay-маски (RGBA), едины для UI и легенды ---------------------
# Зелёный = обычные, красный = тонкие, синий = тальк, серый = артефакт.
CLASS_COLORS = {
    CLASS_BACKGROUND: (0, 0, 0, 0),          # прозрачный
    CLASS_ORDINARY: (0, 200, 0, 130),        # зелёный
    CLASS_FINE: (220, 30, 30, 130),          # красный
    CLASS_TALC: (30, 90, 230, 130),          # синий
    CLASS_ARTIFACT: (128, 128, 128, 110),    # серый
}

# --- Ограничения на изображения --------------------------------------------
MAX_DIMENSION_WARN = 10000     # предупреждать про панорамы больше 10000 px
SUPPORTED_FORMATS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def ensure_dirs() -> None:
    """Создать все локальные папки данных, если их ещё нет."""
    for d in (DATA_DIR, UPLOADS_DIR, RESULTS_DIR, SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
