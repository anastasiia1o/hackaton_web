"""
ML CLIENT — единственная точка, через которую сайт общается с ML.

Ключевая идея: остальной код НЕ знает, mock сейчас или real. Он просто зовёт
`analyze(image_path)` и получает MLResponse. Переключение делается ОДНОЙ
настройкой config.ML_MODE ("mock" | "real") или переменной окружения
OREVISION_ML_MODE.

- mock  : результат генерирует mock_ml.generator локально;
- real  : POST multipart/form-data на http://localhost:8001/analyze,
          ответ разбирается тем же MLResponse.from_json.

Так ML-команда может подключиться позже, ничего не ломая в сайте.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import config
from . import contract
from .schemas import MLResponse


def analyze(
    image_path: str,
    params: Optional[dict[str, Any]] = None,
    mode: Optional[str] = None,
    validate: Optional[bool] = None,
) -> MLResponse:
    """
    Проанализировать одно изображение. Возвращает MLResponse (см. schemas.py).
    mode переопределяет config.ML_MODE (удобно для тестов и batch).

    Перед разбором ответ прогоняется через валидатор контракта
    (src/contract.py). Если ML прислал что-то не по контракту — упадём с
    ПОНЯТНОЙ ошибкой ContractError, а не где-то в глубине metrics.
    """
    mode = mode or config.ML_MODE
    if validate is None:
        validate = config.VALIDATE_ML_RESPONSE

    if mode == "real":
        raw = _analyze_real(image_path, params)
    else:
        raw = _analyze_mock(image_path, params)

    if validate:
        # Жёсткие нарушения -> ContractError; мягкие ([warning]) не роняют.
        contract.assert_valid(raw)

    return MLResponse.from_json(raw)


def _analyze_mock(image_path: str, params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Локальная генерация — импортируем внутри, чтобы mock был опциональным."""
    from mock_ml.generator import generate

    out_dir = config.RESULTS_DIR / Path(image_path).stem
    return generate(image_path, out_dir=out_dir, params=params)


def _analyze_real(image_path: str, params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Реальный вызов ML-сервиса.
    Контракт запроса/ответа — в API_CONTRACT.md.
    """
    import requests  # импорт внутри: сайт запустится даже без requests в mock-режиме

    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f)}
        data = {}
        if params:
            import json
            data["params"] = json.dumps(params)
        resp = requests.post(
            config.ML_ANALYZE_ENDPOINT,
            files=files,
            data=data,
            timeout=config.ML_TIMEOUT_SEC,
        )
    resp.raise_for_status()
    return resp.json()


def health_check(mode: Optional[str] = None) -> tuple[bool, str]:
    """
    Проверить доступность ML. Для mock всегда True.
    Возвращает (ок?, сообщение) — удобно показать в UI индикатором.
    """
    mode = mode or config.ML_MODE
    if mode != "real":
        return True, "MOCK-режим: ML имитируется локально."
    try:
        import requests

        r = requests.get(f"{config.ML_SERVICE_URL}/health", timeout=3)
        if r.ok:
            return True, f"ML-сервис доступен: {config.ML_SERVICE_URL}"
        return False, f"ML ответил кодом {r.status_code}"
    except Exception as e:  # noqa: BLE001 — показываем геологу простое сообщение
        return False, f"ML-сервис недоступен ({config.ML_SERVICE_URL}): {e}"
