"""
ML CLIENT — единственная точка, через которую сайт общается с моделью.

Модель ВШИТА в репозиторий (ml_service/). MOCK-режима нет. Два способа считать
(config.ML_MODE), но остальной код о них не знает — он просто зовёт
`analyze(image_path)` и получает MLResponse:

- local : модель грузится и считает В ПРОЦЕССЕ сайта (ml_service.infer) —
          один `streamlit run`, без отдельного сервера. Нужен torch.
- real  : POST multipart/form-data на http://localhost:8001/analyze
          (отдельный сервис ml_service/server.py, напр. на GPU-машине).

Оба пути возвращают ОДИН И ТОТ ЖЕ JSON по API_CONTRACT.md, поэтому переключение
режима не меняет ничего в остальном сайте.
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
    (src/contract.py) — нарушения падают с ПОНЯТНОЙ ContractError.
    """
    mode = mode or config.ML_MODE
    if validate is None:
        validate = config.VALIDATE_ML_RESPONSE

    if mode == "real":
        raw = _analyze_real(image_path, params)
    else:
        raw = _analyze_local(image_path, params)

    if validate:
        contract.assert_valid(raw)

    return MLResponse.from_json(raw)


def _analyze_local(image_path: str, params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Инференс встроенной модели В ПРОЦЕССЕ сайта. Тяжёлый torch импортируется
    внутри (и только здесь), модель кешируется в ml_service.model.load_model.
    """
    from ml_service import infer

    out_dir = config.RESULTS_DIR / Path(image_path).stem
    return infer.analyze_image(image_path, out_dir=str(out_dir), params=params)


def _analyze_real(image_path: str, params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Реальный вызов ML-сервиса по HTTP. Контракт — в API_CONTRACT.md."""
    import requests  # импорт внутри: local-режим не требует requests

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
    Проверить готовность ML. Возвращает (ок?, сообщение) — для индикатора в UI.

    local: проверяем, что установлен torch и на месте файл весов (сама модель
    грузится лениво при первом анализе, поэтому health не тянет 100 МБ в память).
    real:  GET /health у сервиса на :8001.
    """
    mode = mode or config.ML_MODE

    if mode == "real":
        try:
            import requests

            r = requests.get(f"{config.ML_SERVICE_URL}/health", timeout=3)
            if r.ok:
                return True, f"ML-сервис доступен: {config.ML_SERVICE_URL}"
            return False, f"ML ответил кодом {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"ML-сервис недоступен ({config.ML_SERVICE_URL}): {e}"

    # local: встроенная модель
    import importlib.util
    import os

    from ml_service import model as M  # лёгкий импорт: torch тут не грузится

    if importlib.util.find_spec("torch") is None:
        return False, (
            "torch не установлен — встроенная модель не запустится. "
            "Установите зависимости: pip install -r requirements.txt"
        )
    if not os.path.exists(M.DEFAULT_CKPT):
        return False, f"Не найден файл весов модели: {M.DEFAULT_CKPT}"

    loaded = M.load_model.cache_info().currsize > 0
    return True, (
        "Встроенная модель готова (grade_unfreeze_best.pth)"
        + (" — загружена в память." if loaded else ", загрузится при первом анализе.")
    )
