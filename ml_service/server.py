"""
server.py — HTTP-сервис РЕАЛЬНОЙ ML-модели (сторона `orevision-ml` контракта).

Реализует ровно два эндпоинта из API_CONTRACT.md, которые уже ждёт сайт
(src/ml_client.py в режиме OREVISION_ML_MODE=real):

    GET  /health   -> 200 OK, индикатор доступности в UI
    POST /analyze  -> multipart/form-data (image[, params]) -> JSON по контракту

Запуск (из корня репозитория):
    pip install -r ml_service/requirements.txt
    python ml_service/server.py                 # слушает :8001

Затем сайт переключается ОДНОЙ настройкой:
    OREVISION_ML_MODE=real streamlit run app.py

torch/модель грузятся ЛЕНИВО при первом /analyze — сервис и /health поднимаются
мгновенно даже до прогрева модели (и даже без установленного torch: тогда
/analyze честно вернёт 503 с понятным текстом, а не упадёт молча).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Позволяем запуск и как `python ml_service/server.py`, и как `-m ml_service.server`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml_service import infer  # запуск из корня репо
except ImportError:  # запуск из папки ml_service
    import infer  # type: ignore

from flask import Flask, jsonify, request

app = Flask(__name__)

# Куда сервис пишет маски/сетки (сайт читает их по абсолютному пути с той же ФС).
OUTPUT_DIR = Path(
    os.getenv("ORE_ML_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "outputs"))
)
HOST = os.getenv("ORE_ML_HOST", "0.0.0.0")
PORT = int(os.getenv("ORE_ML_PORT", "8001"))


@app.get("/health")
def health():
    """Лёгкая проверка доступности: модель здесь НЕ грузим (см. модуль-док)."""
    ckpt = infer.M.DEFAULT_CKPT
    return jsonify(
        {
            "status": "ok",
            "service": "orevision-ml",
            "model_version": infer.MODEL_VERSION,
            "checkpoint_present": os.path.exists(ckpt),
            "model_loaded": infer.M.load_model.cache_info().currsize > 0,
        }
    )


@app.post("/analyze")
def analyze():
    """POST /analyze — принять изображение, вернуть JSON по API_CONTRACT.md."""
    if "image" not in request.files:
        return jsonify({"error": "нет файла 'image' в multipart-запросе"}), 400

    params = {}
    raw_params = request.form.get("params")
    if raw_params:
        import json

        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"params не валидный JSON: {e}"}), 400

    f = request.files["image"]
    suffix = Path(f.filename or "upload.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.save(tmp.name)
        tmp.close()
        out_dir = OUTPUT_DIR / Path(f.filename or "upload").stem
        result = infer.analyze_image(tmp.name, str(out_dir), params)
        return jsonify(result)
    except (ImportError, ModuleNotFoundError) as e:
        # torch / segmentation_models_pytorch не установлены.
        return (
            jsonify(
                {
                    "error": "ML-зависимости не установлены на сервисе",
                    "detail": str(e),
                    "hint": "pip install -r ml_service/requirements.txt",
                }
            ),
            503,
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:  # noqa: BLE001 — вернём понятную 500 вместо стектрейса в консоль
        traceback.print_exc()
        return jsonify({"error": "сбой инференса", "detail": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    print(f"orevision-ml -> http://{HOST}:{PORT}  (checkpoint: {infer.M.DEFAULT_CKPT})")
    app.run(host=HOST, port=PORT, threaded=False)
