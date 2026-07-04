"""
test_contract_shape.py — проверка, что ответ сервиса СООТВЕТСТВУЕТ контракту,
БЕЗ установки torch (заглушаем сам инференс модели).

Проверяем ровно шов интеграции: infer.analyze_image() собирает JSON, который
1) проходит валидатор сайта (src/contract.py) без жёстких ошибок,
2) корректно разбирается в MLResponse и проходит весь пайплайн сайта
   (metrics -> classification). Реальная модель отличается только числами в
   softmax — форма ответа от неё не зависит.

Запуск:  python ml_service/test_contract_shape.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_service import infer
from ml_service import model as M


def _install_fake_model():
    """Заменяем torch-инференс детерминированной заглушкой (0..2 softmax)."""
    rng = np.random.default_rng(0)

    def fake_infer_batch(_model, tensors):
        # Разные классы, чтобы получить непустую сетку и objects.
        logits = rng.random((len(tensors), 3))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    M.load_model = lambda *a, **k: object()          # noqa: E731
    M.preprocess = lambda crop, tile: crop           # noqa: E731 (заглушка тензора)
    M.infer_batch = fake_infer_batch
    M.device = lambda: "cpu"                          # noqa: E731


def main() -> int:
    _install_fake_model()

    tmp_img = os.path.join(os.path.dirname(__file__), "outputs", "_selftest.png")
    Image.fromarray(
        (np.random.default_rng(1).random((256, 320, 3)) * 255).astype(np.uint8)
    ).save(tmp_img)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "_selftest")
    result = infer.analyze_image(tmp_img, out_dir, params={"tile": 64, "mode": "grid"})

    # 1) Валидатор контракта сайта — жёстких ошибок быть не должно.
    from src import contract

    errors = contract.validate_ml_response(result)
    hard = [e for e in errors if not e.startswith("[warning]")]
    assert not hard, f"Нарушения контракта:\n  - " + "\n  - ".join(hard)
    print(f"[ok] contract.validate_ml_response: 0 жёстких ошибок "
          f"({len(errors)} мягких предупреждений)")

    # 2) Полный пайплайн сайта поверх ответа сервиса (как в src/pipeline.py).
    from src.schemas import MLResponse
    from src import metrics as metrics_mod
    from src import classification as clf_mod
    from src.pipeline import load_mask

    ml = MLResponse.from_json(result)
    m = metrics_mod.compute_metrics(load_mask(ml.mask_path), ml)
    c = clf_mod.classify(m)
    print(f"[ok] пайплайн: сетка {result['inference_params']['grid']}, "
          f"objects={len(result['objects'])}, "
          f"класс руды = «{c.ore_class}»")

    # 3) Ключевые инварианты формата.
    assert result["mask"].endswith(".png") and os.path.exists(result["mask"])
    assert result["patch_grid"]["labels"].endswith(".png")
    assert set(result["class_legend"].keys()) >= {0, 1, 2, 3, 4}
    print("[ok] файлы маски/сетки на месте, легенда полная (0..4)")
    print("\nВСЁ ЗЕЛЁНОЕ — форма ответа реального сервиса совпадает с контрактом.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
