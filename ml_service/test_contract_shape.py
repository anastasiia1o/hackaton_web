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


def _install_fake_model(bg_prob=0.0):
    """Заменяем torch-инференс детерминированной заглушкой (grade softmax + фон).

    bg_prob — целокадровая вероятность фона, которую вернёт вентиль
    (model.bg_image_probability). 0.0 -> обычный grade-разбор; >порога -> весь
    кадр помечается фоном (код 0).
    """
    rng = np.random.default_rng(0)

    def _softmax(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def fake_infer_batch(_model, tensors):
        # Разные классы, чтобы получить непустую сетку и objects.
        return _softmax(rng.random((len(tensors), 3)))

    M.load_model = lambda *a, **k: object()          # noqa: E731
    M.load_bg_head = lambda *a, **k: object()        # noqa: E731 (вентиль фона активен)
    M.bg_image_probability = lambda *a, **k: bg_prob  # noqa: E731
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

    # 2b) Вентиль фона: детектор включён, кадр размера FOV -> вентиль применён.
    #     bg_prob=0.0 -> фона нет, идёт обычный grade-разбор.
    ip = result["inference_params"]
    assert ip["bg_detector"] is True, "bg-детектор должен быть активен"
    assert ip["bg_gate_applied"] is True, "вентиль фона должен примениться на FOV-кадре"
    mask_arr = np.array(Image.open(result["mask"]).convert("L"))
    assert not (mask_arr == 0).any(), "при bg_prob=0 кадр не должен быть фоном"
    print(f"[ok] вентиль фона: применён, bg_prob={ip['bg_prob']}, кадр = руда")

    # 2c) Фоновый кадр: bg_prob выше порога -> ВЕСЬ кадр = код 0.
    _install_fake_model(bg_prob=0.99)
    bg_res = infer.analyze_image(tmp_img, out_dir, params={"tile": 64, "mode": "grid"})
    bg_mask = np.array(Image.open(bg_res["mask"]).convert("L"))
    assert (bg_mask == 0).all(), "фоновый кадр должен быть целиком кодом 0"
    assert not bg_res["objects"], "у фонового кадра нет рудных объектов"
    from src import contract as _c
    assert not [e for e in _c.validate_ml_response(bg_res) if not e.startswith("[warning]")]
    print(f"[ok] фоновый кадр: весь кадр = фон (bg_prob={bg_res['inference_params']['bg_prob']})")
    _install_fake_model(bg_prob=0.0)  # вернуть обычный режим

    # 3) Ключевые инварианты формата.
    assert result["mask"].endswith(".png") and os.path.exists(result["mask"])
    assert result["patch_grid"]["labels"].endswith(".png")
    assert set(result["class_legend"].keys()) >= {0, 1, 2, 3, 4}
    print("[ok] файлы маски/сетки на месте, легенда полная (0..4)")
    print("\nВСЁ ЗЕЛЁНОЕ — форма ответа реального сервиса совпадает с контрактом.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
