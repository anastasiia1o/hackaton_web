"""
model.py — обёртка над РЕАЛЬНОЙ моделью классификации сортов руды.

Модель (grade_unfreeze_best.pth) — из ../ore_classification (команда ML,
Задача 3 «Скажи мне, кто твой шлиф»). Backbone se_resnext50_32x4d (MicroNet),
голова Linear(2048→256→3). Классифицирует квадратный тайл в один из 3 сортов:

    вывод модели   класс руды          код контракта (src/config.py)
    0  talc        Оталькованная   ->  3  CLASS_TALC
    1  ordinary    Рядовая         ->  1  CLASS_ORDINARY
    2  fine        Труднообогатимая->  2  CLASS_FINE

Класс «Фон» (пакет ../ore_4class_package): поверх ТОГО ЖЕ замороженного энкодера
обучена вторая голова-детектор фона `bg_head` — Linear(2048→1). На один проход
энкодера из общего 2048-эмбеддинга считаются оба выхода (`infer_batch_cascade`):

    sigmoid(bg_head(emb)) > порог   -> тайл = ФОН -> код контракта 0 (фон/матрица)
    иначе                            -> argmax(grade_head) -> сорт руды (1/2/3)

Детекция фона идёт ПО ТАЙЛАМ на любых снимках, включая панорамы. ⚠️ Вшитый
`bg_head_best.pth` обучался на снимках-целиком и на 512-px тайлах панорамы
переоценивает фон — это осознанно: голова предназначена для ручного дообучения
под тайлы (ORE_ML_BG_CKPT). Полностью отключить фон: пустой ORE_ML_BG_CKPT.

`grade_unfreeze_best.pth` побайтово тот же, что и раньше (macro-F1=0.944);
добавился только `bg_head_best.pth` (2049 параметров). Т.к. активное обучение
дообучает лишь голову сорта при ЗАМОРОЖЕННОМ энкодере, эмбеддинги не меняются и
`bg_head` остаётся совместимым с дообученными чекпоинтами.

Здесь только загрузка весов и батч-инференс тайлов. Тайлинг панорамы и сборка
JSON по контракту — в infer.py. torch импортируется лениво, чтобы сам сервис
и его /health поднимались даже без установленного torch (тогда /analyze вернёт
понятную 503-ошибку, а не упадёт на импорте).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# ImageNet-нормировка (та же, что при обучении модели — см. panorama_infer.py).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Порядок выходов модели: индекс softmax -> имя сорта (см. ОПИСАНИЕ_КОДА.txt §1).
MODEL_CLASS_NAMES = ["talc", "ordinary", "fine"]

# Перевод индекса выхода модели -> код класса КОНТРАКТА (src/config.py, 0..4).
# 0 talc -> 3 тальк ; 1 ordinary -> 1 обычные ; 2 fine -> 2 тонкие.
MODEL_TO_CONTRACT = np.array([3, 1, 2], dtype=np.uint8)

# Код контракта для кадров, распознанных как ФОН (нерудная матрица).
CONTRACT_BACKGROUND = 0

# Путь к весам: рядом с этим файлом (по умолчанию), либо через переменную окружения.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.getenv(
    "ORE_ML_CKPT", os.path.join(_HERE, "grade_unfreeze_best.pth")
)
# Голова-детектор фона (Linear(2048→1)) поверх того же энкодера. Пустая строка
# в ORE_ML_BG_CKPT полностью отключает детекцию фона (режим «3 класса»).
DEFAULT_BG_CKPT = os.getenv(
    "ORE_ML_BG_CKPT", os.path.join(_HERE, "bg_head_best.pth")
)
# Порог sigmoid: выше -> кадр считается фоном (обучение шло с threshold=0.5).
BG_THRESHOLD = float(os.getenv("ORE_ML_BG_THRESHOLD", "0.5"))


def _torch():
    """Ленивая загрузка torch — сервис стартует и без него (см. /health)."""
    import torch  # noqa: PLC0415

    return torch


def device() -> str:
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_module():
    """GradeClassifier: encoder se_resnext50_32x4d + AvgPool + Linear(2048→256→3).

    Архитектура должна В ТОЧНОСТИ совпадать с обучающей (иначе load_state_dict
    не сойдётся по ключам). Скопировано из ../ore_classification/panorama_infer.py.
    """
    torch = _torch()
    import torch.nn as nn  # noqa: PLC0415
    import segmentation_models_pytorch as smp  # noqa: PLC0415

    class GradeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            dummy = smp.Unet(
                encoder_name="se_resnext50_32x4d",
                encoder_weights=None,
                in_channels=3,
                classes=2,
            )
            self.encoder = dummy.encoder
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)
            )

        def get_embedding(self, x):
            """2048-мерный эмбеддинг (общий вход для головы сорта и bg_head)."""
            return self.pool(self.encoder(x)[-1]).view(x.size(0), 2048)

        def forward(self, x):
            return self.head(self.get_embedding(x))

    return GradeClassifier()


@lru_cache(maxsize=1)
def load_model(ckpt_path: str = DEFAULT_CKPT):
    """Загрузить веса один раз и закешировать (модель тяжёлая — грузим лениво)."""
    torch = _torch()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Не найден чекпоинт модели: {ckpt_path}. "
            f"Положите grade_unfreeze_best.pth в ml_service/ или задайте ORE_ML_CKPT."
        )
    dev = device()
    model = _build_module().to(dev)
    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@lru_cache(maxsize=2)
def load_bg_head(ckpt_path: str = DEFAULT_BG_CKPT):
    """Голова-детектор фона Linear(2048→1) поверх общего энкодера.

    Возвращает None, если детекция фона отключена (пустой путь) или чекпоинт не
    найден — тогда infer.py работает в режиме «3 класса» без фона (мягкая
    деградация, а не падение). Кешируется по пути, как и grade-модель.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    torch = _torch()
    import torch.nn as nn  # noqa: PLC0415

    dev = device()
    head = nn.Linear(2048, 1).to(dev)
    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    head.load_state_dict(state)
    return head.eval()


def preprocess(crop_np: np.ndarray, tile_size: int):
    """RGB-кроп (H,W,3 uint8) -> нормированный тензор (3,tile,tile) float32."""
    from PIL import Image  # noqa: PLC0415

    h, w = crop_np.shape[:2]
    if h != tile_size or w != tile_size:
        crop_np = np.array(
            Image.fromarray(crop_np).resize((tile_size, tile_size), Image.BILINEAR)
        )
    arr = (crop_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return _torch().from_numpy(arr.transpose(2, 0, 1))


def infer_batch(model, tensors) -> np.ndarray:
    """Список тензоров (3,H,W) -> softmax-вероятности (N,3) как numpy."""
    torch = _torch()
    import torch.nn.functional as F  # noqa: PLC0415

    batch = torch.stack(tensors).to(device())
    with torch.no_grad():
        return F.softmax(model(batch), dim=1).cpu().numpy()


def infer_batch_cascade(model, bg_head, tensors):
    """Каскад за ОДИН проход энкодера: -> (grade_probs (N,3), bg_prob (N,)).

    Из общего 2048-эмбеддинга считаем softmax головы сорта и sigmoid головы фона.
    Если bg_head=None — bg_prob нули (режим «3 класса»). Так энкодер (самая
    тяжёлая часть) прогоняется один раз, а не дважды.
    """
    torch = _torch()
    import torch.nn.functional as F  # noqa: PLC0415

    batch = torch.stack(tensors).to(device())
    with torch.no_grad():
        emb = model.get_embedding(batch)                      # (N,2048)
        grade = F.softmax(model.head(emb), dim=1).cpu().numpy()
        if bg_head is not None:
            bg = torch.sigmoid(bg_head(emb).squeeze(1)).cpu().numpy()
        else:
            bg = np.zeros(len(tensors), dtype=np.float32)
    return grade, bg
