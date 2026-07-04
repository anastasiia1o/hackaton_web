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
обучена вторая голова-детектор фона `bg_head` — Linear(2048→1). Из общего
2048-эмбеддинга `sigmoid(bg_head(emb))` даёт вероятность, что кадр — нерудная
матрица -> код контракта 0. ВАЖНО: голова надёжна только на СНИМКЕ ЦЕЛИКОМ (в том
режиме и обучалась, F1=1.0). На 512-px тайлах панорамы она вне распределения и
массово ошибается, поэтому `bg_image_probability` считает её как ЦЕЛОКАДРОВЫЙ
вентиль, а infer.py применяет его лишь к снимкам размера одного FOV (панорамы
идут прежним grade-путём без изменений).

`grade_unfreeze_best.pth` побайтово тот же, что и раньше (macro-F1=0.944);
добавился только `bg_head_best.pth` (F1=1.000, 2049 параметров). Т.к. активное
обучение дообучает лишь голову сорта при ЗАМОРОЖЕННОМ энкодере, эмбеддинги не
меняются и `bg_head` остаётся совместимым с дообученными чекпоинтами.

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


def bg_image_probability(model, bg_head, img_np) -> float:
    """Вероятность, что ВЕСЬ снимок — фон (нерудная матрица), 0..1.

    Голова-детектор фона надёжна только в том режиме, в котором обучалась и
    валидировалась (F1=1.0): один снимок целиком, препроцессинг val-трансформа
    Resize(256)→CenterCrop(224). На отдельных снимках руды по сортам (~2272×1704 —
    одно train-FOV) это даёт ~0.0 для руды и ~0.95+ для фоновых кадров. Поэтому
    infer.py применяет её как ЦЕЛОКАДРОВЫЙ вентиль на снимках размера одного FOV,
    а НЕ по тайлам панорамы (там 512-px тайлы вне распределения → ложный фон).
    """
    from PIL import Image  # noqa: PLC0415

    torch = _torch()
    pil = Image.fromarray(img_np)
    w, h = pil.size
    s = 256.0 / max(1, min(w, h))
    pil = pil.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = pil.size
    left, top = (w - 224) // 2, (h - 224) // 2
    pil = pil.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(pil, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device())
    with torch.no_grad():
        return float(torch.sigmoid(bg_head(model.get_embedding(x)).squeeze()))
