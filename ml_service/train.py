#!/usr/bin/env python3
"""
train.py — ДООБУЧЕНИЕ вшитой модели сортов руды на патчах active learning.

Замыкает цикл активного обучения ПОЛНОСТЬЮ внутри этого репозитория (раньше
обучение жило только в ../ore_classification). Ест ImageFolder-экспорт патчей
из `src/dataset_storage.export_active_learning_patch` (см. docs/PATCH_AL_REDESIGN.md
§6) и выдаёт новый чекпоинт, который без правок грузит `ml_service/model.py`.

    /analyze (patch_grid + conf)
       → active_query: worklist по неопределённости
       → эксперт размечает связную область + класс
       → export active_learning_patch  (imgs/<class>/*.jpg + manifest.csv + weights)
       → **train.py: base + AL-патчи (взвешенный сэмплинг)**   ← этот файл
       → новый .pth → ORE_ML_CKPT → redeploy /analyze

Архитектура сети берётся из `ml_service/model.py` (тот же GradeClassifier), поэтому
сохранённые здесь веса грузятся обратно один-в-один. torch импортируется лениво:
модуль можно импортировать и разбирать манифест без установленного torch-стека
(инференс и тесты формата этим пользуются).

Пример:

    # дообучить вшитую модель на одном экспорте патчей, сохранить новый чекпоинт
    python -m ml_service.train \\
        --patch-export data/datasets/<id>/exports/active_learning_patch/<export_id> \\
        --save-ckpt ml_service/grade_al_v2.pth

    # смешать несколько экспортов и исходный датасет-«эталон» (ImageFolder по сортам)
    python -m ml_service.train \\
        --patch-export EXP1 --patch-export EXP2 \\
        --base-dataset /path/to/imgs_by_class \\
        --epochs 20 --save-ckpt ml_service/grade_al_v3.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import (
    DEFAULT_CKPT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_CLASS_NAMES,
    MODEL_TO_CONTRACT,
    _build_module,
    _load_weights_into,
    _torch,
    device,
)

IMG_SIZE = 512  # train-FOV модели (см. train_grade_classifier_v2.py)

# Код класса КОНТРАКТА (src/config.py, он же id разметки) -> индекс выхода модели.
# Обратно к MODEL_TO_CONTRACT (= [3,1,2]): 3 тальк->0, 1 обычные->1, 2 тонкие->2.
CONTRACT_TO_MODEL = {int(c): i for i, c in enumerate(MODEL_TO_CONTRACT.tolist())}

# Имя папки в ImageFolder-экспорте -> код класса контракта. Первично читаем
# числовую колонку `label` манифеста; это фолбэк, если манифеста нет, а также
# принимает короткие псевдонимы имён классов.
FOLDER_TO_CONTRACT = {
    "talc": 3,
    "ordinary_intergrowth": 1,
    "ordinary": 1,
    "fine_intergrowth": 2,
    "fine": 2,
}

_IMG_EXT = (".jpg", ".jpeg", ".png")


@dataclass
class Sample:
    """Один обучающий патч: путь к картинке, индекс выхода модели, вес."""
    path: str
    model_idx: int
    weight: float
    contract_label: int


# --------------------------------------------------------------------------- #
# Сбор датасета (torch не нужен — потому и вынесено, тестируется без него)
# --------------------------------------------------------------------------- #

def read_patch_export(export_dir: str | Path) -> list[Sample]:
    """
    Прочитать ОДИН экспорт патчей (`active_learning_patch/<export_id>`).

    Первично — по `manifest.csv` (колонки `path`,`label`,`weight`): `label` это
    код класса контракта, `path` — относительный путь к патчу, `weight` — вес
    сэмпла (покрытие класса в области). Если манифеста нет — обходим
    `imgs/<class_name>/*.jpg` и мапим имя папки через FOLDER_TO_CONTRACT.
    Нетренируемые классы (фон/неопределённое) молча пропускаются.
    """
    export_dir = Path(export_dir)
    manifest = export_dir / "manifest.csv"
    out: list[Sample] = []

    if manifest.exists():
        with open(manifest, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    contract = int(row["label"])
                except (KeyError, ValueError, TypeError):
                    contract = FOLDER_TO_CONTRACT.get(str(row.get("label_name", "")).lower(), -1)
                if contract not in CONTRACT_TO_MODEL:
                    continue
                rel = row.get("path", "")
                if not rel:
                    continue
                try:
                    weight = float(row.get("weight", 1.0) or 1.0)
                except (ValueError, TypeError):
                    weight = 1.0
                out.append(Sample(
                    path=str(export_dir / rel),
                    model_idx=CONTRACT_TO_MODEL[contract],
                    weight=max(weight, 1e-3),
                    contract_label=contract,
                ))
        return out

    # Фолбэк: манифеста нет — читаем структуру папок imgs/<class>/.
    return _scan_imagefolder(export_dir / "imgs")


def read_base_dataset(root: str | Path) -> list[Sample]:
    """
    Прочитать «эталонный» датасет как ImageFolder: подпапки, имя = класс
    (talc / ordinary[_intergrowth] / fine[_intergrowth]). Вес всех сэмплов = 1.0.
    Служит якорем при дообучении, чтобы не забыть исходное распределение.
    """
    return _scan_imagefolder(Path(root))


def _scan_imagefolder(root: Path) -> list[Sample]:
    out: list[Sample] = []
    if not root.is_dir():
        return out
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        contract = FOLDER_TO_CONTRACT.get(cls_dir.name.lower())
        if contract is None or contract not in CONTRACT_TO_MODEL:
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in _IMG_EXT:
                out.append(Sample(
                    path=str(f), model_idx=CONTRACT_TO_MODEL[contract],
                    weight=1.0, contract_label=contract,
                ))
    return out


def class_histogram(samples: list[Sample]) -> dict[str, int]:
    """{имя_класса_модели: число сэмплов} — для лога и отчёта."""
    hist = {name: 0 for name in MODEL_CLASS_NAMES}
    for s in samples:
        hist[MODEL_CLASS_NAMES[s.model_idx]] += 1
    return hist


def stratified_split(samples: list[Sample], val_frac: float, seed: int
                     ) -> tuple[list[Sample], list[Sample]]:
    """Стратифицированное разбиение train/val по классу выхода модели."""
    rng = random.Random(seed)
    train, val = [], []
    by_cls: dict[int, list[Sample]] = {}
    for s in samples:
        by_cls.setdefault(s.model_idx, []).append(s)
    for _cls, items in by_cls.items():
        items = items[:]
        rng.shuffle(items)
        n_val = int(len(items) * val_frac)
        # держим хотя бы 1 пример класса в train, val — только если есть запас
        n_val = min(n_val, max(0, len(items) - 1))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# --------------------------------------------------------------------------- #
# torch Dataset (ленивый импорт)
# --------------------------------------------------------------------------- #

def _make_dataset(samples: list[Sample], augment: bool):
    torch = _torch()
    from PIL import Image  # noqa: PLC0415

    class PatchDataset(torch.utils.data.Dataset):
        def __init__(self, items, augment):
            self.items, self.augment = items, augment

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            s = self.items[idx]
            img = Image.open(s.path).convert("RGB")
            W, H = img.size
            side = min(H, W)
            if self.augment:
                y0 = random.randint(0, H - side)
                x0 = random.randint(0, W - side)
            else:
                y0, x0 = (H - side) // 2, (W - side) // 2
            img = img.crop((x0, y0, x0 + side, y0 + side)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32)
            if self.augment:
                if random.random() > 0.5:
                    arr = arr[:, ::-1].copy()
                if random.random() > 0.5:
                    arr = arr[::-1].copy()
                if random.random() > 0.5:
                    arr = np.rot90(arr, random.randint(1, 3)).copy()
                if random.random() > 0.3:
                    arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 255)
            arr = (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(arr.transpose(2, 0, 1)), s.model_idx

    return PatchDataset(samples, augment)


def macro_f1(preds, labels, n=3):
    preds, labels = np.array(preds), np.array(labels)
    f1s = []
    for c in range(n):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return float(np.mean(f1s)), [float(x) for x in f1s]


def _evaluate(model, loader, dev):
    torch = _torch()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(model(imgs.to(dev)).argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())
    if not labels:
        return 0.0, 0.0, [0.0, 0.0, 0.0]
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    f1, f1s = macro_f1(preds, labels)
    return acc, f1, f1s


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #

def finetune(
    samples: list[Sample],
    *,
    from_ckpt: str = DEFAULT_CKPT,
    save_ckpt: str,
    epochs: int = 15,
    lr: float = 1e-4,
    enc_lr_mult: float = 0.1,
    batch_size: int = 8,
    val_frac: float = 0.2,
    patience: int = 6,
    freeze_encoder: bool = False,
    seed: int = 42,
    weighted_sampling: bool = True,
    log=print,
) -> dict:
    """
    Дообучить GradeClassifier на списке патчей и сохранить лучший чекпоинт.

    Стратегия — как в train_grade_classifier_v2.py: старт с `from_ckpt`,
    разморозка энкодера с encoder_lr = lr*enc_lr_mult, cosine-расписание,
    взвешенная CE по обратной частоте класса. Дополнительно — WeightedRandomSampler
    по весам патчей (покрытие класса в области эксперта). Возвращает отчёт (dict).
    """
    torch = _torch()
    import torch.nn as nn  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device()

    train, val = stratified_split(samples, val_frac, seed)
    log(f"Патчей: всего {len(samples)}  train {len(train)}  val {len(val)}")
    log(f"  train по классам: {class_histogram(train)}")
    if val:
        log(f"  val по классам:   {class_histogram(val)}")

    train_ds = _make_dataset(train, augment=True)
    val_ds = _make_dataset(val, augment=False) if val else None

    if weighted_sampling and train:
        w = torch.tensor([s.weight for s in train], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train), replacement=True)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = (torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
              if val_ds is not None else None)

    model = _build_module().to(dev)
    if from_ckpt and os.path.exists(from_ckpt):
        # _load_weights_into: понимает и мультиголовый (encoder+head+bg_head), и
        # старый grade-only формат чекпоинта (bg_head тогда — фолбэк на вшитый
        # bg_head_best.pth), и переносит bg_head в СОХРАНЯЕМЫЙ ниже чекпоинт —
        # иначе он остался бы со случайной инициализацией.
        _load_weights_into(model, from_ckpt)
        log(f"Старт с чекпоинта: {from_ckpt}")
    else:
        log(f"[WARN] Чекпоинт {from_ckpt} не найден — обучение с нуля (случайная инициализация головы)")

    for p in model.encoder.parameters():
        p.requires_grad = not freeze_encoder
    if freeze_encoder:
        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
        log(f"Энкодер заморожен. LR головы={lr}")
    else:
        optimizer = torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": lr * enc_lr_mult},
            {"params": model.head.parameters(), "lr": lr},
        ])
        log(f"Энкодер разморожен. LR: head={lr}, encoder={lr * enc_lr_mult}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # веса классов = обратная частота (как в оригинале)
    counts = [max(1, sum(1 for s in train if s.model_idx == c)) for c in range(3)]
    cw = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
    cw = cw / cw.mean()
    criterion = nn.CrossEntropyLoss(weight=cw.to(dev))
    log(f"Веса классов {MODEL_CLASS_NAMES}: {[round(v, 2) for v in cw.tolist()]}")

    save_path = str(save_ckpt)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, patience_cnt = -1.0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            loss = criterion(model(imgs), lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / max(1, len(train_dl))

        if val_dl is not None:
            acc, f1, f1s = _evaluate(model, val_dl, dev)
            log(f"[{epoch:3d}/{epochs}] loss={avg_loss:.4f}  acc={acc:.4f}  macro-F1={f1:.4f}")
            history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, "macro_f1": f1})
            improved = f1 > best_f1
        else:
            log(f"[{epoch:3d}/{epochs}] loss={avg_loss:.4f}  (нет val — сохраняем каждую эпоху)")
            history.append({"epoch": epoch, "loss": avg_loss})
            improved = True  # без val сохраняем последнюю

        if improved:
            if "macro_f1" in history[-1]:
                best_f1 = max(best_f1, history[-1]["macro_f1"])
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            log(f"  ✓ сохранён {save_path}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                log(f"Ранняя остановка на эпохе {epoch}")
                break

    report = {
        "save_ckpt": save_path,
        "from_ckpt": from_ckpt,
        "num_patches": len(samples),
        "num_train": len(train),
        "num_val": len(val),
        "class_histogram_train": class_histogram(train),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_macro_f1": round(best_f1, 4) if best_f1 >= 0 else None,
        "history": history,
        "model_class_names": MODEL_CLASS_NAMES,
    }
    return report


# --------------------------------------------------------------------------- #
# Быстрое дообучение голов (интерактивное активное обучение на странице OreVision)
# --------------------------------------------------------------------------- #
# Полный finetune() разогревает энкодер и идёт минуты — для интерактивного цикла
# «правка эксперта → сразу результат» это слишком долго. Здесь энкодер ЗАМОРОЖЕН,
# его признаки (2048-D) считаются ОДИН раз на патч, а обучаются только головы
# (сорт: Linear 2048→256→3, фон: Linear 2048→1) — быстро даже на CPU, энкодер не
# трогаем. quick_finetune_multihead() сохраняет ОБЕ головы в ОДИН чекпоинт.

def _square_crops(pil, k: int, rng: random.Random) -> list[np.ndarray]:
    """k квадратных кропов патча -> массивы (IMG_SIZE,IMG_SIZE,3) float32.

    При k<=1 — центральный кроп без аугментаций; иначе случайные кропы + флипы/
    повороты (лёгкая аугментация, чтобы голова не переобучилась на один кадр).
    """
    from PIL import Image  # noqa: PLC0415

    W, H = pil.size
    side = min(W, H)
    arrs = []
    for _ in range(max(1, k)):
        if k <= 1 or W == side and H == side:
            x0, y0 = (W - side) // 2, (H - side) // 2
        else:
            x0 = rng.randint(0, W - side)
            y0 = rng.randint(0, H - side)
        crop = pil.crop((x0, y0, x0 + side, y0 + side)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(crop, dtype=np.float32)
        if k > 1:
            if rng.random() > 0.5:
                arr = arr[:, ::-1].copy()
            if rng.random() > 0.5:
                arr = arr[::-1].copy()
            if rng.random() > 0.5:
                arr = np.rot90(arr, rng.randint(1, 3)).copy()
        arrs.append(arr)
    return arrs


def encode_features(model, items, *, augment_k: int = 6, dev=None, batch: int = 16, seed: int = 42):
    """
    items: [(PIL.Image, model_idx, weight), ...]. Возвращает (feats, labels, weights)
    как torch-тензоры: feats (N,2048) — выход замороженного энкодера + avgpool,
    N = число_патчей × augment_k. Считается один раз, дальше учим только голову.
    """
    torch = _torch()
    dev = dev or device()
    rng = random.Random(seed)

    tensors, labels, weights = [], [], []
    for pil, idx, wt in items:
        for arr in _square_crops(pil, augment_k, rng):
            a = (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(torch.from_numpy(a.transpose(2, 0, 1)))
            labels.append(int(idx))
            weights.append(float(wt))

    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tensors), batch):
            b = torch.stack(tensors[i:i + batch]).to(dev)
            f = model.pool(model.encoder(b)[-1]).view(b.size(0), -1)
            feats.append(f.cpu())
    feats_t = torch.cat(feats, 0) if feats else torch.zeros((0, 2048))
    return (feats_t,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32))


def quick_finetune_multihead(
    items,
    bg_items,
    *,
    from_ckpt: str = DEFAULT_CKPT,
    save_ckpt: str,
    epochs: int = 60,
    lr: float = 1e-3,
    bg_epochs: int = 150,
    bg_lr: float = 5e-3,
    augment_k: int = 6,
    seed: int = 42,
    log=print,
) -> dict:
    """
    Быстрое дообучение ОБЕИХ голов (сорт + фон) поверх ОДНОГО замороженного
    энкодера и сохранение ОДНОГО мультиголового чекпоинта (encoder+head+bg_head
    в одном state_dict — см. ml_service/model.py). Раньше это были два отдельных
    чекпоинта (grade-only + bg-only Linear), из-за чего `ORE_ML_CKPT` не мог
    сам по себе подключить дообученный фон — теперь один файл, один env var.

    items: [(PIL.Image, model_idx, weight)] — патчи для головы сорта (правки
    эксперта с большим весом + якоря с малым, чтобы не схлопнуть классы).
    bg_items: [(PIL.Image, bg_label, weight)], bg_label 1=фон/0=руда — патчи для
    головы фона. Любой из списков может быть пустым — тогда соответствующая
    голова остаётся такой же, как в from_ckpt (не трогаем, не переинициализируем).
    """
    torch = _torch()
    import torch.nn as nn  # noqa: PLC0415

    if not items and not bg_items:
        raise ValueError("нет обучающих патчей ни для головы сорта, ни для головы фона")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device()

    model = _build_module().to(dev)
    if from_ckpt and os.path.exists(from_ckpt):
        _load_weights_into(model, from_ckpt)  # накопительно; понимает оба формата
        log(f"Старт с чекпоинта: {from_ckpt}")
    else:
        log(f"[WARN] Чекпоинт {from_ckpt} не найден — головы обучаются с нуля")
    for p in model.encoder.parameters():
        p.requires_grad = False

    report: dict = {"save_ckpt": str(save_ckpt), "from_ckpt": from_ckpt}

    if items:
        feats, labels, weights = encode_features(model, items, augment_k=augment_k, dev=dev, seed=seed)
        if feats.shape[0] == 0:
            raise ValueError("нет обучающих патчей для дообучения головы сорта")
        feats, labels, weights = feats.to(dev), labels.to(dev), weights.to(dev)

        counts = [max(1, int((labels == c).sum())) for c in range(3)]
        cw = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
        cw = cw / cw.mean()
        criterion = nn.CrossEntropyLoss(weight=cw.to(dev), reduction="none")
        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)

        model.head.train()
        losses = []
        for _ep in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = (criterion(model.head(feats), labels) * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.head.eval()
        with torch.no_grad():
            acc = float((model.head(feats).argmax(1) == labels).float().mean())

        log(f"[сорт] loss={losses[-1]:.4f}  train_acc={acc:.3f}")
        report.update({
            "num_patches": len(items),
            "num_feature_vectors": int(feats.shape[0]),
            "epochs": epochs,
            "final_loss": round(losses[-1], 4),
            "train_acc": round(acc, 4),
            "class_counts": {MODEL_CLASS_NAMES[c]: counts[c] for c in range(3)},
        })

    if bg_items:
        feats_bg, labels_bg, weights_bg = encode_features(
            model, bg_items, augment_k=augment_k, dev=dev, seed=seed
        )
        if feats_bg.shape[0] == 0:
            raise ValueError("нет обучающих патчей для дообучения головы фона")
        y = labels_bg.to(dev).float()                      # bg_label 0/1
        feats_bg, weights_bg = feats_bg.to(dev), weights_bg.to(dev)

        n_pos = float((y > 0.5).sum())                  # фон
        n_neg = float((y <= 0.5).sum())                 # руда
        # pos_weight = neg/pos балансирует редкий класс; ограничим, чтобы не взорвать.
        pw = min(20.0, max(0.05, (n_neg / n_pos) if n_pos > 0 else 1.0))
        criterion_bg = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], device=dev), reduction="none"
        )
        optimizer_bg = torch.optim.Adam(model.bg_head.parameters(), lr=bg_lr, weight_decay=1e-4)

        model.bg_head.train()
        losses_bg = []
        for _ep in range(1, bg_epochs + 1):
            optimizer_bg.zero_grad()
            logit = model.bg_head(feats_bg).squeeze(1)
            loss = (criterion_bg(logit, y) * weights_bg).mean()
            loss.backward()
            optimizer_bg.step()
            losses_bg.append(float(loss.item()))
        model.bg_head.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model.bg_head(feats_bg).squeeze(1))
            acc_bg = float(((prob > 0.5).float() == y).float().mean())
        model.bg_enabled = True

        log(f"[фон] loss={losses_bg[-1]:.4f}  train_acc={acc_bg:.3f}")
        report["bg"] = {
            "num_patches": len(bg_items),
            "num_feature_vectors": int(feats_bg.shape[0]),
            "epochs": bg_epochs,
            "pos_weight": round(pw, 3),
            "final_loss": round(losses_bg[-1], 4),
            "train_acc": round(acc_bg, 4),
            "label_counts": {"фон": int(n_pos), "руда": int(n_neg)},
        }

    model.eval()
    Path(save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_ckpt))
    log(f"✓ сохранён мультиголовый чекпоинт {save_ckpt}")

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def collect_samples(patch_exports, base_datasets) -> list[Sample]:
    samples: list[Sample] = []
    for d in patch_exports or []:
        got = read_patch_export(d)
        print(f"  экспорт патчей {d}: {len(got)} патчей")
        samples.extend(got)
    for d in base_datasets or []:
        got = read_base_dataset(d)
        print(f"  base-датасет {d}: {len(got)} изображений")
        samples.extend(got)
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser(description="Дообучение модели сортов руды на патчах active learning")
    ap.add_argument("--patch-export", action="append", default=[],
                    help="папка экспорта патчей (active_learning_patch/<id>); можно повторять")
    ap.add_argument("--base-dataset", action="append", default=[],
                    help="ImageFolder-датасет по классам (talc/ordinary/fine); можно повторять")
    ap.add_argument("--from-ckpt", default=DEFAULT_CKPT, help="стартовый чекпоинт (по умолчанию вшитый)")
    ap.add_argument("--save-ckpt", default=os.path.join(os.path.dirname(DEFAULT_CKPT), "grade_al_finetuned.pth"),
                    help="куда сохранить новый чекпоинт")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--freeze-encoder", action="store_true", help="учить только голову (быстро, консервативно)")
    ap.add_argument("--no-weighted-sampling", action="store_true", help="не использовать веса патчей в сэмплере")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not args.patch_export and not args.base_dataset:
        ap.error("нужен хотя бы один --patch-export или --base-dataset")

    print("Сбор датасета…")
    samples = collect_samples(args.patch_export, args.base_dataset)
    if not samples:
        ap.error("не найдено ни одного обучаемого патча — проверьте пути экспортов")
    print(f"Итого патчей: {len(samples)}  по классам: {class_histogram(samples)}")

    report = finetune(
        samples,
        from_ckpt=args.from_ckpt,
        save_ckpt=args.save_ckpt,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        patience=args.patience,
        freeze_encoder=args.freeze_encoder,
        weighted_sampling=not args.no_weighted_sampling,
        seed=args.seed,
    )

    report_path = str(Path(args.save_ckpt).with_suffix(".report.json"))
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово. Чекпоинт: {args.save_ckpt}\nОтчёт: {report_path}")
    print("Подключить новый чекпоинт к сервису:")
    print(f"  ORE_ML_CKPT={args.save_ckpt} python ml_service/server.py   # (или OREVISION_ML_MODE=local с тем же env)")


if __name__ == "__main__":
    main()
