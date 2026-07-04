"""
QUANTIZER — превращение связной пиксельной ОБЛАСТИ эксперта в набор
квадратных ПАТЧЕЙ train-разрешения (см. docs/PATCH_AL_REDESIGN.md, §5).

Эксперт обводит связную область (лассо) и ставит ей ОДИН класс. Модель же
классификатор патчей, поэтому для дообучения область надо нарезать на
кусочно-блочные патчи `S×S`, все с меткой этого класса. Патчи МОГУТ
перекрываться (overlap>0) — это осознанно, дешёвая пространственная
аугментация: одна область даёт больше обучающих примеров.

ИНВАРИАНТ (важно для устойчивости конвейера): `quantize_region` НИКОГДА не
падает и ВСЕГДА возвращает `(list_of_patches, reason)` — список может быть
пустым. Любое исключение внутри превращается в `([], "error: ...")`.

Границы кадра, тонкие жилы, области меньше патча, стык с соседним классом —
все крайние случаи обрабатываются явно (см. таблицу в §5 патча).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from . import config

try:
    from scipy import ndimage  # эрозия/компоненты/дистанс-трансформ
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001 — деградируем без scipy (см. _label/_erode ниже)
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #

@dataclass
class Patch:
    """Один обучающий патч, вырезанный из области (готов к записи в ImageFolder)."""
    image: Image.Image      # SxS RGB
    label: int              # код класса (единое пространство, config.CLASS_*)
    weight: float           # вес сэмпла (1.0 обычный, 0.5 — апскейл-патч)
    inside: float           # доля площади патча внутри области (0..1)
    upsampled: bool         # True — окно было меньше S и растянуто до S
    x: int                  # левый-верх патча в координатах источника
    y: int
    source_size: int        # сторона окна, реально взятого из источника (до resize до S)
    provenance: dict[str, Any] = field(default_factory=dict)  # source_image_id/region_id доб. вызывающим


# --------------------------------------------------------------------------- #
# Публичная функция
# --------------------------------------------------------------------------- #

def quantize_region(
    M: np.ndarray,
    image: Image.Image,
    L: int,
    S: int = config.PATCH_SIZE,
    tau: float = config.PATCH_TAU_COVERAGE,
    overlap: float = config.PATCH_OVERLAP,
    N: int = config.PATCH_CAP_N,
    seed: int = 0,
) -> tuple[list[Patch], str]:
    """
    Нарезать связную область `M` (bool-маска в пиксельной сетке `image`) на
    патчи `S×S`, все с меткой `L`.

    Параметры:
      M       — 2D bool/uint8 маска области (True/≠0 = область). Форма (H, W),
                совпадает с размером `image`.
      image   — PIL.Image источника (патчи режутся именно из него).
      L       — код класса патчей (config.CLASS_*).
      S       — сторона патча в пикселях источника (одно train-FOV).
      tau     — минимальная доля патча, лежащая внутри области (порог покрытия).
      overlap — доля перекрытия соседних патчей (>0 — намеренно, аугментация).
      N       — кап на число патчей (farthest-point-разнос).
      seed    — сид RNG (обычно hash(region_id)) — экспорт воспроизводим.

    Возвращает (patches, reason). reason ∈ {"ok","thin_region","empty_region",
    "too_small","error: ..."}.
    """
    try:
        return _quantize_region(M, image, L, S, tau, overlap, N, seed)
    except Exception as e:  # noqa: BLE001 — инвариант: никогда не падаем
        return [], f"error: {type(e).__name__}: {e}"


def _quantize_region(
    M: np.ndarray, image: Image.Image, L: int, S: int,
    tau: float, overlap: float, N: int, seed: int,
) -> tuple[list[Patch], str]:
    M = np.asarray(M).astype(bool)
    if M.ndim != 2:
        raise ValueError("M должна быть 2D-маской")
    if not M.any():
        return [], "empty_region"

    S = max(1, int(S))
    tau = float(min(max(tau, 0.0), 1.0))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    # 1) Отодвигаемся от границы с соседним классом: работаем по «ядру» области.
    margin = int(S * (1.0 - tau) / 2)
    core = _erode(M, margin) if margin > 0 else M
    src = core if core.any() else M   # тонкая область — работаем по всей M

    stride = max(1, int(S * (1.0 - overlap)))   # overlap>0 → перекрытие
    patches: list[Patch] = []

    for comp in _components(src):
        area = int(comp.sum())
        if area >= tau * S * S:
            patches.extend(_grid_patches(comp, M, image, L, S, stride, tau, rng))
        elif area >= 0.25 * S * S:
            p = _shrunk_patch(comp, image, L, S)
            if p is not None:
                patches.append(p)
        else:
            continue  # слишком мелкая — см. фолбэк ниже

    if not patches:
        return _thin_region_fallback(M, image, L, S)

    patches = _dedup_by_xy(patches)
    return _farthest_point_cap(patches, N, seed), "ok"


# --------------------------------------------------------------------------- #
# Обычный случай: сетка патчей по bbox компоненты
# --------------------------------------------------------------------------- #

def _grid_patches(
    comp: np.ndarray, M: np.ndarray, image: Image.Image, L: int,
    S: int, stride: int, tau: float, rng: np.random.Generator,
) -> list[Patch]:
    ys, xs = np.where(comp)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    jit = max(0, stride // 2)

    out: list[Patch] = []
    yy = y0
    while yy <= y1:
        xx = x0
        while xx <= x1:
            py = yy + int(rng.integers(-jit, jit + 1)) if jit else yy
            px = xx + int(rng.integers(-jit, jit + 1)) if jit else xx
            inside = _inside_fraction(M, px, py, S)
            if inside >= tau:
                crop = _crop_pad(image, px, py, S)
                if crop is not None:
                    out.append(Patch(
                        image=crop, label=L, weight=1.0, inside=inside,
                        upsampled=False, x=px, y=py, source_size=S,
                    ))
            xx += stride
        yy += stride
    return out


# --------------------------------------------------------------------------- #
# Область меньше патча: ужать окно под площадь и апскейлить до S
# --------------------------------------------------------------------------- #

def _shrunk_patch(comp: np.ndarray, image: Image.Image, L: int, S: int) -> Optional[Patch]:
    area = int(comp.sum())
    tau = config.PATCH_TAU_COVERAGE
    s2 = int(math.ceil(math.sqrt(area / max(tau, 1e-6))))
    s2 = max(1, s2)
    ys, xs = np.where(comp)
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))
    bx0 = cx - s2 // 2
    by0 = cy - s2 // 2
    crop = _crop_pad(image, bx0, by0, s2)
    if crop is None:
        return None
    crop = crop.resize((S, S), Image.BILINEAR)
    inside = area / float(s2 * s2)
    return Patch(
        image=crop, label=L, weight=0.5, inside=min(inside, 1.0),
        upsampled=True, x=bx0, y=by0, source_size=s2,
    )


# --------------------------------------------------------------------------- #
# Тонкая/вытянутая область (жила) или всё отсеялось: центроидный апскейл-патч
# --------------------------------------------------------------------------- #

def _thin_region_fallback(
    M: np.ndarray, image: Image.Image, L: int, S: int,
) -> tuple[list[Patch], str]:
    ys, xs = np.where(M)
    if ys.size == 0:
        return [], "empty_region"
    area = int(M.sum())
    # Слишком крошечная область (шум/дубли) — в трейн не берём, но не теряем:
    # причина too_small (вызывающий переведёт регион в needs_expert_review).
    if area < max(16, int(0.02 * S * S)):
        return [], "too_small"

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    side = int(max(y1 - y0, x1 - x0)) + 1
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    bx0 = cx - side // 2
    by0 = cy - side // 2
    crop = _crop_pad(image, bx0, by0, side)
    if crop is None:
        return [], "too_small"
    crop = crop.resize((S, S), Image.BILINEAR)
    inside = _inside_fraction(M, bx0, by0, side)
    return [Patch(
        image=crop, label=L, weight=0.5, inside=inside,
        upsampled=True, x=bx0, y=by0, source_size=side,
    )], "thin_region"


# --------------------------------------------------------------------------- #
# Геометрия
# --------------------------------------------------------------------------- #

def _inside_fraction(M: np.ndarray, x0: int, y0: int, S: int) -> float:
    """Доля площади окна [x0,y0,S] (от S*S), реально лежащая внутри области M."""
    H, W = M.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + S), min(H, y0 + S)
    if cx1 <= cx0 or cy1 <= cy0:
        return 0.0
    return float(M[cy0:cy1, cx0:cx1].sum()) / float(S * S)


def _crop_pad(image: Image.Image, x0: int, y0: int, S: int) -> Optional[Image.Image]:
    """
    Вырезать окно [x0,y0,S] из image. Часть, вылезшую за край кадра, добираем
    reflect-паддингом до S×S (как договорено в §5 — «добор у края»).
    """
    W, H = image.size
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + S), min(H, y0 + S)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = image.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    arr = np.asarray(crop)
    pad_top = cy0 - y0
    pad_left = cx0 - x0
    ph, pw = arr.shape[:2]
    pad_bottom = S - ph - pad_top
    pad_right = S - pw - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        pad_top, pad_bottom = max(0, pad_top), max(0, pad_bottom)
        pad_left, pad_right = max(0, pad_left), max(0, pad_right)
        mode = "reflect" if (ph > 1 and pw > 1) else "edge"
        arr = np.pad(
            arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode
        )
        arr = arr[:S, :S]
    return Image.fromarray(arr, mode="RGB")


def _dedup_by_xy(patches: list[Patch]) -> list[Patch]:
    """Дедуп ТОЛЬКО по идентичным (x,y) — перекрытие само по себе намеренно."""
    seen: set[tuple[int, int]] = set()
    out: list[Patch] = []
    for p in patches:
        key = (p.x, p.y)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _farthest_point_cap(patches: list[Patch], N: int, seed: int) -> list[Patch]:
    """Оставить ≤ N пространственно РАЗНЕСЁННЫХ патчей (жадный farthest-point)."""
    if N <= 0 or len(patches) <= N:
        return patches
    pts = np.array([(p.x, p.y) for p in patches], dtype=np.float64)
    start = int(seed) % len(patches)
    chosen = [start]
    dists = np.linalg.norm(pts - pts[start], axis=1)
    while len(chosen) < N:
        i = int(np.argmax(dists))
        if i in chosen:
            break
        chosen.append(i)
        dists = np.minimum(dists, np.linalg.norm(pts - pts[i], axis=1))
    return [patches[i] for i in chosen]


# --------------------------------------------------------------------------- #
# Морфология (scipy при наличии, иначе аккуратный numpy-фолбэк)
# --------------------------------------------------------------------------- #

def _erode(M: np.ndarray, margin: int) -> np.ndarray:
    """Эрозия диском радиуса ≈margin — через дистанс-трансформ (один проход)."""
    if margin <= 0:
        return M
    if _HAVE_SCIPY:
        return ndimage.distance_transform_edt(M) >= margin
    # Фолбэк без scipy: приблизительная прямоугольная эрозия по осям.
    out = M.copy()
    for _ in range(margin):
        shifted = out.copy()
        shifted[1:, :] &= out[:-1, :]
        shifted[:-1, :] &= out[1:, :]
        shifted[:, 1:] &= out[:, :-1]
        shifted[:, :-1] &= out[:, 1:]
        out = shifted
        if not out.any():
            break
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Список bool-масок связных компонент (4-связность)."""
    if not mask.any():
        return []
    if _HAVE_SCIPY:
        labeled, n = ndimage.label(mask)
        return [labeled == i for i in range(1, n + 1)]
    # Простой BFS-фолбэк без scipy.
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    H, W = mask.shape
    for sy in range(H):
        for sx in range(W):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            comp = np.zeros_like(mask, dtype=bool)
            stack = [(sy, sx)]
            visited[sy, sx] = True
            while stack:
                y, x = stack.pop()
                comp[y, x] = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(comp)
    return comps
