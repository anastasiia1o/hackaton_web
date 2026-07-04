# -*- coding: utf-8 -*-
"""
seglib — общее ядро сегментации аншлифов по цветовой палитре микроскопа.

Содержит:
  * выбор бэкенда (GPU через CuPy при наличии, иначе CPU/NumPy);
  * добычу «плоских» пикселей и построение палитры KMeans (как в ноутбуке 07);
  * разметку палитры на семантические группы и контрастный алфавит;
  * пространственную сегментацию (mean-field Potts), идентичную ноутбукам 08–09,
    но работающую как на NumPy, так и на CuPy без изменения алгоритма.

Алгоритм НЕ меняется между CPU и GPU: те же формулы, тайлинг и halo-перекрытие,
меняется только модуль массивов (numpy / cupy) и функция uniform_filter.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WIN, PCT = 7, 35                      # окно локального СКО и перцентиль «плоскости»
PALETTE_K, PALETTE_MERGE = 24, 0      # 24 центра KMeans, без слияния


# --------------------------------------------------------------------------- #
#  Бэкенд: GPU (CuPy) при наличии, иначе CPU (NumPy)
# --------------------------------------------------------------------------- #
def get_backend(prefer_gpu=True):
    """Return (array_module, uniform_filter1d, device_name)."""
    if prefer_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import uniform_filter1d as cupy_uf1d
            if cp.cuda.runtime.getDeviceCount() > 0:
                name = cp.cuda.runtime.getDeviceProperties(0)['name']
                if isinstance(name, bytes):
                    name = name.decode('utf-8', 'ignore')
                return cp, cupy_uf1d, f'cuda:0 ({name})'
        except Exception as error:
            print(
                f'GPU backend unavailable, fallback to CPU: '
                f'{type(error).__name__}: {error}'
            )
    from scipy.ndimage import uniform_filter1d as scipy_uf1d
    return np, scipy_uf1d, 'cpu'

def _to_host(xp, arr):
    return xp.asnumpy(arr) if xp.__name__ == 'cupy' else np.asarray(arr)


# --------------------------------------------------------------------------- #
#  Добыча палитры (как в ноутбуке 07)
# --------------------------------------------------------------------------- #
def merge_weighted_centers(centers, masses, threshold):
    centers, masses = centers.copy(), masses.astype(float).copy()
    while len(centers) > 1:
        d = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= threshold:
            break
        total = masses[i] + masses[j]
        centers[i] = (centers[i] * masses[i] + centers[j] * masses[j]) / total
        masses[i] = total
        centers = np.delete(centers, j, axis=0)
        masses = np.delete(masses, j)
    order = np.argsort(-masses)
    return centers[order], masses[order]


def sample_flat_pixels(path, cap, seed, max_side=3500, tile_rows=512):
    """Плоские пиксели снимка. Большие панорамы: draft-декод (память) + нарезка на тайлы."""
    from scipy.ndimage import uniform_filter
    with Image.open(path) as im:
        im.draft('RGB', (max_side, max_side))         # уменьшённое DCT-декодирование
        arr = np.asarray(im.convert('RGB'))
    H, W, _ = arr.shape
    chunks = []
    for y0 in range(0, H, tile_rows):
        tile = arr[y0:y0 + tile_rows].astype(np.float32)
        chans = []
        for c in range(3):
            v = tile[..., c]
            m = uniform_filter(v, WIN)
            m2 = uniform_filter(v * v, WIN)
            chans.append(np.sqrt(np.maximum(m2 - m * m, 0)))
        flatness = np.stack(chans, -1).max(-1)
        chunks.append(tile[flatness <= np.percentile(flatness, PCT)])
    pixels = np.concatenate(chunks)
    if len(pixels) > cap:
        pixels = pixels[np.random.default_rng(seed).choice(len(pixels), cap, replace=False)]
    return pixels, (W, H)


def build_palette(pixels, k=PALETTE_K, merge=PALETTE_MERGE, seed=0, fit_cap=150_000):
    from sklearn.cluster import KMeans
    X = pixels
    if len(X) > fit_cap:
        X = X[np.random.default_rng(seed).choice(len(X), fit_cap, replace=False)]
    model = KMeans(n_clusters=min(k, len(X)), n_init=5, random_state=seed).fit(X / 255.0)
    masses = np.bincount(model.labels_, minlength=model.n_clusters).astype(float)
    return merge_weighted_centers(model.cluster_centers_ * 255.0, masses, merge)


# --------------------------------------------------------------------------- #
#  Разметка палитры и контрастный алфавит (как в ноутбуке 07, ячейки 10–11)
# --------------------------------------------------------------------------- #
def tentative_domain_label(rgb):
    """Доменная гипотеза (нейтрально-серая гамма отражённого света).
    Яркость=отражательность; на ярком конце тёплость отделяет сульфид от оксида.
    НЕ ground truth."""
    Y = float(np.mean(rgb))
    warm = float((rgb[0] + rgb[1]) / 2 - rgb[2])
    if Y < 19:
        return 'очень тёмное'
    if Y < 60:
        return 'тёмная силикатная матрица'
    if Y < 125:
        return 'среднее тёплое (сульфид)' if warm > 8 else 'серый оксид'
    if Y < 152:
        return 'яркий сульфид'
    return 'очень яркий сульфид'


GROUP_OF_CLASS = {
    'очень тёмное': 'фиолетовый',
    'тёмная силикатная матрица': 'синий',
    'серый оксид': 'зелёный',
    'среднее тёплое (сульфид)': 'оранжевый',
    'яркий сульфид': 'оранжевый',
    'очень яркий сульфид': 'красный',
}
GROUP_BASE = {
    'фиолетовый': (0.55, 0.20, 0.75),
    'синий':      (0.15, 0.35, 0.90),
    'зелёный':    (0.15, 0.70, 0.25),
    'оранжевый':  (0.95, 0.55, 0.10),
    'красный':    (0.90, 0.15, 0.15),
}
GROUP_MEANING = {
    'фиолетовый': 'Поры / трещины / темнейшее',
    'синий':      'Силикатная матрица (гангу)',
    'зелёный':    'Магнетит / серый оксид',
    'оранжевый':  'Пирротин / пентландит (сульфид)',
    'красный':    'Халькопирит / пирит (яркий сульфид)',
}
GROUP_ORDER = ['фиолетовый', 'синий', 'зелёный', 'оранжевый', 'красный']


def build_contrast_alphabet(palette):
    """Возвращает (contrast, group): цвет = группа, оттенок (value) = яркость внутри группы."""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    palette = np.asarray(palette, np.float32)
    group = [GROUP_OF_CLASS[tentative_domain_label(c)] for c in palette]
    brightness = palette.mean(axis=1)
    contrast = np.zeros((len(palette), 3), np.float32)
    for gname in GROUP_ORDER:
        members = [i for i in range(len(palette)) if group[i] == gname]
        if not members:
            continue
        hue, sat, _ = rgb_to_hsv(np.array(GROUP_BASE[gname]))
        b = brightness[members]
        shade = (b - b.min()) / (b.max() - b.min() + 1e-9) if len(members) > 1 else np.ones(1)
        for idx, t in zip(members, np.atleast_1d(shade)):
            contrast[idx] = hsv_to_rgb((hue, sat, 0.55 + 0.45 * t))
    return contrast, np.array(group)


def apply_gray_darkest(palette, contrast, n=5):
    """Топ-N самых тёмных цветов → контрастная ЧБ-шкала (как в ноутбуке 09)."""
    palette = np.asarray(palette, np.float32)
    contrast = np.asarray(contrast, np.float32).copy()
    if n <= 0:
        return contrast
    brightness = palette.mean(axis=1)
    order = np.argsort(brightness)[:n]
    order = order[np.argsort(brightness[order])]
    grays = np.linspace(0.12, 0.88, len(order))
    for gi, idx in enumerate(order):
        contrast[idx] = grays[gi]
    return contrast


# --------------------------------------------------------------------------- #
#  Пространственная сегментация (mean-field Potts) — CPU/GPU одинаково
# --------------------------------------------------------------------------- #
def nearest_reference_labels_spatial(rgb, centers, xp, uniform_filter1d,
                                     tau=12.0, lam=0.9, iters=3, radius=3,
                                     max_tile_pixels=500_000):
    """Mean-field palette segmentation using float32 and tiled processing.

    Squared distances use the GEMM identity without an (N,K,3) temporary.
    The 2D box filter is evaluated as two separable uniform_filter1d calls.
    """
    H, W, _ = rgb.shape
    centers = xp.asarray(centers, dtype=xp.float32)
    n_centers = len(centers)
    center_norm = xp.sum(centers * centers, axis=1)[None, :]
    inv_tau2 = xp.float32(-1.0 / (tau * tau))
    lam_f = xp.float32(lam)
    size = 2 * radius + 1
    halo = radius * iters
    rows_per_tile = max(1, int(max_tile_pixels // W))
    labels = np.empty((H, W), np.int16)

    for y0 in range(0, H, rows_per_tile):
        y1 = min(H, y0 + rows_per_tile)
        a, b = max(0, y0 - halo), min(H, y1 + halo)
        h = b - a

        block = xp.asarray(rgb[a:b], dtype=xp.float32).reshape(-1, 3)
        block_norm = xp.sum(block * block, axis=1, keepdims=True)
        unary = block_norm + center_norm - xp.float32(2.0) * (block @ centers.T)
        xp.maximum(unary, xp.float32(0.0), out=unary)
        unary *= inv_tau2
        unary -= unary.max(axis=1, keepdims=True)

        belief = xp.exp(unary)
        belief /= belief.sum(axis=1, keepdims=True)
        belief = belief.reshape(h, W, n_centers)
        unary_3d = unary.reshape(h, W, n_centers)

        for _ in range(iters):
            temp = uniform_filter1d(
                belief, size=size, axis=0, mode='nearest')
            logit = uniform_filter1d(
                temp, size=size, axis=1, mode='nearest')
            logit *= lam_f
            logit += unary_3d
            logit -= logit.max(axis=2, keepdims=True)
            xp.exp(logit, out=logit)
            logit /= logit.sum(axis=2, keepdims=True)
            belief = logit

        lab = belief.argmax(axis=2).astype(xp.int16)
        top = y0 - a
        labels[y0:y1] = _to_host(xp, lab[top:top + (y1 - y0)])

    return labels

def save_palette(folder, palette, masses, contrast, group):
    """Сохраняет палитру микроскопа в папку `folder`:
      palette.npz  — исходные цвета микроскопа (palette, masses, group);
      contrast.npz — соответствующая контрастная (цветная) шкала (contrast).
    """
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'palette.npz',
                        palette=np.asarray(palette, np.float32),
                        masses=np.asarray(masses), group=np.asarray(group))
    np.savez_compressed(folder / 'contrast.npz',
                        contrast=np.asarray(contrast, np.float32))
    return folder


def load_palette(path):
    """Загружает палитру микроскопа. Возвращает dict с ключами palette, contrast, group, masses.

    `path` может быть:
      * папкой микроскопа (palettes/<имя>/) с файлами palette.npz + contrast.npz;
      * единым *.npz (совместимость со старым *_palette_labeled.npz из ноутбука 07).
    Если contrast/group отсутствуют — строятся на лету из самой палитры.
    """
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        pal = np.load(path / 'palette.npz', allow_pickle=True)
        palette = pal['palette'].astype(np.float32)
        masses = pal['masses'] if 'masses' in pal.files else np.ones(len(palette))
        group = pal['group'] if 'group' in pal.files else None
        cpath = path / 'contrast.npz'
        if cpath.exists():
            contrast = np.load(cpath, allow_pickle=True)['contrast'].astype(np.float32)
        else:
            contrast, group = build_contrast_alphabet(palette)
        return {'palette': palette, 'contrast': contrast, 'group': group, 'masses': masses}

    d = np.load(path, allow_pickle=True)
    palette = d['palette'].astype(np.float32)
    if 'contrast' in d.files:
        contrast = d['contrast'].astype(np.float32)
        group = d['group'] if 'group' in d.files else None
    else:
        contrast, group = build_contrast_alphabet(palette)
    masses = d['masses'] if 'masses' in d.files else np.ones(len(palette))
    return {'palette': palette, 'contrast': contrast, 'group': group, 'masses': masses}


def load_all_palettes(pal_dir):
    """Загружает все палитры-подпапки из pal_dir. Возвращает dict имя -> {palette, contrast, ...}."""
    from pathlib import Path
    pal_dir = Path(pal_dir)
    out = {}
    for sub in sorted(pal_dir.iterdir()):
        if sub.is_dir() and (sub / 'palette.npz').exists():
            out[sub.name] = load_palette(sub)
    return out


# --------------------------------------------------------------------------- #
#  Автоопределение палитры снимка (какой микроскоп)
# --------------------------------------------------------------------------- #
def quant_error(pixels, centers):
    """Средняя ошибка квантования: среднее расстояние пикселя до ближайшего цвета палитры.
    Чем меньше — тем лучше снимок описывается этой палитрой."""
    X = np.asarray(pixels, np.float32)
    C = np.asarray(centers, np.float32)
    d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(2))
    return float(d.min(1).mean())


def sample_pixels_for_classify(path, max_side=240, cap=3000, seed=0):
    """Быстрая выборка пикселей уменьшённой копии снимка — для классификации палитры."""
    with Image.open(path) as im:
        im.draft('RGB', (max_side, max_side))
        a = np.asarray(im.convert('RGB')).reshape(-1, 3).astype(np.float32)
    if len(a) > cap:
        a = a[np.random.default_rng(seed).choice(len(a), cap, replace=False)]
    return a


def classify_palette(pixels, palettes, reject_error=None):
    """Определяет, какой палитре принадлежит снимок, по минимальной ошибке квантования.

    pixels   — (n,3) выборка пикселей снимка;
    palettes — dict имя -> centers (k,3) ЛИБО имя -> {'palette': centers, ...};
    reject_error — если задан и минимальная ошибка выше — вернёт 'unknown' (чужая палитра).

    Возвращает (имя_лучшей, {имя: ошибка}).
    """
    errs = {}
    for name, val in palettes.items():
        centers = val['palette'] if isinstance(val, dict) else val
        errs[name] = quant_error(pixels, centers)
    best = min(errs, key=errs.get)
    if reject_error is not None and errs[best] > reject_error:
        best = 'unknown'
    return best, errs
