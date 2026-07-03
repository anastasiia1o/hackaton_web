"""
VIEWER — построение цветного overlay поверх исходного изображения.

Поток B (UI) отвечает за отрисовку, но саму цветную маску удобно собирать
здесь одной функцией, чтобы цвета классов были едиными (берём из config).

Для MVP делаем статичный overlay (наложение полупрозрачной маски).
Zoom/pan/minimap для больших панорам — задача потока B поверх этого
(см. PLAN_AGENT_B.md): tiled-загрузка и streamlit-image-zoom/навигатор.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

from src import config

# Максимальная сторона картинки, которую отдаём в браузер интерактивному вьюеру.
# Панорамы бывают гигапиксельными (27000×21000 ≈ 573 Мп) — целиком их в base64
# не отдать, поэтому уменьшаем для показа, а zoom/pan работают на стороне клиента.
DISPLAY_MAX_DIM = 2600


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Превратить маску кодов классов в цветное RGBA-изображение."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in config.CLASS_COLORS.items():
        rgba[mask == cls] = color
    return Image.fromarray(rgba, mode="RGBA")


def make_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    show_classes: set[int] | None = None,
    opacity: float = 1.0,
) -> Image.Image:
    """
    Наложить цветную маску на исходное изображение.

    show_classes — какие классы показывать (для вкл/выкл слоёв в UI).
    opacity      — общий множитель прозрачности слоя маски (0..1).
    """
    base = base_image.convert("RGBA")
    if base.size != (mask.shape[1], mask.shape[0]):
        # Приводим маску к размеру картинки (на случай масштабирования превью).
        mask_img = Image.fromarray(mask, mode="L").resize(base.size, Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8)

    if show_classes is not None:
        # Обнуляем скрытые классы, чтобы они не подсвечивались.
        filtered = np.where(np.isin(mask, list(show_classes)), mask, 0).astype(np.uint8)
        mask = filtered

    overlay = colorize_mask(mask)
    if opacity < 1.0:
        alpha = np.array(overlay.split()[-1], dtype=np.float32) * float(opacity)
        overlay.putalpha(Image.fromarray(alpha.astype(np.uint8), mode="L"))

    return Image.alpha_composite(base, overlay)


# Потолок памяти при инспекции: сколько мегапикселей максимум декодируем за раз.
# Гигапиксельный JPEG нельзя crop'нуть без полного декодирования, поэтому
# ограничиваем пиковую память (≈ N Мп * 3 байта).
INSPECT_MAX_MP = 120


def crop_region_highres(
    image_path: str,
    x0f: float, y0f: float, x1f: float, y1f: float,
    out_max: int = 1600,
    decode_cap_mp: int = INSPECT_MAX_MP,
) -> tuple[Image.Image, dict]:
    """
    Вырезать участок изображения в максимально возможном разрешении (в пределах
    лимита памяти) — «инспектор» для панорам.

    Координаты участка — в долях (0..1). Для нормальных изображений возвращаем
    участок в НАТИВНОМ разрешении; для гигапиксельных сначала декодируем с
    понижением до decode_cap_mp Мп (иначе не хватит памяти), поэтому детализация
    ограничена — об этом сообщаем во втором элементе (dict с метаданными).
    """
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image_path)
    W0, H0 = im.size
    full_mp = (W0 * H0) / 1e6

    # Насколько пришлось понизить полное изображение, чтобы влезть в лимит памяти.
    if full_mp > decode_cap_mp:
        factor = (full_mp / decode_cap_mp) ** 0.5
        try:
            im.draft("RGB", (int(W0 / factor), int(H0 / factor)))
        except Exception:
            pass
    im = im.convert("RGB")
    Wd, Hd = im.size                      # реальный размер после draft-декода
    native_scale = Wd / W0                # 1.0 = нативно, <1 = понижено

    x0, y0 = int(min(x0f, x1f) * Wd), int(min(y0f, y1f) * Hd)
    x1, y1 = int(max(x0f, x1f) * Wd), int(max(y0f, y1f) * Hd)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    crop = im.crop((x0, y0, x1, y1))

    downscaled = False
    if max(crop.size) > out_max:
        crop.thumbnail((out_max, out_max), Image.LANCZOS)
        downscaled = True

    meta = {
        "orig_size": (W0, H0),
        "region_px_orig": [int(min(x0f, x1f) * W0), int(min(y0f, y1f) * H0),
                           int(abs(x1f - x0f) * W0), int(abs(y1f - y0f) * H0)],
        "native_scale": round(native_scale, 4),     # 1.0 => участок нативный
        "shown_size": crop.size,
        "capped": native_scale < 1.0,                # True => исходник был понижен
        "shown_downscaled": downscaled,
    }
    return crop, meta


def preview_region(
    image: Image.Image,
    x0f: float, y0f: float, x1f: float, y1f: float,
    color: tuple[int, int, int, int] = (255, 210, 60, 90),
) -> Image.Image:
    """
    Нарисовать прямоугольную область поверх изображения (для экспертной коррекции).

    Координаты задаются В ДОЛЯХ размера изображения (0..1), чтобы не зависеть от
    масштаба превью. Возвращает RGB-копию с полупрозрачной заливкой и рамкой.
    """
    from PIL import ImageDraw

    base = image.convert("RGBA")
    w, h = base.size
    x0, y0 = int(min(x0f, x1f) * w), int(min(y0f, y1f) * h)
    x1, y1 = int(max(x0f, x1f) * w), int(max(y0f, y1f) * h)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    outline = (color[0], color[1], color[2], 255)
    draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline, width=max(2, w // 400))
    return Image.alpha_composite(base, layer).convert("RGB")


def load_confidence(conf_path: str) -> np.ndarray:
    """Загрузить grayscale-карту уверенности как 2D-массив 0..255 (ярче = увереннее)."""
    return np.array(Image.open(conf_path).convert("L"), dtype=np.uint8)


def colorize_confidence(conf: np.ndarray, opacity: float = 0.5) -> Image.Image:
    """
    Раскрасить карту уверенности в тепловой слой RGBA.

    Идея слоя — сразу видеть, ГДЕ модель не уверена: низкая уверенность светится
    красным, средняя — жёлтым, высокая — зелёным. Прозрачность общая (opacity),
    чтобы под слоем читалось исходное изображение.
    """
    c = conf.astype(np.float32) / 255.0                 # 0..1, выше = увереннее
    r = np.clip(2.0 * (1.0 - c), 0.0, 1.0)              # мало уверенности -> красный
    g = np.clip(2.0 * c, 0.0, 1.0)                       # много уверенности -> зелёный
    b = np.zeros_like(c)
    a = np.full_like(c, float(opacity))
    rgba = (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def add_confidence_layer(
    composite: Image.Image, conf: np.ndarray, opacity: float = 0.5
) -> Image.Image:
    """Наложить тепловой слой уверенности поверх уже собранного изображения."""
    layer = colorize_confidence(conf, opacity=opacity)
    if layer.size != composite.size:
        layer = layer.resize(composite.size, Image.BILINEAR)
    return Image.alpha_composite(composite.convert("RGBA"), layer)


def confidence_legend() -> list[tuple[str, str]]:
    """Легенда теплового слоя уверенности: (подпись, hex-цвет)."""
    return [
        ("низкая уверенность", "#ff0000"),
        ("средняя", "#ffff00"),
        ("высокая", "#00cc00"),
    ]


def legend_items() -> list[tuple[str, str]]:
    """Легенда для UI: (название класса, hex-цвет)."""
    items = []
    for cls, name in config.CLASS_NAMES.items():
        if cls == config.CLASS_BACKGROUND:
            continue
        r, g, b, _a = config.CLASS_COLORS[cls]
        items.append((name, f"#{r:02x}{g:02x}{b:02x}"))
    return items


def load_display_image(image_path: str, max_dim: int = DISPLAY_MAX_DIM) -> Image.Image:
    """
    Безопасно открыть изображение для показа, уменьшив большие панорамы.

    Гигапиксельные JPEG превышают лимит PIL (`MAX_IMAGE_PIXELS`) и не влезают
    в память, если декодировать в полном разрешении. Поэтому:
    - снимаем лимит "decompression bomb" (данные локальные, доверенные);
    - для JPEG используем `draft()` — декодер сразу читает в уменьшенном
      масштабе (1/2, 1/4, 1/8), это кратно экономит время и память;
    - затем `thumbnail()` доводит до нужной стороны с сохранением пропорций.

    Возвращает RGB-изображение со стороной не больше `max_dim`.
    """
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image_path)
    try:
        # draft работает только для JPEG; для остальных форматов молча пропускаем.
        im.draft("RGB", (max_dim, max_dim))
    except Exception:
        pass
    im = im.convert("RGB")
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return im


def _to_data_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    """Закодировать PIL-изображение в data:URI (base64) для вставки в HTML."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        image.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def interactive_viewer_html(image: Image.Image, height: int = 640) -> str:
    """
    Собрать самодостаточный HTML-вьюер с zoom/pan и minimap-навигатором.

    Никаких внешних библиотек и CDN — чистый inline JS/CSS, поэтому работает
    офлайн и на "чистой" машине. Картинка вставляется как data:URI, увеличение
    и панорамирование считаются на стороне браузера (исходник не перегружается).

    Управление: колесо — зум к курсору, перетаскивание — панорама,
    двойной клик — сброс, клик по minimap — переход к точке.
    """
    w, h = image.size
    src = _to_data_uri(image)

    template = r"""
<style>
  #ov-root{position:relative;width:100%;height:__ROOTH__px;overflow:hidden;
    background:#111318;border:1px solid #333;border-radius:8px;
    cursor:grab;touch-action:none;user-select:none;}
  #ov-root.grabbing{cursor:grabbing;}
  #ov-img{position:absolute;top:0;left:0;transform-origin:0 0;
    will-change:transform;image-rendering:auto;pointer-events:none;}
  #ov-bar{position:absolute;top:8px;left:8px;z-index:5;display:flex;gap:6px;
    align-items:center;background:rgba(20,22,28,.82);border:1px solid #3a3d44;
    border-radius:6px;padding:4px 6px;font:13px/1.2 system-ui,sans-serif;color:#e7e9ee;}
  #ov-bar button{width:28px;height:26px;border:1px solid #4a4d55;border-radius:5px;
    background:#22252c;color:#e7e9ee;font-size:15px;cursor:pointer;}
  #ov-bar button:hover{background:#2d313a;}
  #ov-pct{min-width:52px;text-align:center;font-variant-numeric:tabular-nums;}
  #ov-mini{position:absolute;right:8px;bottom:8px;z-index:5;border:1px solid #4a4d55;
    border-radius:5px;overflow:hidden;background:#000;box-shadow:0 2px 8px rgba(0,0,0,.5);
    cursor:pointer;}
  #ov-mini img{display:block;}
  #ov-vp{position:absolute;border:2px solid #ffcf33;background:rgba(255,207,51,.15);
    box-sizing:border-box;pointer-events:none;}
  #ov-hint{position:absolute;left:8px;bottom:8px;z-index:5;
    background:rgba(20,22,28,.72);border-radius:5px;padding:3px 8px;
    font:12px system-ui,sans-serif;color:#aeb2bb;}
</style>
<div id="ov-root">
  <img id="ov-img" src="__SRC__" draggable="false"/>
  <div id="ov-bar">
    <button id="ov-out" title="Отдалить">&minus;</button>
    <span id="ov-pct">100%</span>
    <button id="ov-in" title="Приблизить">+</button>
    <button id="ov-fit" title="Вписать в экран">⤢</button>
    <button id="ov-1to1" title="Масштаб 1:1 (нативные пиксели)">1:1</button>
  </div>
  <div id="ov-mini"><img id="ov-mini-img" src="__SRC__"/><div id="ov-vp"></div></div>
  <div id="ov-hint">Колесо — зум · тянуть — панорама · двойной клик — сброс</div>
</div>
<script>
(function(){
  var IMG_W=__W__, IMG_H=__H__;
  var MINI_W=180;
  var root=document.getElementById('ov-root');
  var img=document.getElementById('ov-img');
  var vp=document.getElementById('ov-vp');
  var mini=document.getElementById('ov-mini');
  var miniImg=document.getElementById('ov-mini-img');
  var pct=document.getElementById('ov-pct');
  var s=1, tx=0, ty=0, fit=1, mm=MINI_W/IMG_W;
  var miniH=Math.round(IMG_H*mm);
  miniImg.style.width=MINI_W+'px'; miniImg.style.height=miniH+'px';

  function cw(){return root.clientWidth;}
  function ch(){return root.clientHeight;}
  function clamp(v,a,b){return v<a?a:(v>b?b:v);}

  function apply(){
    // не даём картинке улететь совсем за пределы окна
    var vw=cw(), vh=ch();
    var minTx=Math.min(0, vw-IMG_W*s), minTy=Math.min(0, vh-IMG_H*s);
    tx=clamp(tx, minTx, 0>vw-IMG_W*s?0:Math.max(0,vw-IMG_W*s));
    ty=clamp(ty, minTy, 0>vh-IMG_H*s?0:Math.max(0,vh-IMG_H*s));
    if(IMG_W*s<=vw) tx=(vw-IMG_W*s)/2;
    if(IMG_H*s<=vh) ty=(vh-IMG_H*s)/2;
    img.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')';
    img.style.width=IMG_W+'px'; img.style.height=IMG_H+'px';
    pct.textContent=Math.round(s/fit*100)+'%';
    // прямоугольник видимой зоны на minimap
    var x0=(-tx)/s, y0=(-ty)/s;
    var vwImg=vw/s, vhImg=vh/s;
    vp.style.left=clamp(x0*mm,0,MINI_W)+'px';
    vp.style.top=clamp(y0*mm,0,miniH)+'px';
    vp.style.width=clamp(vwImg*mm,0,MINI_W)+'px';
    vp.style.height=clamp(vhImg*mm,0,miniH)+'px';
  }
  function doFit(){
    fit=Math.min(cw()/IMG_W, ch()/IMG_H);
    s=fit; tx=(cw()-IMG_W*s)/2; ty=(ch()-IMG_H*s)/2; apply();
  }
  function zoomAt(cx, cy, factor){
    var ns=clamp(s*factor, fit*0.9, fit*40);
    var ix=(cx-tx)/s, iy=(cy-ty)/s;
    s=ns; tx=cx-ix*s; ty=cy-iy*s; apply();
  }
  function setScaleAt(cx, cy, target){
    // Абсолютный масштаб (не привязанный к "fit"), чтобы «1:1» всегда достижим.
    var ns=clamp(target, 0.02, 64);
    var ix=(cx-tx)/s, iy=(cy-ty)/s;
    s=ns; tx=cx-ix*s; ty=cy-iy*s; apply();
  }
  root.addEventListener('wheel', function(e){
    e.preventDefault();
    var r=root.getBoundingClientRect();
    zoomAt(e.clientX-r.left, e.clientY-r.top, e.deltaY<0?1.15:1/1.15);
  }, {passive:false});

  var drag=false, px=0, py=0;
  root.addEventListener('pointerdown', function(e){
    if(e.target===mini||e.target===miniImg||e.target===vp) return;
    drag=true; px=e.clientX; py=e.clientY; root.classList.add('grabbing');
    root.setPointerCapture(e.pointerId);
  });
  root.addEventListener('pointermove', function(e){
    if(!drag) return;
    tx+=e.clientX-px; ty+=e.clientY-py; px=e.clientX; py=e.clientY; apply();
  });
  function endDrag(){drag=false; root.classList.remove('grabbing');}
  root.addEventListener('pointerup', endDrag);
  root.addEventListener('pointercancel', endDrag);
  root.addEventListener('dblclick', function(){doFit();});

  document.getElementById('ov-in').onclick=function(){zoomAt(cw()/2,ch()/2,1.4);};
  document.getElementById('ov-out').onclick=function(){zoomAt(cw()/2,ch()/2,1/1.4);};
  document.getElementById('ov-fit').onclick=function(){doFit();};
  document.getElementById('ov-1to1').onclick=function(){setScaleAt(cw()/2,ch()/2,1.0);};

  // клик по minimap — центрируем окно на выбранной точке
  mini.addEventListener('click', function(e){
    var r=mini.getBoundingClientRect();
    var ix=(e.clientX-r.left)/mm, iy=(e.clientY-r.top)/mm;
    tx=cw()/2-ix*s; ty=ch()/2-iy*s; apply();
  });

  window.addEventListener('resize', function(){doFit();});
  if(img.complete) doFit(); else img.onload=doFit;
})();
</script>
"""
    return (
        template
        .replace("__SRC__", src)
        .replace("__ROOTH__", str(int(height)))  # высота окна вьюера (CSS)
        .replace("__W__", str(w))
        .replace("__H__", str(h))                # натуральный размер картинки (JS)
    )


# --- Интерактивный выбор области мышью (без ползунков по X/Y) ---------------
# Самодостаточный компонент (vanilla JS, без npm/React) — тот же подход, что и
# interactive_viewer_html: рисуем свой HTML/JS-протокол Streamlit Components
# вручную, никаких внешних зависимостей и CDN. Объявляется один раз на процесс.
_REGION_PICKER_DIR = Path(__file__).resolve().parent / "region_picker_frontend"
_region_picker_component = components.declare_component(
    "orevision_region_picker", path=str(_REGION_PICKER_DIR)
)


def region_picker(
    image: Image.Image,
    key: str,
    bbox: tuple[float, float, float, float] = (0.30, 0.30, 0.60, 0.60),
    color: str = "rgba(255, 207, 51, .28)",
    border_color: str = "#ffcf33",
) -> tuple[float, float, float, float]:
    """
    Интерактивное выделение прямоугольной области на изображении: тянуть,
    двигать, менять размер за уголки/края мышью — как выделение области на
    рабочем столе (без отдельных ползунков по X и по Y).

    bbox — начальные координаты (x0, y0, x1, y1) в долях 0..1, используются
    только при первой отрисовке (дальше геолог управляет сам).
    Возвращает текущие (x0, y0, x1, y1) в долях 0..1 (x0<x1, y0<y1).

    Если JS-компонент по какой-то причине не пришлёт значение (например,
    редкая несовместимость браузера), функция просто продолжит возвращать
    bbox — сбоя не будет, только область останется на месте по умолчанию.
    """
    src = _to_data_uri(image)
    default = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}
    value = _region_picker_component(
        src=src,
        bbox=list(bbox),
        color=color,
        border_color=border_color,
        key=key,
        default=default,
    )
    if not isinstance(value, dict):
        return bbox
    try:
        x0, y0, x1, y1 = (
            float(value["x0"]), float(value["y0"]),
            float(value["x1"]), float(value["y1"]),
        )
    except (KeyError, TypeError, ValueError):
        return bbox
    if x1 <= x0 or y1 <= y0:
        return bbox
    return (x0, y0, x1, y1)
