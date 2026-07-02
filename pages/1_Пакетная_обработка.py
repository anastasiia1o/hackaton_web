"""
СТРАНИЦА «Пакетная обработка» (поток B).

Запуск batch-обработки папки с изображениями прямо из UI, с прогресс-баром
и сводной таблицей. Это витрина CLI-скрипта batch_process.py: та же логика
(ML → метрики → классификация → экспорт CSV/JSON/PDF + запись в лог), но с
живым прогрессом и результатом на экране.

Логику потока A не трогаем — вызываем готовый run_analysis и reports.export_all.
Изображения грузим через viewer.load_display_image, чтобы гигапиксельные
панорамы не роняли приложение.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src import config, reports
from src.pipeline import run_analysis, load_mask
from ui import viewer
from batch_process import find_images

st.set_page_config(page_title="OreVision — Пакетная обработка", page_icon="🗂️", layout="wide")
config.ensure_dirs()
Image.MAX_IMAGE_PIXELS = None  # панорамы бывают гигапиксельными

st.title("🗂️ Пакетная обработка")
st.caption(
    "Обработать сразу целую папку образцов: для каждого изображения — анализ, "
    "классификация и экспорт отчётов. Результаты попадают в «Историю и лог»."
)

# --- Параметры запуска ------------------------------------------------------
default_folder = str(config.UPLOADS_DIR)
folder_str = st.text_input(
    "Папка с изображениями",
    value=default_folder,
    help="Абсолютный или относительный путь. По умолчанию — data/uploads.",
)

scenario = None
if config.ML_MODE == "mock":
    scenario = st.selectbox(
        "Демо-сценарий (mock применяется ко всем изображениям)",
        options=["refractory", "ordinary", "talc", "review"],
        format_func=lambda s: {
            "refractory": "Труднообогатимая (тонкие)",
            "ordinary": "Рядовая (обычные)",
            "talc": "Оталькованная (тальк >10%)",
            "review": "Пограничный (проверка)",
        }[s],
    )

folder = Path(folder_str.strip()) if folder_str.strip() else None
images = find_images(folder) if folder and folder.is_dir() else []

if folder and not folder.is_dir():
    st.error(f"Папка не найдена: {folder}")
elif folder:
    st.info(f"Найдено изображений: **{len(images)}** в `{folder}`")

run = st.button("▶️ Запустить обработку", type="primary", disabled=not images)

# --- Выполнение -------------------------------------------------------------
if run and images:
    params = {"scenario": scenario} if scenario else None
    progress = st.progress(0.0, text="Старт…")
    status = st.empty()
    rows: list[dict] = []
    errors: list[str] = []

    for i, img in enumerate(images, start=1):
        status.write(f"Обрабатываю **{img.name}** ({i}/{len(images)})…")
        try:
            result = run_analysis(str(img), params=params)
            # overlay для PDF — из уменьшенной картинки (безопасно для панорам)
            base = viewer.load_display_image(str(img))
            mask = load_mask(result.ml.mask_path)
            overlay = viewer.make_overlay(base, mask, opacity=0.55)
            overlay_png = config.RESULTS_DIR / img.stem / "overlay.png"
            overlay_png.parent.mkdir(parents=True, exist_ok=True)
            overlay.convert("RGB").save(overlay_png)

            reports.export_all(result, overlay_png=overlay_png)  # пишет и в лог
            rows.append({
                "Изображение": img.name,
                "Класс руды": result.classification.ore_class,
                "Тальк, %": round(result.metrics.talc_fraction * 100, 1),
                "Тонкие/сульфиды, %": round(result.metrics.fine_of_sulphides * 100, 1),
                "Проверка": "да" if result.classification.needs_review else "",
            })
        except Exception as e:  # noqa: BLE001
            errors.append(f"{img.name}: {e}")
        progress.progress(i / len(images), text=f"{i}/{len(images)}")

    status.empty()
    progress.empty()
    st.success(f"Готово: обработано {len(rows)} из {len(images)}.")

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Сводка")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Распределение по классам руды")
        st.bar_chart(df["Класс руды"].value_counts())

        st.download_button(
            "Скачать сводку (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_summary.csv",
            mime="text/csv",
        )
        st.caption(f"Отчёты по каждому образцу сохранены в `{config.RESULTS_DIR}`.")

    if errors:
        with st.expander(f"Ошибки ({len(errors)})"):
            for e in errors:
                st.error(e)
