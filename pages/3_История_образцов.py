"""
СТРАНИЦА «История образцов».

Читает общий журнал анализов data/results/analysis_log.jsonl (его пишет
reports.export_all — по одной строке на каждый обработанный образец) и
показывает историю: таблицу, распределение по классам, фильтр и выгрузку.

Только чтение готового лога.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="OreVision — История образцов", page_icon="📜", layout="wide")
config.ensure_dirs()

st.title("📜 История образцов")
st.caption(
    "Все запуски анализа (одиночные и пакетные) фиксируются в общий журнал — "
    "для воспроизводимости и аудита. Ниже — история, новые записи сверху."
)

LOG_PATH = config.RESULTS_DIR / "analysis_log.jsonl"


def _read_log(path) -> list[dict]:
    """Прочитать JSONL-лог; битые строки пропускаем, не падаем."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


records = _read_log(LOG_PATH)

if not records:
    st.info(
        "Журнал пока пуст. Запустите анализ на странице **Пакетная обработка** — "
        "и записи появятся здесь."
    )
    st.stop()

# Новые записи — сверху.
records = list(reversed(records))
df = pd.DataFrame(records)

# --- Сводные метрики --------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Всего анализов", len(df))
if "ore_class" in df:
    c2.metric("Уникальных классов", df["ore_class"].nunique())
if "ts" in df:
    c3.metric("Последний запуск", str(df["ts"].iloc[0]).replace("T", " "))

# --- Фильтр по классу -------------------------------------------------------
if "ore_class" in df:
    classes = ["(все)"] + sorted(df["ore_class"].dropna().unique().tolist())
    pick = st.selectbox("Фильтр по классу руды", classes)
    view = df if pick == "(все)" else df[df["ore_class"] == pick]
else:
    view = df

# --- Читаемая таблица (доли -> проценты) ------------------------------------
pretty = view.copy()
# Вложенные/служебные поля (напр. thresholds — словарь порогов) не показываем
# в основной таблице, чтобы она читалась; они остаются в выгрузке JSONL.
for drop_col in ["thresholds", "app_version"]:
    if drop_col in pretty:
        pretty = pretty.drop(columns=[drop_col])
for frac_col, pct_col in [("talc_fraction", "Тальк, %"),
                          ("fine_of_sulphides", "Тонкие/сульфиды, %")]:
    if frac_col in pretty:
        pretty[pct_col] = (pretty[frac_col] * 100).round(1)
        pretty = pretty.drop(columns=[frac_col])
if "needs_review" in pretty:
    pretty["needs_review"] = pretty["needs_review"].map(lambda v: "да" if v else "")

rename = {"ts": "Время", "image": "Изображение", "ore_class": "Класс руды",
          "model_version": "Модель", "needs_review": "Проверка",
          "inference_time_ms": "Время инференса, мс"}
pretty = pretty.rename(columns={k: v for k, v in rename.items() if k in pretty})

st.subheader(f"Записи ({len(view)})")
st.dataframe(pretty, use_container_width=True, hide_index=True)

# --- Распределение по классам ----------------------------------------------
if "ore_class" in df:
    st.subheader("Распределение по классам руды")
    st.bar_chart(df["ore_class"].value_counts())

# --- Выгрузка сырого лога ---------------------------------------------------
st.download_button(
    "Скачать журнал (JSONL)",
    data=LOG_PATH.read_bytes(),
    file_name="analysis_log.jsonl",
    mime="application/x-ndjson",
)
