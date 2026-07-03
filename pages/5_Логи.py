"""
СТРАНИЦА «Логи» (поток B).

Показывает журналы событий из src/event_log.py: импорт, пакетная обработка,
ошибки, сохранения разметки, экспорт для active learning. НЕ показывает и не
трогает data/results/analysis_log.jsonl (отдельный журнал анализов потока A —
см. страницу «История и лог»).

«Очистить логи» — двухшаговый безопасный сценарий: сначала выбор области
(текущий датасет / все логи приложения) и явное подтверждение, только потом
удаление. Функция очистки (src/event_log.clear_logs) физически не может
затронуть изображения/ROI/маски/экспорты/отчёты/модели/конфиги — она даже не
знает их путей, работает только с data/logs/*.jsonl.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, event_log as ev

st.set_page_config(page_title="OreVision — Логи", page_icon="🧾", layout="wide")
config.ensure_dirs()

st.title("🧾 Логи")
st.caption(
    "События импорта, пакетной обработки, ошибки, сохранения разметки и "
    "экспорт для active learning. Отдельно от «Истории и лога» (там — журнал "
    "самого ML-анализа)."
)

KIND_LABELS_RU = {
    ev.KIND_IMPORT: "Импорт",
    ev.KIND_ERROR: "Ошибка",
    ev.KIND_ANNOTATION_SAVE: "Сохранение разметки",
    ev.KIND_EXPORT: "Экспорт (active learning)",
    ev.KIND_BATCH: "Пакетная обработка",
}

events = ev.read_all_events()

if not events:
    st.info("Журналы пока пусты.")
else:
    df = pd.DataFrame(list(reversed(events)))
    if "kind" in df:
        df["Тип события"] = df["kind"].map(lambda k: KIND_LABELS_RU.get(k, k))

    c1, c2, c3 = st.columns(3)
    c1.metric("Всего записей", len(df))
    if "dataset_id" in df:
        c2.metric("Датасетов затронуто", df["dataset_id"].nunique())
    if "ts" in df and len(df):
        c3.metric("Последнее событие", str(df["ts"].iloc[0]).replace("T", " "))

    kinds = ["(все)"] + sorted({e.get("kind", "") for e in events})
    pick_kind = st.selectbox("Фильтр по типу события", kinds)
    view = df if pick_kind == "(все)" else df[df["kind"] == pick_kind]

    st.dataframe(view, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Очистить логи")
st.warning(
    "Это действие удаляет ТОЛЬКО записи журналов событий (data/logs/*.jsonl). "
    "Изображения, ROI, маски разметки, экспортированные датасеты, отчёты, "
    "модели и конфигурационные файлы никогда не затрагиваются."
)

st.session_state.setdefault("logs_clear_step", 0)

if st.session_state["logs_clear_step"] == 0:
    if st.button("🗑️ Очистить логи"):
        st.session_state["logs_clear_step"] = 1
        st.rerun()
else:
    st.info(
        "Подтверждение: будут удалены только записи логов (см. предупреждение "
        "выше). Изображения/ROI/маски/экспорты/отчёты/модели/конфиги удалены "
        "НЕ будут."
    )
    scope_label = st.radio(
        "Что очистить",
        options=["dataset", "all"],
        format_func=lambda s: (
            "Только логи текущего датасета" if s == "dataset" else "Все логи приложения"
        ),
    )
    dataset_id_to_clear = None
    if scope_label == "dataset":
        dataset_id_to_clear = st.text_input(
            "ID датасета", value=st.session_state.get("dataset_id", "default"),
        )

    confirmed = st.checkbox("Я понимаю, что будут удалены только записи логов, и подтверждаю удаление.")
    conf_col, cancel_col = st.columns([1, 1])
    with conf_col:
        do_delete = st.button("✅ Подтвердить и удалить", disabled=not confirmed, type="primary")
    with cancel_col:
        do_cancel = st.button("Отмена")

    if do_cancel:
        st.session_state["logs_clear_step"] = 0
        st.rerun()

    if do_delete:
        result = ev.clear_logs(
            scope_label, dataset_id=dataset_id_to_clear if scope_label == "dataset" else None,
        )
        st.session_state["logs_clear_step"] = 0
        scope_text = (
            f"датасета «{dataset_id_to_clear}»" if scope_label == "dataset" else "всего приложения"
        )
        st.success(
            f"Удалено записей логов: {result['total_deleted']} "
            f"(импорт/ошибки/разметка/экспорт: {result['app_deleted']}, "
            f"пакетная обработка: {result['batch_deleted']}). Область: {scope_text}."
        )
        st.rerun()
