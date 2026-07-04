"""
UI-компоненты Streamlit: легенда, карточка классификации, таблица метрик.

Вынесены отдельно, чтобы app.py оставался коротким и читаемым, и чтобы
поток B мог развивать визуал независимо от логики потока A.
"""

from __future__ import annotations

import streamlit as st

from src.schemas import AnalysisResult, ORE_TALC, ORE_ORDINARY, ORE_REFRACTORY
from src.metrics import as_percent_rows
from ui.viewer import legend_items

# Цвет карточки под каждый класс руды.
_ORE_COLOR = {
    ORE_TALC: "#1f5fe0",
    ORE_ORDINARY: "#1a9a3a",
    ORE_REFRACTORY: "#d41f1f",
}


def classification_card(result: AnalysisResult) -> None:
    """Крупная цветная карточка с итоговым классом и объяснением."""
    c = result.classification
    color = _ORE_COLOR.get(c.ore_class, "#8a6d00")  # жёлтый для "проверки"
    st.markdown(
        f"""
        <div style="border-left:8px solid {color};background:{color}14;
                    padding:14px 18px;border-radius:8px;margin:6px 0;">
          <div style="font-size:1.4rem;font-weight:700;color:{color};">
            {c.ore_class}
          </div>
          <div style="margin-top:6px;color:#222;">{c.reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if c.needs_review:
        st.warning("Модель не уверена в результате — рекомендуется ручная проверка геологом.")


def metrics_table(result: AnalysisResult) -> None:
    """Таблица количественных метрик."""
    st.dataframe(as_percent_rows(result.metrics), use_container_width=True, hide_index=True)


def legend_bar() -> None:
    """Горизонтальная легенда цветов классов."""
    chips = []
    for name, hexcol in legend_items():
        chips.append(
            f'<span style="display:inline-block;margin:2px 8px 2px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{hexcol};border-radius:2px;margin-right:5px;"></span>'
            f'{name}</span>'
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)
