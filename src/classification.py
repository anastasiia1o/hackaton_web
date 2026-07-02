"""
RULE-BASED КЛАССИФИКАЦИЯ РУДЫ — сердце "геологической логики".

Это НАМЕРЕННО простой, читаемый геологом код, а не нейросеть.
Геолог должен уметь прочитать этот файл и согласиться с логикой.

Правила (из постановки задачи), в порядке приоритета:

  1. Валидная площадь = вся площадь − артефакты (класс 4). Считается в metrics.
  2. Если доля талька > 10% (от валидной площади) → «Оталькованная руда».
     Это ПРИОРИТЕТ над всем остальным.
  3. Иначе смотрим сульфиды (классы 1 и 2). Доля тонких = класс2 / (класс1+класс2):
       - преобладают обычные (класс 1) → «Рядовая руда»;
       - преобладают тонкие (класс 2) → «Труднообогатимая руда».
  4. Пограничные случаи (тальк 9–11%, низкая уверенность, много артефактов)
     → «Требуется экспертная проверка».

КРИТИЧНО: труднообогатимость определяется СТЕПЕНЬЮ ЗАМЕЩЕНИЯ (это уже заложено
в том, что модель отнесла пиксель к классу 2), а НЕ размером включения.
Сайт НЕ доклассифицирует по размеру — только агрегирует площади классов.
"""

from __future__ import annotations

from . import config
from .schemas import (
    Metrics,
    Classification,
    ORE_TALC,
    ORE_ORDINARY,
    ORE_REFRACTORY,
    ORE_REVIEW,
)


def classify(m: Metrics) -> Classification:
    """
    Применить геологические правила к посчитанным метрикам.
    Возвращает итоговый класс + человекочитаемое объяснение + след правил.

    Порядок проверок:
      0) вырожденные случаи (нет валидной площади, нет сульфидов) → проверка;
      4) пограничность (тальк 9–11%, низкая уверенность, много артефактов) → проверка;
      2) тальк > 10% → оталькованная (приоритет);
      3) преобладание тонких/обычных → трудная/рядовая; ничья 50/50 → проверка.
    """
    trace: list[str] = []

    talc_pct = m.talc_fraction * 100
    fine_of_sulph_pct = m.fine_of_sulphides * 100
    artifact_pct = m.artifact_fraction * 100
    conf_pct = m.mean_confidence * 100

    sulphide_px = (
        m.class_area_px.get(config.CLASS_ORDINARY, 0)
        + m.class_area_px.get(config.CLASS_FINE, 0)
    )

    # --- ПРАВИЛО 0a: нет валидной площади (всё — артефакты/пусто) ----------
    if m.valid_px <= 0:
        trace.append("Валидная площадь равна нулю (изображение — сплошь артефакты/пустое).")
        return Classification(
            ore_class=ORE_REVIEW,
            reason=(
                "Классификация невозможна: после исключения артефактов не осталось "
                f"валидной площади (артефакты — {artifact_pct:.1f}% изображения). "
                "Требуется другой снимок или ручная проверка."
            ),
            needs_review=True,
            rule_trace=trace,
        )

    # --- Флаги "пограничности" (собираем, решаем в конце) ------------------
    borderline_talc = (
        config.TALC_BORDERLINE_LOW < m.talc_fraction < config.TALC_BORDERLINE_HIGH
    )
    low_confidence = m.mean_confidence < config.LOW_CONFIDENCE_THRESHOLD
    too_many_artifacts = m.artifact_fraction > config.ARTIFACT_WARN_FRACTION

    review_reasons: list[str] = []
    if borderline_talc:
        review_reasons.append(
            f"доля талька {talc_pct:.1f}% находится в пограничной зоне "
            f"{config.TALC_BORDERLINE_LOW*100:.0f}–{config.TALC_BORDERLINE_HIGH*100:.0f}%"
        )
    if low_confidence:
        review_reasons.append(
            f"низкая средняя уверенность модели ({conf_pct:.1f}%)"
        )
    if too_many_artifacts:
        review_reasons.append(
            f"много артефактов на изображении ({artifact_pct:.1f}% площади)"
        )

    # --- ПРАВИЛО 4 (приоритетно перекрывает автоматический вывод) ----------
    # Если ситуация пограничная — честно просим геолога проверить.
    if review_reasons:
        trace.append("Сработало правило экспертной проверки.")
        reason = (
            "Автоматическая классификация ненадёжна: "
            + "; ".join(review_reasons)
            + ". Рекомендуется ручная проверка геологом."
        )
        return Classification(
            ore_class=ORE_REVIEW,
            reason=reason,
            needs_review=True,
            rule_trace=trace,
        )

    # --- ПРАВИЛО 2: приоритет талька --------------------------------------
    trace.append(
        f"Проверка талька: {talc_pct:.1f}% vs порог "
        f"{config.TALC_THRESHOLD*100:.0f}%."
    )
    if m.talc_fraction > config.TALC_THRESHOLD:
        trace.append("Тальк превышает порог → Оталькованная руда (приоритет).")
        reason = (
            f"Руда классифицирована как оталькованная. "
            f"Содержание талька {talc_pct:.1f}% от валидной площади превышает "
            f"порог {config.TALC_THRESHOLD*100:.0f}%. "
            f"Тонкие срастания составляют {fine_of_sulph_pct:.1f}% "
            f"площади всех сульфидов."
        )
        return Classification(
            ore_class=ORE_TALC,
            reason=reason,
            needs_review=False,
            rule_trace=trace,
        )

    # --- ПРАВИЛО 0b: сульфидов почти нет — тип срастаний определить нельзя --
    if m.sulphide_fraction < config.MIN_SULPHIDE_FRACTION or sulphide_px == 0:
        trace.append("Сульфиды практически отсутствуют — преобладание определить нельзя.")
        return Classification(
            ore_class=ORE_REVIEW,
            reason=(
                "Сульфидные срастания на изображении почти не обнаружены "
                f"({m.sulphide_fraction*100:.1f}% валидной площади), "
                "поэтому тип срастаний определить нельзя. Требуется экспертная проверка."
            ),
            needs_review=True,
            rule_trace=trace,
        )

    # --- ПРАВИЛО 3: тип срастаний ------------------------------------------
    trace.append(
        f"Тальк в норме. Доля тонких среди сульфидов: {fine_of_sulph_pct:.1f}%."
    )

    # Ничья 50/50 (в пределах TIE_MARGIN): явного преобладания нет → проверка.
    if abs(m.fine_of_sulphides - 0.5) < config.TIE_MARGIN:
        trace.append("Обычные и тонкие срастания примерно поровну (ничья) → проверка.")
        return Classification(
            ore_class=ORE_REVIEW,
            reason=(
                f"Обычные и тонкие срастания представлены примерно поровну "
                f"(тонких {fine_of_sulph_pct:.1f}% среди сульфидов), явного преобладания нет. "
                "Требуется экспертная проверка."
            ),
            needs_review=True,
            rule_trace=trace,
        )

    if m.fine_of_sulphides > 0.5:
        trace.append("Преобладают тонкие срастания → Труднообогатимая руда.")
        reason = (
            f"Руда классифицирована как труднообогатимая. "
            f"Содержание талька {talc_pct:.1f}% не превышает порог "
            f"{config.TALC_THRESHOLD*100:.0f}%. "
            f"Преобладают тонкие срастания: {fine_of_sulph_pct:.1f}% "
            f"площади всех сульфидов."
        )
        ore = ORE_REFRACTORY
    else:
        trace.append("Преобладают обычные срастания → Рядовая руда.")
        reason = (
            f"Руда классифицирована как рядовая. "
            f"Содержание талька {talc_pct:.1f}% не превышает порог "
            f"{config.TALC_THRESHOLD*100:.0f}%. "
            f"Преобладают обычные срастания: "
            f"{(100 - fine_of_sulph_pct):.1f}% площади всех сульфидов."
        )
        ore = ORE_ORDINARY

    return Classification(
        ore_class=ore,
        reason=reason,
        needs_review=False,
        rule_trace=trace,
    )
