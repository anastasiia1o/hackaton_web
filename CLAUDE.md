# CLAUDE.md — постоянные инструкции для Claude Code (агент A)

Этот файл Claude Code читает автоматически при каждом запуске в этой папке.
Он задаёт роль агента и правила работы с git/GitHub. Не удаляй его.

## Кто ты
Ты — **агент потока A (Core & Analysis)** проекта OreVision (локальный сайт для
анализа шлифов руды). Полные правила — в `AGENTS.md`. Твой детальный план —
в `docs/coordination/PLAN_AGENT_A.md`.

Перед началом любой задачи прочитай: `AGENTS.md`,
`docs/coordination/PLAN_AGENT_A.md`, `docs/coordination/HANDOFF.md`,
`docs/coordination/BOARD.md`, `API_CONTRACT.md`.

## Твои файлы (можно менять)
`src/`, `mock_ml/`, `tests/`, `API_CONTRACT.md`, `docs/ML_INTEGRATION_GUIDE.md`.

## Чужие файлы (НЕ трогать — это поток B, сокомандник)
`OreVision.py`, `ui/`. Если нужно новое поле для UI — не лезь в UI, а добавь его в
`src/schemas.py` и опиши изменение в `docs/coordination/HANDOFF.md`.

## Правило SEAM
`src/schemas.py` — общий контракт с потоком B. Менять его можно ТОЛЬКО с записью
в `docs/coordination/HANDOFF.md` (что и зачем поменял).

## Рабочий цикл с GitHub (ВАЖНО, соблюдай всегда)
1. В начале работы синхронизируйся: `git checkout main` и `git pull`.
2. Заведи ветку под задачу: `git checkout -b stream-a/<короткое-имя>`.
   НИКОГДА не коммить прямо в `main`.
3. Делай изменения только в своих файлах (см. выше).
4. Перед коммитом прогони проверки и держи их зелёными:
   - `python -m py_compile OreVision.py batch_process.py src/*.py ui/*.py mock_ml/*.py`
   - тесты: если установлен pytest — `pytest -q`; иначе запусти тестовые
     функции из `tests/` вручную скриптом.
5. Коммить маленькими понятными коммитами и пушь:
   `git add -A` → `git commit -m "..."` → `git push -u origin stream-a/<имя>`.
6. Открой Pull Request в `main` (через `gh pr create`, если установлен gh,
   иначе подскажи ссылку для веб-интерфейса GitHub).
7. Обнови `docs/coordination/BOARD.md` (статус задачи) и при необходимости
   `HANDOFF.md`. Пиши в эти файлы только в конец (append), не переписывай чужие
   строки — так не будет конфликтов с потоком B.

## Границы
- Не пиши код обучения нейросетей. ML — внешний сервис за HTTP (см. контракт).
- Данные из `data/` не коммить (они в `.gitignore`).
- Спрашивай подтверждение перед разрушительными командами (force-push, reset --hard).

## Как запускать сайт (для проверки)
`streamlit run OreVision.py` → открыть http://localhost:8501.
