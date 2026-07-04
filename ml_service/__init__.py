"""orevision-ml — HTTP-сервис реальной модели классификации сортов руды.

Реализует сторону ML из API_CONTRACT.md (POST /analyze, GET /health) поверх
модели grade_unfreeze_best.pth. Сайт остаётся неизменным: переключение mock->real
делается одной настройкой OREVISION_ML_MODE=real.
"""
