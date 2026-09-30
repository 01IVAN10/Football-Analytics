"""Шляхи до файлів проєкту — в одному місці.

Навіщо окремий модуль: раніше PROJECT_ROOT жив у data_loader.py, а той на
першому ж рядку імпортує statsbombpy (клієнт API). Через це і радар, і стиль,
і будь-який модуль, якому потрібен лише шлях до файлу, тягнули за собою
API-клієнт. Веб-застосунку API не потрібен: він читає готові файли з data/app.
Тепер шляхи не залежать ні від чого, крім стандартної бібліотеки.
"""
from pathlib import Path

# Корінь проєкту = папка на рівень вище за src/ (працює звідки б не запускали)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW = PROJECT_ROOT / "data" / "raw"              # сирі дані StatsBomb (не в git)
PROCESSED = PROJECT_ROOT / "data" / "processed"  # таблиці метрик (не в git)
APP_DATA = PROJECT_ROOT / "data" / "app"         # полегшені дані для застосунку (у git)
FIGURES = PROJECT_ROOT / "reports" / "figures"   # PNG з CLI (не в git)
