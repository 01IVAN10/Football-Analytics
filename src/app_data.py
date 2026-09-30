"""Полегшені дані для веб-застосунку: збірка (CLI) і читання.

Проблема: data/raw і data/processed не в git — це 21 МБ подій, які генеруються
скриптами. А застосунок у хмарі (Streamlit Community Cloud) бачить лише те,
що лежить у репозиторії.

Рішення: окремий набір data/app, який комітимо:
  per90.parquet   — гравці з 270+ хв, метрики на 90 (для радару, перцентилів, схожих);
  totals.parquet  — усі 493 гравці, сумарно (для карт і заголовків);
  events.parquet  — події лише з тими 12 колонками, які читають карти.
Події займають ~1 МБ на диску і ~75 МБ у пам'яті замість 21 МБ і ~590 МБ:
зі 113 колонок StatsBomb картам потрібні 12. Це важливо, бо безкоштовний
хостинг дає застосунку близько 1 ГБ пам'яті.

Запуск з кореня проєкту (після python -m src.metrics):
    python -m src.app_data
"""
import numpy as np
import pandas as pd

from src.metrics import classify_passes, non_penalty_shots
from src.paths import APP_DATA, PROCESSED

PER90_PATH = APP_DATA / "per90.parquet"
TOTALS_PATH = APP_DATA / "totals.parquet"
EVENTS_PATH = APP_DATA / "events.parquet"

# Колонки, які читають функції з src.pitch_maps (і src.metrics, яку вони викликають).
# Додамо нову карту, якій треба ще щось (напр., carry_end_location) — додаємо сюди
# і перезбираємо: python -m src.app_data.
EVENT_COLS = [
    "match_id", "player_id", "type", "period", "location",
    "pass_end_location", "pass_outcome", "pass_type", "pass_assisted_shot_id",
    "shot_type", "shot_outcome", "shot_statsbomb_xg",
]


# ---------- збірка ----------

def slim_events(events: pd.DataFrame) -> pd.DataFrame:
    """Лише події гравців, з координатами, без серії пенальті, і лише потрібні колонки.

    Рядки без координат (заміни, тактичні зміни, початок/кінець тайму) картам
    не потрібні: на полі їх не намалюєш. Серію пенальті (period 5) карти й так
    відкидають — прибираємо одразу.
    """
    keep = events["player_id"].notna() & (events["period"] < 5) & events["location"].notna()
    slim = events.loc[keep, EVENT_COLS].reset_index(drop=True)
    slim["player_id"] = slim["player_id"].astype(int)
    return slim


def check_events(slim: pd.DataFrame, totals: pd.DataFrame) -> None:
    """Перевірка: з полегшених подій виходять ті самі числа, що в таблиці метрик.

    Рахуємо тими самими функціями, що й карти (non_penalty_shots, classify_passes),
    і порівнюємо з player_totals для кожного гравця. Якщо колись приберемо з
    EVENT_COLS потрібну колонку або відфільтруємо зайві рядки — тут впаде.
    """
    shots = non_penalty_shots(slim)
    passes = slim[slim["type"] == "Pass"]
    flags = classify_passes(passes)

    from_slim = pd.DataFrame({
        "np_shots": shots.groupby("player_id").size(),
        "np_goals": (shots["shot_outcome"] == "Goal").groupby(shots["player_id"]).sum(),
        "npxg": shots.groupby("player_id")["shot_statsbomb_xg"].sum(),
        "progressive_passes": flags["progressive"].groupby(passes["player_id"]).sum(),
        "key_passes": flags["key"].groupby(passes["player_id"]).sum(),
    })
    expected = totals.set_index("player_id")[from_slim.columns]
    # reindex: гравці без жодного удару/пасу у from_slim відсутні — для них 0
    from_slim = from_slim.reindex(expected.index).fillna(0)

    # np.isclose, а не ==: npxG — сума дробових чисел, порядок додавання може
    # дати різницю в останньому знаку після коми
    mismatch = ~np.isclose(from_slim, expected).all(axis=1)
    if mismatch.any():
        raise AssertionError(f"Розбіжності з player_totals у {mismatch.sum()} гравців: "
                             f"{list(expected.index[mismatch][:5])}")


def build() -> None:
    """Збирає data/app з data/raw і data/processed."""
    from src.data_loader import load_events   # тягне statsbombpy — потрібен лише тут

    APP_DATA.mkdir(parents=True, exist_ok=True)
    per90 = pd.read_parquet(PROCESSED / "player_per90.parquet")
    totals = pd.read_parquet(PROCESSED / "player_totals.parquet")
    events = slim_events(load_events())

    check_events(events, totals)

    for df, path in [(per90, PER90_PATH), (totals, TOTALS_PATH), (events, EVENTS_PATH)]:
        df.to_parquet(path, index=False)
        memory = df.memory_usage(deep=True).sum() / 1e6
        print(f"{path.name:15} {len(df):>7} рядків · файл {path.stat().st_size / 1e6:.2f} МБ"
              f" · у пам'яті {memory:.1f} МБ")
    print(f"Перевірка пройдена: удари, голи, npxG, прогресивні й ключові паси "
          f"= player_totals для всіх {len(totals)} гравців")


# ---------- читання (для застосунку) ----------

def load_per90() -> pd.DataFrame:
    return pd.read_parquet(PER90_PATH)


def load_totals() -> pd.DataFrame:
    return pd.read_parquet(TOTALS_PATH)


def load_events() -> pd.DataFrame:
    return pd.read_parquet(EVENTS_PATH)


if __name__ == "__main__":
    build()
