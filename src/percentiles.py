"""Перцентилі гравців всередині групи порівняння.

Перцентиль = яку частку гравців тієї ж ролі цей гравець випереджає за метрикою.
80 за npxG серед нападників = більше npxG на 90, ніж у ~80% нападників турніру.

Навіщо перцентилі, а не сирі значення на 90:
  - метрики мають різні шкали (0.3 xG проти 45 пасів) — перцентиль зводить усе до 0–100,
    тому їх можна показати на одному радарі;
  - "добре" для центрбека і для вінгера — різні числа, тому рахуємо всередині ролі.
"""
import unicodedata

import pandas as pd

from src.data_loader import PROJECT_ROOT

PER90_PATH = PROJECT_ROOT / "data" / "processed" / "player_per90.parquet"

# 8 позиційних груп -> 6 груп порівняння (так само ділять скаутські звіти FBref).
# Причина: з 270+ хв у CM лише 8 гравців, в AM — 15. На 8 гравцях перцентиль
# стрибає кроками по 12.5 і нічого не означає. DM і CM, AM і W виконують схожу
# роботу, тож зливаємо їх у пули по ~35 гравців.
COMPARISON_POOLS = {
    "GK": "GK",
    "CB": "CB",
    "FB": "FB",
    "DM": "MF", "CM": "MF",
    "AM": "AM/W", "W": "AM/W",
    "FW": "FW",
}

POOL_NAMES = {
    "GK": "goalkeepers",
    "CB": "centre-backs",
    "FB": "full-backs",
    "MF": "central midfielders",
    "AM/W": "attacking midfielders & wingers",
    "FW": "forwards",
}

# Метрика -> підпис на графіку. Підписи англійською: портфоліо для міжнародної аудиторії.
LABELS = {
    "np_goals": "Non-penalty goals",
    "npxg": "npxG",
    "np_shots": "Shots",
    "npxg_per_shot": "npxG per shot",
    "assists": "Assists",
    "xa": "xA",
    "key_passes": "Key passes",
    "passes": "Passes",
    "pass_completion": "Pass completion",
    "progressive_passes": "Progressive passes",
    "passes_final_third": "Passes into final third",
    "passes_into_box": "Passes into box",
    "progressive_carries": "Progressive carries",
    "carries_into_box": "Carries into box",
    "dribbles_completed": "Successful dribbles",
    "tackles_won": "Tackles won",
    "interceptions": "Interceptions",
    "ball_recoveries": "Ball recoveries",
    "pressures": "Pressures",
    "clearances": "Clearances",
    "aerials_won": "Aerials won",
}

# Які 10 метрик показувати на радарі кожної ролі.
# Порядок: атака -> робота з м'ячем -> оборона, щоб блоки стояли поруч на колі.
# Навмисно не беремо dribble_success: 1 з 1 = 100%, на малій вибірці це шум.
# Воротарів немає: спецметрик для них ми не рахували (відоме обмеження етапу 2).
RADAR_TEMPLATES = {
    "FW": ["np_goals", "npxg", "np_shots", "npxg_per_shot", "xa", "key_passes",
           "dribbles_completed", "carries_into_box", "aerials_won", "pressures"],
    "AM/W": ["np_goals", "npxg", "np_shots", "xa", "key_passes", "passes_into_box",
             "progressive_carries", "carries_into_box", "dribbles_completed", "pressures"],
    "MF": ["key_passes", "passes", "pass_completion", "progressive_passes", "passes_final_third",
           "progressive_carries", "tackles_won", "interceptions", "ball_recoveries", "pressures"],
    "FB": ["xa", "passes_into_box", "progressive_passes", "pass_completion", "progressive_carries",
           "dribbles_completed", "tackles_won", "interceptions", "ball_recoveries", "pressures"],
    "CB": ["passes", "pass_completion", "progressive_passes", "progressive_carries", "tackles_won",
           "interceptions", "ball_recoveries", "pressures", "clearances", "aerials_won"],
}

ID_COLS = ["player_id", "player", "team", "minutes", "position_group"]


def load_per90() -> pd.DataFrame:
    """Таблиця з src.metrics: гравці з 270+ хв, лічильники на 90."""
    return pd.read_parquet(PER90_PATH)


def percentile_table(per90: pd.DataFrame) -> pd.DataFrame:
    """Перцентилі (0–100) усіх метрик для кожного гравця всередині його пулу.

    Повертає ту саму кількість рядків, що й per90: ID-колонки + pool + метрики,
    але замість значень на 90 — перцентилі.
    """
    df = per90.copy()
    df["pool"] = df["position_group"].map(COMPARISON_POOLS)
    metrics = list(LABELS)

    # groupby("pool") + rank: ранг рахується ОКРЕМО всередині кожного пулу.
    # pct=True ділить ранг на кількість гравців у пулі -> частка від 0 до 1.
    # method="average": однакові значення отримують однаковий (середній) ранг.
    # Напр., 40 з 49 центрбеків без голів — усі 40 отримують ~42, а не 2..82
    # залежно від випадкового порядку рядків.
    # NaN (напр., npxG за удар без ударів) лишається NaN — "немає даних", а не 0.
    pct = df.groupby("pool")[metrics].rank(pct=True, method="average") * 100

    return df[ID_COLS + ["pool"]].join(pct)


def normalize_name(text: str) -> str:
    """'Mbappé' -> 'mbappe': прибираємо діакритику і регістр для пошуку."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def find_player(df: pd.DataFrame, query: str) -> pd.Series:
    """Знаходить одного гравця за частиною імені ('kane', 'mbappe', 'lamine')."""
    mask = df["player"].map(normalize_name).str.contains(normalize_name(query), regex=False)
    found = df[mask]
    if found.empty:
        raise ValueError(f"'{query}' не знайдено серед {len(df)} гравців таблиці")
    if len(found) > 1:
        names = ", ".join(found["player"])
        raise ValueError(f"'{query}' підходить кільком гравцям: {names}. Уточни запит.")
    return found.iloc[0]


if __name__ == "__main__":
    # Запуск з кореня проєкту:  python -m src.percentiles
    pct = percentile_table(load_per90())

    print("Розмір пулів:")
    print(pct["pool"].value_counts().to_string(), "\n")

    # Перевірка здоровим глуздом: хто лідер свого пулу за ключовою метрикою
    for pool, metric in [("FW", "npxg"), ("AM/W", "xa"), ("MF", "progressive_passes"),
                         ("FB", "progressive_carries"), ("CB", "progressive_passes")]:
        top = pct[pct["pool"] == pool].nlargest(3, metric)
        print(f"{pool:5} топ-3 за {metric}: " + ", ".join(
            f"{r.player} ({getattr(r, metric):.0f})" for r in top.itertuples()))
