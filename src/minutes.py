"""Зіграні хвилини та основні позиції гравців.

Хвилини рахуємо з подій:
  вийшов на поле  -> Starting XI (0-ва хвилина) або Substitution (як заміна);
  пішов з поля    -> Substitution (його замінили), червона/друга жовта
                     або фінальний свисток.
Хвилини = час виходу з поля - час виходу на поле.
"""
import pandas as pd

# Годинник StatsBomb у кожному таймі стартує з фіксованої хвилини
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105}

RED_CARDS = ["Red Card", "Second Yellow"]

# 25 позицій StatsBomb -> 8 груп для порівняння гравців.
# Порівнювати центрального захисника з вінгером за однаковими метриками немає сенсу,
# тому перцентилі й пошук схожих гравців рахуватимемо всередині групи.
POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "AM", "Left Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Midfield": "W", "Right Midfield": "W", "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "FW", "Left Center Forward": "FW", "Right Center Forward": "FW",
    "Secondary Striker": "FW",
}


def add_elapsed(events: pd.DataFrame) -> pd.DataFrame:
    """Додає колонку elapsed — реальний час від початку матчу в хвилинах.

    Навіщо: поле minute у StatsBomb "скидається" на початку кожного тайму.
    Компенсація 1-го тайму йде як 45, 46, 47..., а 2-й тайм знову стартує з 45.
    Тобто minute=46 може бути і в 1-му, і в 2-му таймі. Тому рахуємо:
    elapsed = тривалість усіх попередніх таймів + час від початку поточного.
    Серію пенальті (period 5) відкидаємо — це не ігровий час.
    """
    ev = events[events["period"] < 5].copy()
    ev["in_period"] = ev["minute"] + ev["second"] / 60 - ev["period"].map(PERIOD_START)

    # Тривалість кожного тайму = час останньої події в ньому
    duration = ev.groupby(["match_id", "period"])["in_period"].max()
    # Зсув = сума тривалостей попередніх таймів (для 1-го тайму 0)
    offset = (duration.groupby(level="match_id").cumsum() - duration).rename("offset")

    ev = ev.join(offset, on=["match_id", "period"])
    ev["elapsed"] = ev["offset"] + ev["in_period"]
    return ev


def player_match_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Один рядок = один гравець в одному матчі: коли вийшов, коли пішов, скільки зіграв."""
    ev = add_elapsed(events)
    match_end = ev.groupby("match_id")["elapsed"].max().rename("match_end")

    # 1) Старт: розгортаємо склад з події Starting XI (поле tactics -> lineup)
    starters = [
        {"match_id": row.match_id, "team": row.team,
         "player_id": p["player"]["id"], "player": p["player"]["name"], "on": 0.0}
        for row in ev[ev["type"] == "Starting XI"].itertuples()
        for p in row.tactics["lineup"]
    ]

    subs = ev[ev["type"] == "Substitution"]

    # 2) Вихід на заміну: гравець із substitution_replacement виходить на поле
    subs_on = subs[["match_id", "team", "substitution_replacement_id",
                    "substitution_replacement", "elapsed"]]
    subs_on.columns = ["match_id", "team", "player_id", "player", "on"]

    on = pd.concat([pd.DataFrame(starters), subs_on], ignore_index=True)

    # 3) Уходи з поля: замінили або вилучили
    is_red = (ev["foul_committed_card"].isin(RED_CARDS)
              | ev["bad_behaviour_card"].isin(RED_CARDS))
    off = pd.concat([subs, ev[is_red]])[["match_id", "player_id", "elapsed"]]
    # Якщо подій уходу кілька (напр., вилучили вже після заміни) — беремо найранішу
    off = off.groupby(["match_id", "player_id"], as_index=False)["elapsed"].min()
    off = off.rename(columns={"elapsed": "off"})

    # 4) Зводимо: хто не пішов з поля — грав до фінального свистка
    pm = on.merge(off, on=["match_id", "player_id"], how="left")
    pm = pm.join(match_end, on="match_id")
    pm["off"] = pm["off"].fillna(pm["match_end"])
    pm["minutes"] = (pm["off"] - pm["on"]).clip(lower=0)
    pm["player_id"] = pm["player_id"].astype(int)
    return pm.drop(columns="match_end")


def main_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Основна позиція = позиція, на якій у гравця найбільше подій за турнір.

    Кожна подія StatsBomb має поле position — де гравець грав у той момент.
    Це враховує і зміни позицій по ходу матчу (Tactical Shift).
    """
    counts = (events.dropna(subset=["player_id", "position"])
              .groupby(["player_id", "position"]).size().rename("n").reset_index())
    # Сортуємо за кількістю подій і лишаємо перший (найчастіший) рядок кожного гравця
    main = counts.sort_values("n", ascending=False).drop_duplicates("player_id")
    main = main.rename(columns={"position": "main_position"})
    main["player_id"] = main["player_id"].astype(int)
    main["position_group"] = main["main_position"].map(POSITION_GROUPS)
    return main[["player_id", "main_position", "position_group"]]


def player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Підсумкова таблиця за турнір: один рядок = один гравець."""
    pm = player_match_minutes(events)
    totals = pm.groupby(["player_id", "player", "team"], as_index=False).agg(
        minutes=("minutes", "sum"),
        matches=("match_id", "nunique"),
    )
    result = totals.merge(main_positions(events), on="player_id", how="left")
    result["minutes"] = result["minutes"].round(1)
    return result.sort_values("minutes", ascending=False, ignore_index=True)


if __name__ == "__main__":
    # Запуск з кореня проєкту:  python -m src.minutes
    from src.data_loader import load_events, PROJECT_ROOT

    table = player_minutes(load_events())
    out = PROJECT_ROOT / "data" / "processed" / "player_minutes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    print(table.head(10).to_string())
    print(f"\nГравців: {len(table)}. Збережено: {out}")
