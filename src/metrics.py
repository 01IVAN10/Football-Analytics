"""Метрики гравців за турнір: сумарні значення і значення на 90 хвилин.

Схема: події -> лічильники дій по гравцях (player_id) -> ділимо на хвилини -> * 90.
"""
import numpy as np
import pandas as pd

# Поле StatsBomb: 120 x 80 ярдів, атака завжди зліва направо, ворота суперника в (120, 40)
GOAL_X, GOAL_Y = 120, 40
BOX_X, BOX_Y_MIN, BOX_Y_MAX = 102, 18, 62     # штрафний майданчик суперника
FINAL_THIRD_X = 80

# Стандарти не рахуємо як "прогресію": кутовий завжди потрапляє в штрафний,
# але це не заслуга гравця, а тип розіграшу.
SET_PIECES = ["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"]

# Метрики-частки (відсотки, xG за удар) не діляться на хвилини
RATIO_COLS = ["pass_completion", "dribble_success", "npxg_per_shot"]

# Лічильники, які переводимо "на 90 хвилин"
COUNT_COLS = [
    "np_goals", "np_shots", "npxg",
    "assists", "key_passes", "xa",
    "passes", "passes_completed", "progressive_passes", "passes_final_third", "passes_into_box",
    "progressive_carries", "carries_into_box", "dribbles", "dribbles_completed",
    "tackles_won", "interceptions", "ball_recoveries", "pressures", "clearances", "aerials_won",
]


# ---------- допоміжні функції ----------

def split_xy(points: pd.Series) -> pd.DataFrame:
    """Колонку з парами [x, y] перетворює на дві колонки x і y."""
    xy = pd.DataFrame(points.tolist(), index=points.index)
    return xy.iloc[:, :2].set_axis(["x", "y"], axis=1)  # iloc: у кінця удару є ще z (висота)


def dist_to_goal(xy: pd.DataFrame) -> pd.Series:
    """Відстань від точки до центру воріт суперника (теорема Піфагора)."""
    return np.hypot(GOAL_X - xy["x"], GOAL_Y - xy["y"])


def in_box(xy: pd.DataFrame) -> pd.Series:
    return (xy["x"] >= BOX_X) & xy["y"].between(BOX_Y_MIN, BOX_Y_MAX)


def is_progressive(start: pd.DataFrame, end: pd.DataFrame) -> pd.Series:
    """Прогресивна дія: м'яч став щонайменше на 25% ближче до воріт.

    Відсоток, а не фіксовані метри, щоб пас з 60 до 40 ярдів від воріт
    і пас з 20 до 12 ярдів обидва вважались "просуванням".
    """
    return dist_to_goal(end) <= 0.75 * dist_to_goal(start)


def count(mask: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Скільки рядків, що задовольняють умову, у кожного гравця."""
    return events.loc[mask].groupby("player_id").size()


# ---------- метрики за блоками ----------

def shooting(ev: pd.DataFrame) -> pd.DataFrame:
    """Удари і голи без пенальті (пенальті — окрема навичка і спотворює xG)."""
    shots = ev[(ev["type"] == "Shot") & (ev["shot_type"] != "Penalty") & (ev["period"] < 5)]
    g = shots.groupby("player_id")
    return pd.DataFrame({
        "np_goals": g["shot_outcome"].apply(lambda s: (s == "Goal").sum()),
        "np_shots": g.size(),
        "npxg": g["shot_statsbomb_xg"].sum(),
    })


def creation(ev: pd.DataFrame) -> pd.DataFrame:
    """Створення моментів: гольові, ключові паси та xA."""
    passes = ev[ev["type"] == "Pass"]

    # xA (expected assists): xG удару, який став наслідком пасу гравця.
    # У пасу є pass_assisted_shot_id — id удару, до якого він привів.
    shots_xg = ev.loc[ev["type"] == "Shot"].set_index("id")["shot_statsbomb_xg"]
    assisting = passes.dropna(subset=["pass_assisted_shot_id"])
    xa = (assisting["pass_assisted_shot_id"].map(shots_xg)
          .groupby(assisting["player_id"]).sum())

    return pd.DataFrame({
        "assists": count(passes["pass_goal_assist"] == True, passes),
        "key_passes": count(passes["pass_assisted_shot_id"].notna(), passes),
        "xa": xa,
    })


def passing(ev: pd.DataFrame) -> pd.DataFrame:
    """Паси: обсяг, точність і просування м'яча вперед."""
    passes = ev[ev["type"] == "Pass"].copy()
    completed = passes["pass_outcome"].isna()        # NaN у StatsBomb = пас точний
    open_play = ~passes["pass_type"].isin(SET_PIECES)

    start = split_xy(passes["location"])
    end = split_xy(passes["pass_end_location"])

    good = completed & open_play                     # точні паси з гри
    return pd.DataFrame({
        "passes": passes.groupby("player_id").size(),
        "passes_completed": count(completed, passes),
        "progressive_passes": count(good & is_progressive(start, end), passes),
        "passes_final_third": count(good & (start["x"] < FINAL_THIRD_X)
                                    & (end["x"] >= FINAL_THIRD_X), passes),
        "passes_into_box": count(good & ~in_box(start) & in_box(end), passes),
    })


def carrying(ev: pd.DataFrame) -> pd.DataFrame:
    """Ведення м'яча і обводки."""
    carries = ev[ev["type"] == "Carry"]
    start = split_xy(carries["location"])
    end = split_xy(carries["carry_end_location"])
    # Мінімум 5 ярдів, щоб не рахувати дрібні "перекати" біля воріт
    long_enough = (dist_to_goal(start) - dist_to_goal(end)) >= 5

    dribbles = ev[ev["type"] == "Dribble"]
    return pd.DataFrame({
        "progressive_carries": count(is_progressive(start, end) & long_enough, carries),
        "carries_into_box": count(~in_box(start) & in_box(end), carries),
        "dribbles": dribbles.groupby("player_id").size(),
        "dribbles_completed": count(dribbles["dribble_outcome"] == "Complete", dribbles),
    })


def defending(ev: pd.DataFrame) -> pd.DataFrame:
    """Дії без м'яча."""
    tackles = ev[(ev["type"] == "Duel") & (ev["duel_type"] == "Tackle")]
    won = tackles["duel_outcome"].isin(["Won", "Success In Play", "Success Out"])
    recoveries = ev[ev["type"] == "Ball Recovery"]

    # Виграні верхові: StatsBomb ставить *_aerial_won у подію, якою гравець зіграв головою
    aerial_cols = ["clearance_aerial_won", "pass_aerial_won",
                   "shot_aerial_won", "miscontrol_aerial_won"]
    aerial_won = ev[aerial_cols].eq(True).any(axis=1)

    return pd.DataFrame({
        "tackles_won": count(won, tackles),
        "interceptions": count(ev["type"] == "Interception", ev),
        "ball_recoveries": count(recoveries["ball_recovery_recovery_failure"] != True, recoveries),
        "pressures": count(ev["type"] == "Pressure", ev),
        "clearances": count(ev["type"] == "Clearance", ev),
        "aerials_won": count(aerial_won, ev),
    })


# ---------- збірка ----------

def player_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Сумарні значення всіх метрик за турнір (один рядок = гравець)."""
    ev = events.dropna(subset=["player_id"]).copy()
    ev["player_id"] = ev["player_id"].astype(int)

    blocks = [shooting(ev), creation(ev), passing(ev), carrying(ev), defending(ev)]
    # axis=1 — склеюємо по колонках; індекс у всіх блоків = player_id.
    # Якщо в гравця немає жодної дії якогось типу, там NaN -> це 0.
    totals = pd.concat(blocks, axis=1).fillna(0)
    totals.index.name = "player_id"
    return totals


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Частки. Ділення на 0 дає NaN — і це чесно: 'немає даних', а не 0%."""
    df["pass_completion"] = df["passes_completed"] / df["passes"].replace(0, np.nan)
    df["dribble_success"] = df["dribbles_completed"] / df["dribbles"].replace(0, np.nan)
    df["npxg_per_shot"] = df["npxg"] / df["np_shots"].replace(0, np.nan)
    return df


def build_table(minutes: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Хвилини + позиції + сумарні метрики + частки в одній таблиці (всі гравці).

    minutes — таблиця з src.minutes.player_minutes.
    """
    df = minutes.merge(totals, left_on="player_id", right_index=True, how="left")
    df[totals.columns] = df[totals.columns].fillna(0)
    return add_ratios(df)


def per_90(table: pd.DataFrame, min_minutes: float = 270) -> pd.DataFrame:
    """Лишає гравців з min_minutes+ і перераховує лічильники на 90 хвилин."""
    df = table[table["minutes"] >= min_minutes].copy()
    # div(..., axis=0) ділить кожен РЯДОК на хвилини саме цього гравця
    df[COUNT_COLS] = df[COUNT_COLS].div(df["minutes"], axis=0) * 90
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Запуск з кореня проєкту:  python -m src.metrics
    from src.data_loader import load_events, PROJECT_ROOT
    from src.minutes import player_minutes

    events = load_events()
    table = build_table(player_minutes(events), player_totals(events))
    p90 = per_90(table)

    processed = PROJECT_ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed / "player_totals.parquet")
    p90.to_parquet(processed / "player_per90.parquet")

    cols = ["player", "team", "position_group", "minutes", "npxg", "xa", "progressive_passes"]
    print(p90.sort_values("npxg", ascending=False)[cols].head(10).round(2).to_string())
    print(f"\nГравців з ≥270 хв: {len(p90)}")
