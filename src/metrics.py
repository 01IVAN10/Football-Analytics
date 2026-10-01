"""Player metrics for the tournament: totals and values per 90 minutes.

Run from the project root (needs data/raw from src.data_loader):
    python -m src.metrics
"""
import numpy as np
import pandas as pd

# StatsBomb pitch: 120 x 80 yards, attacking left to right, opponent's goal at (120, 40)
GOAL_X, GOAL_Y = 120, 40
BOX_X, BOX_Y_MIN, BOX_Y_MAX = 102, 18, 62     # opponent's penalty area
FINAL_THIRD_X = 80

# Set pieces never count as progression: a corner always lands in the box, but that is
# the type of restart, not the player's merit.
SET_PIECES = ["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"]

# Per-90 threshold: below 3 full matches per-90 values are mostly noise
# (one shot in 60 minutes = 1.5 shots per 90). Shared by per_90 and the app.
MIN_MINUTES = 270

# Ratios are not divided by minutes
RATIO_COLS = ["pass_completion", "dribble_success", "npxg_per_shot"]

# Counting metrics converted to per 90
COUNT_COLS = [
    "np_goals", "np_shots", "npxg",
    "assists", "key_passes", "xa",
    "passes", "passes_completed", "progressive_passes", "passes_final_third", "passes_into_box",
    "progressive_carries", "carries_into_box", "dribbles", "dribbles_completed",
    "tackles_won", "interceptions", "ball_recoveries", "pressures", "clearances", "aerials_won",
]


# ---------- helpers ----------

def split_xy(points: pd.Series) -> pd.DataFrame:
    """Column of [x, y] pairs -> two columns x and y."""
    if points.empty:   # e.g. a player without a single shot
        return pd.DataFrame({"x": [], "y": []}, index=points.index, dtype=float)
    xy = pd.DataFrame(points.tolist(), index=points.index)
    return xy.iloc[:, :2].set_axis(["x", "y"], axis=1)  # shot end locations also have z


def dist_to_goal(xy: pd.DataFrame) -> pd.Series:
    """Distance from a point to the centre of the opponent's goal."""
    return np.hypot(GOAL_X - xy["x"], GOAL_Y - xy["y"])


def in_box(xy: pd.DataFrame) -> pd.Series:
    return (xy["x"] >= BOX_X) & xy["y"].between(BOX_Y_MIN, BOX_Y_MAX)


def is_progressive(start: pd.DataFrame, end: pd.DataFrame) -> pd.Series:
    """Progressive action: the ball ends up at least 25% closer to goal.

    Relative rather than a fixed distance, so that both 60 -> 40 yards and
    20 -> 12 yards from goal count as moving the ball forward.
    """
    return dist_to_goal(end) <= 0.75 * dist_to_goal(start)


def count(mask: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Number of rows matching the mask, per player."""
    return events.loc[mask].groupby("player_id").size()


# ---------- metric blocks ----------

def non_penalty_shots(ev: pd.DataFrame) -> pd.DataFrame:
    """Shots excluding penalties and the shootout (a separate skill that inflates xG).
    Shared by the metrics and the shot map, so the definition lives in one place."""
    return ev[(ev["type"] == "Shot") & (ev["shot_type"] != "Penalty") & (ev["period"] < 5)]


def shooting(ev: pd.DataFrame) -> pd.DataFrame:
    """Shots, goals and npxG, penalties excluded."""
    shots = non_penalty_shots(ev)
    g = shots.groupby("player_id")
    return pd.DataFrame({
        "np_goals": g["shot_outcome"].apply(lambda s: (s == "Goal").sum()),
        "np_shots": g.size(),
        "npxg": g["shot_statsbomb_xg"].sum(),
    })


def creation(ev: pd.DataFrame) -> pd.DataFrame:
    """Chance creation: assists, key passes and xA."""
    passes = ev[ev["type"] == "Pass"]

    # xA = xG of the shot that followed the player's pass
    # (the pass carries the shot's id in pass_assisted_shot_id)
    shots_xg = ev.loc[ev["type"] == "Shot"].set_index("id")["shot_statsbomb_xg"]
    assisting = passes.dropna(subset=["pass_assisted_shot_id"])
    xa = (assisting["pass_assisted_shot_id"].map(shots_xg)
          .groupby(assisting["player_id"]).sum())

    return pd.DataFrame({
        "assists": count(passes["pass_goal_assist"] == True, passes),
        "key_passes": count(passes["pass_assisted_shot_id"].notna(), passes),
        "xa": xa,
    })


def open_play(passes: pd.DataFrame) -> pd.Series:
    """True for open-play passes, False for set pieces."""
    return ~passes["pass_type"].isin(SET_PIECES)


def classify_passes(passes: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags per pass (same index): which metrics it counts towards.
    Shared by the metrics and the pass map."""
    completed = passes["pass_outcome"].isna()        # StatsBomb: no outcome = completed

    start = split_xy(passes["location"])
    end = split_xy(passes["pass_end_location"])

    good = completed & open_play(passes)             # completed open-play passes
    return pd.DataFrame({
        "completed": completed,
        "progressive": good & is_progressive(start, end),
        "final_third": good & (start["x"] < FINAL_THIRD_X) & (end["x"] >= FINAL_THIRD_X),
        "into_box": good & ~in_box(start) & in_box(end),
        "key": passes["pass_assisted_shot_id"].notna(),   # pass followed by a shot
    }, index=passes.index)


def passing(ev: pd.DataFrame) -> pd.DataFrame:
    """Passing: volume, accuracy, moving the ball forward."""
    passes = ev[ev["type"] == "Pass"]
    flags = classify_passes(passes)
    return pd.DataFrame({
        "passes": passes.groupby("player_id").size(),
        "passes_completed": count(flags["completed"], passes),
        "progressive_passes": count(flags["progressive"], passes),
        "passes_final_third": count(flags["final_third"], passes),
        "passes_into_box": count(flags["into_box"], passes),
    })


def carrying(ev: pd.DataFrame) -> pd.DataFrame:
    """Carries and take-ons."""
    carries = ev[ev["type"] == "Carry"]
    start = split_xy(carries["location"])
    end = split_xy(carries["carry_end_location"])
    # At least 5 yards, so that small touches near the goal do not count
    long_enough = (dist_to_goal(start) - dist_to_goal(end)) >= 5

    dribbles = ev[ev["type"] == "Dribble"]
    return pd.DataFrame({
        "progressive_carries": count(is_progressive(start, end) & long_enough, carries),
        "carries_into_box": count(~in_box(start) & in_box(end), carries),
        "dribbles": dribbles.groupby("player_id").size(),
        "dribbles_completed": count(dribbles["dribble_outcome"] == "Complete", dribbles),
    })


def defending(ev: pd.DataFrame) -> pd.DataFrame:
    """Defensive actions."""
    tackles = ev[(ev["type"] == "Duel") & (ev["duel_type"] == "Tackle")]
    won = tackles["duel_outcome"].isin(["Won", "Success In Play", "Success Out"])
    recoveries = ev[ev["type"] == "Ball Recovery"]

    # StatsBomb puts *_aerial_won on the event the player played with his head
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


# ---------- assembly ----------

def player_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Tournament totals of all metrics, one row per player."""
    ev = events.dropna(subset=["player_id"]).copy()
    ev["player_id"] = ev["player_id"].astype(int)

    blocks = [shooting(ev), creation(ev), passing(ev), carrying(ev), defending(ev)]
    # A player without any action of some type gets NaN there, which means 0
    totals = pd.concat(blocks, axis=1).fillna(0)
    totals.index.name = "player_id"
    return totals


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios. Division by zero gives NaN on purpose: no attempts is "no data", not 0%."""
    df["pass_completion"] = df["passes_completed"] / df["passes"].replace(0, np.nan)
    df["dribble_success"] = df["dribbles_completed"] / df["dribbles"].replace(0, np.nan)
    df["npxg_per_shot"] = df["npxg"] / df["np_shots"].replace(0, np.nan)
    return df


def build_table(minutes: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Minutes, positions, totals and ratios in one table (all players).

    minutes: the table from src.minutes.player_minutes.
    """
    df = minutes.merge(totals, left_on="player_id", right_index=True, how="left")
    df[totals.columns] = df[totals.columns].fillna(0)
    return add_ratios(df)


def per_90(table: pd.DataFrame, min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """Keep players with min_minutes+ and convert counting metrics to per 90."""
    df = table[table["minutes"] >= min_minutes].copy()
    df[COUNT_COLS] = df[COUNT_COLS].div(df["minutes"], axis=0) * 90
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_events
    from src.paths import PROCESSED
    from src.minutes import player_minutes

    events = load_events()
    table = build_table(player_minutes(events), player_totals(events))
    p90 = per_90(table)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_parquet(PROCESSED / "player_totals.parquet")
    p90.to_parquet(PROCESSED / "player_per90.parquet")

    cols = ["player", "team", "position_group", "minutes", "npxg", "xa", "progressive_passes"]
    print(p90.sort_values("npxg", ascending=False)[cols].head(10).round(2).to_string())
    print(f"\nPlayers with 270+ minutes: {len(p90)}")
