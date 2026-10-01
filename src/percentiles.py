"""Percentiles within the comparison pool.

Percentile = the share of players in the same role this player is ahead of:
80 for npxG among forwards = more npxG per 90 than ~80% of the tournament's forwards.
Percentiles put metrics with different scales on one 0-100 radar, and "good" is
defined relative to the role: a good number of tackles differs for a CB and a winger.

Run from the project root:
    python -m src.percentiles
"""
import unicodedata

import pandas as pd

from src.paths import PROCESSED

PER90_PATH = PROCESSED / "player_per90.parquet"

# 8 position groups -> 6 comparison pools (the split FBref scouting reports use).
# With 270+ minutes there are only 8 CMs and 15 AMs: on 8 players a percentile moves
# in steps of 12.5 and means little. DM/CM and AM/W do similar jobs, so they are merged
# into pools of ~35 players.
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

# Metric -> chart label
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
    "dribbles": "Dribbles attempted",
    "dribbles_completed": "Successful dribbles",
    "tackles_won": "Tackles won",
    "interceptions": "Interceptions",
    "ball_recoveries": "Ball recoveries",
    "pressures": "Pressures",
    "clearances": "Clearances",
    "aerials_won": "Aerials won",
}

# The 10 radar metrics of each pool, ordered attack -> on the ball -> defending
# so that related metrics sit next to each other on the circle.
# No dribble_success: 1 of 1 = 100%, pure noise on small samples.
# No goalkeepers: there are no goalkeeping metrics.
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
    """The src.metrics table: players with 270+ minutes, per 90."""
    return pd.read_parquet(PER90_PATH)


def percentile_table(per90: pd.DataFrame) -> pd.DataFrame:
    """Percentiles (0-100) of all metrics, each player within his pool.

    Same rows as per90: ID columns + pool + metrics, with percentiles as values.
    """
    df = per90.copy()
    df["pool"] = df["position_group"].map(COMPARISON_POOLS)
    metrics = list(LABELS)

    # Ties share the average rank: 40 of 49 centre-backs without a goal all get ~42,
    # instead of 2..82 depending on row order.
    # NaN (e.g. npxG per shot without shots) stays NaN: no data, not zero.
    pct = df.groupby("pool")[metrics].rank(pct=True, method="average") * 100

    return df[ID_COLS + ["pool"]].join(pct)


def normalize_name(text: str) -> str:
    """'Mbappé' -> 'mbappe': strip accents and case for search."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def find_player(df: pd.DataFrame, query: str) -> pd.Series:
    """Find exactly one player by part of the name ('kane', 'mbappe', 'lamine')."""
    mask = df["player"].map(normalize_name).str.contains(normalize_name(query), regex=False)
    found = df[mask]
    if found.empty:
        raise ValueError(f"'{query}' not found among {len(df)} players")
    if len(found) > 1:
        names = ", ".join(found["player"])
        raise ValueError(f"'{query}' matches several players: {names}. Be more specific.")
    return found.iloc[0]


if __name__ == "__main__":
    pct = percentile_table(load_per90())

    print("Pool sizes:")
    print(pct["pool"].value_counts().to_string(), "\n")

    # Sanity check: the leaders of each pool in a key metric
    for pool, metric in [("FW", "npxg"), ("AM/W", "xa"), ("MF", "progressive_passes"),
                         ("FB", "progressive_carries"), ("CB", "progressive_passes")]:
        top = pct[pct["pool"] == pool].nlargest(3, metric)
        print(f"{pool:5} top 3 by {metric}: " + ", ".join(
            f"{r.player} ({getattr(r, metric):.0f})" for r in top.itertuples()))
