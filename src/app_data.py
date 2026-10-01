"""Slim data for the web app: build (CLI) and load.

Streamlit Community Cloud only sees the repository, so the app gets its own committed
set: per-90 table, totals, and events with only the 12 columns the pitch maps read
(~75 MB in memory instead of ~590 MB; Community Cloud guarantees ~690 MB).

Run from the project root (after python -m src.metrics):
    python -m src.app_data
"""
import numpy as np
import pandas as pd

from src.metrics import classify_passes, non_penalty_shots
from src.paths import APP_DATA, PROCESSED

PER90_PATH = APP_DATA / "per90.parquet"
TOTALS_PATH = APP_DATA / "totals.parquet"
EVENTS_PATH = APP_DATA / "events.parquet"

# Columns read by src.pitch_maps (and the src.metrics helpers it calls).
# A new map that needs more (e.g. carry_end_location): add it here and rebuild.
EVENT_COLS = [
    "match_id", "player_id", "type", "period", "location",
    "pass_end_location", "pass_outcome", "pass_type", "pass_assisted_shot_id",
    "shot_type", "shot_outcome", "shot_statsbomb_xg",
]


# ---------- build ----------

def slim_events(events: pd.DataFrame) -> pd.DataFrame:
    """Player events with a location, no shootout, only the columns the maps need.

    Rows without a location (substitutions, tactical shifts, half start/end)
    cannot be drawn on a pitch.
    """
    keep = events["player_id"].notna() & (events["period"] < 5) & events["location"].notna()
    slim = events.loc[keep, EVENT_COLS].reset_index(drop=True)
    slim["player_id"] = slim["player_id"].astype(int)
    return slim


def check_events(slim: pd.DataFrame, totals: pd.DataFrame) -> None:
    """The slim events must reproduce player_totals for every player (catches a column
    missing from EVENT_COLS or rows filtered out by mistake)."""
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
    # Players without a single shot/pass are missing from from_slim: that is 0
    from_slim = from_slim.reindex(expected.index).fillna(0)

    # isclose: npxG is a sum of floats, the order of addition changes the last digit
    mismatch = ~np.isclose(from_slim, expected).all(axis=1)
    if mismatch.any():
        raise AssertionError(f"Mismatch with player_totals for {mismatch.sum()} players: "
                             f"{list(expected.index[mismatch][:5])}")


def build() -> None:
    """Build data/app from data/raw and data/processed."""
    from src.data_loader import load_events   # imports statsbombpy, only needed here

    APP_DATA.mkdir(parents=True, exist_ok=True)
    per90 = pd.read_parquet(PROCESSED / "player_per90.parquet")
    totals = pd.read_parquet(PROCESSED / "player_totals.parquet")
    events = slim_events(load_events())

    check_events(events, totals)

    for df, path in [(per90, PER90_PATH), (totals, TOTALS_PATH), (events, EVENTS_PATH)]:
        df.to_parquet(path, index=False)
        memory = df.memory_usage(deep=True).sum() / 1e6
        print(f"{path.name:15} {len(df):>7} rows · file {path.stat().st_size / 1e6:.2f} MB"
              f" · in memory {memory:.1f} MB")
    print(f"Check passed: shots, goals, npxG, progressive and key passes "
          f"= player_totals for all {len(totals)} players")


# ---------- load (used by the app) ----------

def load_per90() -> pd.DataFrame:
    return pd.read_parquet(PER90_PATH)


def load_totals() -> pd.DataFrame:
    return pd.read_parquet(TOTALS_PATH)


def load_events() -> pd.DataFrame:
    return pd.read_parquet(EVENTS_PATH)


if __name__ == "__main__":
    build()
