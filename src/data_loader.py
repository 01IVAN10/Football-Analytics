"""Download StatsBomb Euro 2024 data once and cache it as parquet in data/raw.

Run from the project root:
    python -m src.data_loader
"""
import warnings

import pandas as pd
from statsbombpy import sb

from src.paths import RAW

# Without credentials statsbombpy warns that it uses open data, which is what we want
warnings.filterwarnings("ignore", message="credentials were not supplied")

COMPETITION_ID = 55  # UEFA Euro
SEASON_ID = 282      # 2024

MATCHES_PATH = RAW / "matches_euro2024.parquet"
EVENTS_PATH = RAW / "events_euro2024.parquet"


def download_matches() -> pd.DataFrame:
    """All 51 matches of the tournament."""
    return sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)


def download_events(match_ids: list[int]) -> pd.DataFrame:
    """Events of all matches in one table, with a match_id column."""
    frames = []
    for i, match_id in enumerate(match_ids, start=1):
        ev = sb.events(match_id=match_id)
        ev["match_id"] = match_id
        frames.append(ev)
        print(f"events {i}/{len(match_ids)}", end="\r")
    print()
    # Matches have slightly different columns (not every match has a penalty, etc.);
    # concat aligns them and fills the gaps with NaN
    return pd.concat(frames, ignore_index=True)


def download_all() -> None:
    """Download matches and events into data/raw. Run once."""
    RAW.mkdir(parents=True, exist_ok=True)

    matches = download_matches()
    matches.to_parquet(MATCHES_PATH)
    match_ids = matches["match_id"].tolist()
    print(f"Matches: {len(match_ids)}")

    download_events(match_ids).to_parquet(EVENTS_PATH)
    print("Done:", RAW)


def load_matches() -> pd.DataFrame:
    return pd.read_parquet(MATCHES_PATH)


def load_events() -> pd.DataFrame:
    return pd.read_parquet(EVENTS_PATH)


if __name__ == "__main__":
    download_all()
