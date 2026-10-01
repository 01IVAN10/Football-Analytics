"""Open dataset: the tables behind the app and the report, as CSV.

Two files, built from the committed app data (data/app), so anyone can
regenerate them without the StatsBomb API:

    python -m src.export_data

euro2024_player_totals.csv  all 493 players, tournament totals
euro2024_player_per90.csv   191 players with 270+ minutes: per-90 values
                            and percentiles within the comparison pool

Column definitions: data/export/README.md.
"""
import pandas as pd

from src import app_data
from src.metrics import COUNT_COLS, RATIO_COLS
from src.paths import EXPORT
from src.percentiles import LABELS, percentile_table

TOTALS_CSV = EXPORT / "euro2024_player_totals.csv"
PER90_CSV = EXPORT / "euro2024_player_per90.csv"

INFO_COLS = ["player_id", "player", "team", "position_group", "main_position",
             "matches", "minutes"]
# xG-based metrics are fractional; every other counting metric is a whole number
FRACTIONAL = ["npxg", "xa"]


def totals_table(totals: pd.DataFrame) -> pd.DataFrame:
    df = totals[INFO_COLS + COUNT_COLS + RATIO_COLS].copy()
    whole = [c for c in COUNT_COLS if c not in FRACTIONAL]
    df[whole] = df[whole].astype(int)
    df = df.round({"minutes": 1, **{c: 3 for c in FRACTIONAL + RATIO_COLS}})
    return df.sort_values(["team", "player"], ignore_index=True)


def per90_table(per90: pd.DataFrame) -> pd.DataFrame:
    pct = percentile_table(per90)          # same rows and index as per90
    df = per90[INFO_COLS + COUNT_COLS + RATIO_COLS].round(3)
    df["minutes"] = df["minutes"].round(1)
    df.insert(df.columns.get_loc("position_group") + 1, "pool", pct["pool"])
    df = df.join(pct[list(LABELS)].round(1).add_suffix("_pct"))
    return df.sort_values(["pool", "team", "player"], ignore_index=True)


def to_csv_text(df: pd.DataFrame) -> str:
    # Fixed line ending: the test compares file text, which must not depend on the OS
    return df.to_csv(index=False, lineterminator="\n")


def build() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    tables = [(totals_table(app_data.load_totals()), TOTALS_CSV),
              (per90_table(app_data.load_per90()), PER90_CSV)]
    for df, path in tables:
        path.write_text(to_csv_text(df), encoding="utf-8")
        print(f"{path.name:30} {len(df):>4} rows × {df.shape[1]} columns"
              f" · {path.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    build()
