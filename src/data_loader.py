"""Завантаження даних StatsBomb для Євро 2024 і кешування у parquet.

Ідея: один раз качаємо все з API, зберігаємо у data/raw,
а далі весь проєкт читає локальні файли (швидко і без інтернету).
"""
import warnings

import pandas as pd
from statsbombpy import sb

from src.paths import RAW

# statsbombpy без логіна попереджає, що використовує відкриті дані, — це нормально
warnings.filterwarnings("ignore", message="credentials were not supplied")

COMPETITION_ID = 55  # UEFA Euro
SEASON_ID = 282      # 2024

MATCHES_PATH = RAW / "matches_euro2024.parquet"
EVENTS_PATH = RAW / "events_euro2024.parquet"


def download_matches() -> pd.DataFrame:
    """Список усіх 51 матчу турніру."""
    return sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)


def download_events(match_ids: list[int]) -> pd.DataFrame:
    """Події всіх матчів в одній таблиці з колонкою match_id."""
    frames = []
    for i, match_id in enumerate(match_ids, start=1):
        ev = sb.events(match_id=match_id)
        ev["match_id"] = match_id
        frames.append(ev)
        print(f"events {i}/{len(match_ids)}", end="\r")
    print()
    # Різні матчі мають трохи різні набори колонок (напр., не в кожному є пенальті).
    # concat об'єднує їх, а відсутні значення заповнює NaN.
    return pd.concat(frames, ignore_index=True)


def download_all() -> None:
    """Качає матчі та події і зберігає у data/raw. Запускати один раз."""
    RAW.mkdir(parents=True, exist_ok=True)

    matches = download_matches()
    matches.to_parquet(MATCHES_PATH)
    match_ids = matches["match_id"].tolist()
    print(f"Матчів: {len(match_ids)}")

    download_events(match_ids).to_parquet(EVENTS_PATH)
    print("Готово:", RAW)


def load_matches() -> pd.DataFrame:
    return pd.read_parquet(MATCHES_PATH)


def load_events() -> pd.DataFrame:
    return pd.read_parquet(EVENTS_PATH)


if __name__ == "__main__":
    # Запуск з кореня проєкту:  python -m src.data_loader
    download_all()
