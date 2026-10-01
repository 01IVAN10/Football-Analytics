"""The open dataset in data/export is in sync with the code that builds it."""
import pytest

from src import app_data
from src.export_data import PER90_CSV, TOTALS_CSV, per90_table, to_csv_text, totals_table


@pytest.mark.parametrize("build, load, path", [
    (totals_table, app_data.load_totals, TOTALS_CSV),
    (per90_table, app_data.load_per90, PER90_CSV),
])
def test_committed_csv_is_up_to_date(build, load, path):
    expected = to_csv_text(build(load()))
    assert path.read_text(encoding="utf-8") == expected, \
        f"{path.name} is stale: run python -m src.export_data"


def test_per90_percentiles():
    df = per90_table(app_data.load_per90())
    pct = df.filter(like="_pct")
    assert len(df) == 191 and df["minutes"].min() >= 270
    assert pct.min().min() > 0 and pct.max().max() == 100
    # in every pool someone has the highest pass volume
    assert (df.groupby("pool")["passes_pct"].max() == 100).all()
