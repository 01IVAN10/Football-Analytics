"""Smoke tests of the web app: every page renders without errors.

Uses Streamlit's AppTest (no browser). Covers edge cases that are easy to miss by hand:
a player without a position, goalkeepers, 269.7 minutes, empty filter results.

Run from the project root (≈90 s):
    python -m pytest tests/ -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_data
from src.metrics import MIN_MINUTES

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
TOTALS = app_data.load_totals()


def player_id(name: str) -> int:
    return int(TOTALS.loc[TOTALS["player"].str.startswith(name), "player_id"].iloc[0])


def switch(at: AppTest, tab: str, **state) -> AppTest:
    """Open a tab (and set other widgets) and rerun. AppTest does not keep the active
    tab between runs, so it is set before every run."""
    at.session_state["tab"] = tab
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


@pytest.fixture
def app() -> AppTest:
    """A fresh app for every test, so widget state does not leak between tests."""
    return AppTest.from_file(APP, default_timeout=60).run()


# ---------- sidebar and profile ----------

def test_default_page(app):
    assert not app.exception
    assert app.title[0].value.startswith("Lamine Yamal")   # title[0] is the page, [1] the sidebar
    assert len(app.main.get("image")) == 1                     # radar


def test_every_profile_renders(app):
    """All 191 players with 270+ minutes: a radar for everyone except goalkeepers."""
    per90 = app_data.load_per90()
    for pid, group in zip(per90["player_id"], per90["position_group"]):
        app.selectbox(key="player").set_value(int(pid)).run()
        assert not app.exception, pid
        if group == "GK":
            assert not app.main.get("image") and "Goalkeepers" in app.info[0].value
        else:
            assert len(app.main.get("image")) == 1, pid


def test_players_under_threshold(app):
    """Slider at 0 lists all 493; below 270 minutes an explanation replaces the radar."""
    app.slider(key="min_minutes").set_value(0).run()
    assert len(app.selectbox(key="player").options) == len(TOTALS)

    low = TOTALS[(TOTALS["minutes"] < MIN_MINUTES) & (TOTALS["position_group"] != "GK")]
    for pid in low["player_id"]:
        app.selectbox(key="player").set_value(int(pid)).run()
        assert not app.exception, pid
        assert "below the" in app.info[0].value, pid


def test_empty_filters_show_warning(app):
    app.selectbox(key="role").set_value("FW").run()
    app.selectbox(key="team").set_value("Ukraine").run()   # Dovbyk: 259 min, below the threshold
    assert not app.exception
    assert "No players match" in app.warning[0].value


def test_open_from_url():
    """A link with ?player=...&min_minutes=0 opens that player."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["player"] = "Artem Dovbyk (Ukraine)"
    at.query_params["min_minutes"] = "0"
    at.run()
    assert at.title[0].value == "Artem Dovbyk"


def test_tab_from_url():
    """?tab=Similar+players opens that tab directly (and it stays in the URL)."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["player"] = "Toni Kroos (Germany)"
    at.query_params["tab"] = "Similar players"
    at.run()
    assert not at.exception
    assert at.title[0].value == "Toni Kroos"
    assert len(at.dataframe) == 1                        # the similar players table, not the profile
    assert at.query_params["tab"] == ["Similar players"]


# ---------- pitch maps ----------

# Stars + edge cases: < 270 minutes, a goalkeeper, a player without a position (3 minutes)
@pytest.mark.parametrize("name", ["Lamine Yamal", "Toni Kroos", "Artem Dovbyk",
                                  "Jordan Pickford", "Daley Blind"])
def test_pitch_maps(app, name):
    app.slider(key="min_minutes").set_value(0).run()
    for kind in ["Shots", "Passes", "Heatmap"]:
        switch(app, "Pitch maps", player=player_id(name), map=kind)
        assert not app.exception, kind
        assert len(app.main.get("image")) == 1, kind
    switch(app, "Pitch maps", map="Passes", open_play=True)
    assert not app.exception


# ---------- similar players ----------

def test_similar_players(app):
    switch(app, "Similar players")
    table = app.dataframe[0].value
    assert len(table) == 10
    assert table["similarity"].is_monotonic_decreasing
    assert app.subheader[0].value.endswith(table["player"].iloc[0])   # closest match by default

    switch(app, "Similar players", other_teams=True)
    assert "Spain" not in app.dataframe[0].value["team"].tolist()     # Yamal plays for Spain


def test_select_row_and_open_profile(app):
    """Selecting row 3 compares with that player; the button opens his profile."""
    yamal = player_id("Lamine Yamal")
    selection = {"selection": {"rows": [2], "columns": [], "cells": []}}
    switch(app, "Similar players", **{f"similar_{yamal}_False": selection})
    third = app.dataframe[0].value["player"].iloc[2]
    assert app.subheader[0].value.endswith(third)

    switch(app, "Similar players", compare_view="Radar")
    assert not app.exception and len(app.main.get("image")) == 1

    app.session_state["tab"] = "Similar players"
    app.button[0].click().run()
    assert app.title[0].value == third
    assert app.session_state["tab"] == "Profile"


def test_unique_profile_warning(app):
    """Kane: best match 0.22, the app says the profile is unique."""
    switch(app, "Similar players", player=player_id("Harry Kane"))
    assert "unique" in app.warning[0].value


def test_no_similarity_below_threshold(app):
    app.slider(key="min_minutes").set_value(0).run()
    switch(app, "Similar players", player=player_id("Artem Dovbyk"))
    assert not app.dataframe and "below the" in app.info[0].value


def test_about(app):
    switch(app, "About")
    assert not app.exception
    assert "Similar players" in app.markdown[0].value
