"""Смоук-тести веб-застосунку: кожна сторінка відкривається без помилок.

AppTest (вбудований у Streamlit) виконує streamlit_app.py без браузера:
віджети можна "клікати" з коду і перевіряти, що з'явилось на сторінці.
Ловить те, що легко пропустити руками: гравець без позиції, воротар,
гравець з 269.7 хв, порожній результат фільтрів.

Запуск з кореня проєкту (≈40 с):
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
    """Відкрити вкладку (і змінити інші віджети через session_state) і перезапустити.

    Активна вкладка — це стан браузера, і AppTest його між прогонами не пам'ятає:
    без цього кожен run() повертався б на першу вкладку. Тому ставимо вкладку
    через її key ("tab") перед кожним прогоном.
    """
    at.session_state["tab"] = tab
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


@pytest.fixture
def app() -> AppTest:
    """Свіжий застосунок для кожного тесту (стан віджетів не протікає між тестами)."""
    return AppTest.from_file(APP, default_timeout=60).run()


# ---------- сайдбар і профіль ----------

def test_default_page(app):
    assert not app.exception
    assert app.title[0].value.startswith("Lamine Yamal")   # title[0] — головна сторінка, [1] — сайдбар
    assert len(app.get("image")) == 1                     # радар


def test_every_profile_renders(app):
    """Усі 191 гравець з 270+ хв: радар у всіх, крім воротарів (у них — пояснення)."""
    per90 = app_data.load_per90()
    for pid, group in zip(per90["player_id"], per90["position_group"]):
        app.selectbox(key="player").set_value(int(pid)).run()
        assert not app.exception, pid
        if group == "GK":
            assert not app.get("image") and "Goalkeepers" in app.info[0].value
        else:
            assert len(app.get("image")) == 1, pid


def test_players_under_threshold(app):
    """Повзунок на 0 відкриває всіх 493; у кого < 270 хв — пояснення замість радару."""
    app.slider(key="min_minutes").set_value(0).run()
    assert len(app.selectbox(key="player").options) == len(TOTALS)

    low = TOTALS[(TOTALS["minutes"] < MIN_MINUTES) & (TOTALS["position_group"] != "GK")]
    for pid in low["player_id"]:
        app.selectbox(key="player").set_value(int(pid)).run()
        assert not app.exception, pid
        assert "below the" in app.info[0].value, pid


def test_empty_filters_show_warning(app):
    app.selectbox(key="role").set_value("FW").run()
    app.selectbox(key="team").set_value("Ukraine").run()   # Довбик — 259 хв, менше порогу
    assert not app.exception
    assert "No players match" in app.warning[0].value


def test_open_from_url():
    """Посилання ?player=...&min_minutes=0 відкриває потрібного гравця."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["player"] = "Artem Dovbyk (Ukraine)"
    at.query_params["min_minutes"] = "0"
    at.run()
    assert at.title[0].value == "Artem Dovbyk"


# ---------- карти ----------

# Зірки + крайні випадки: < 270 хв, воротар, гравець без позиції (3 хв на полі)
@pytest.mark.parametrize("name", ["Lamine Yamal", "Toni Kroos", "Artem Dovbyk",
                                  "Jordan Pickford", "Daley Blind"])
def test_pitch_maps(app, name):
    app.slider(key="min_minutes").set_value(0).run()
    for kind in ["Shots", "Passes", "Heatmap"]:
        switch(app, "Pitch maps", player=player_id(name), map=kind)
        assert not app.exception, kind
        assert len(app.get("image")) == 1, kind
    switch(app, "Pitch maps", map="Passes", open_play=True)
    assert not app.exception


# ---------- схожі гравці ----------

def test_similar_players(app):
    switch(app, "Similar players")
    table = app.dataframe[0].value
    assert len(table) == 10
    assert table["similarity"].is_monotonic_decreasing
    assert app.subheader[0].value.endswith(table["player"].iloc[0])   # за замовчуванням — найсхожіший

    switch(app, "Similar players", other_teams=True)
    assert "Spain" not in app.dataframe[0].value["team"].tolist()     # Ямаль грає за Іспанію


def test_select_row_and_open_profile(app):
    """Вибір 3-го рядка -> порівняння з ним; кнопка -> його профіль."""
    yamal = player_id("Lamine Yamal")
    selection = {"selection": {"rows": [2], "columns": [], "cells": []}}
    switch(app, "Similar players", **{f"similar_{yamal}_False": selection})
    third = app.dataframe[0].value["player"].iloc[2]
    assert app.subheader[0].value.endswith(third)

    switch(app, "Similar players", compare_view="Radar")
    assert not app.exception and len(app.get("image")) == 1

    app.session_state["tab"] = "Similar players"
    app.button[0].click().run()
    assert app.title[0].value == third
    assert app.session_state["tab"] == "Profile"


def test_unique_profile_warning(app):
    """Кейн: найкращий збіг 0.22 — застосунок чесно каже, що профіль унікальний."""
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
