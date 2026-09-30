"""Смоук-тести веб-застосунку: кожна сторінка відкривається без помилок.

AppTest (вбудований у Streamlit) виконує streamlit_app.py без браузера:
віджети можна "клікати" з коду і перевіряти, що з'явилось на сторінці.
Ловить те, що легко пропустити руками: гравець без позиції, воротар,
гравець з 269.7 хв, порожній результат фільтрів.

Запуск з кореня проєкту (≈30 с):
    python -m pytest tests/ -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_data
from src.metrics import MIN_MINUTES

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def app() -> AppTest:
    """Свіжий застосунок для кожного тесту (стан віджетів не протікає між тестами)."""
    return AppTest.from_file(APP, default_timeout=60).run()


def test_default_page(app):
    assert not app.exception
    assert app.title[0].value.startswith("Lamine Yamal")   # title[0] — головна сторінка, [1] — сайдбар
    assert len(app.get("image")) == 1                     # радар


def test_every_profile_renders(app):
    """Усі 191 гравець з 270+ хв: радар у всіх, крім воротарів (у них — пояснення)."""
    per90 = app_data.load_per90()
    for player_id, group in zip(per90["player_id"], per90["position_group"]):
        app.selectbox(key="player").set_value(int(player_id)).run()
        assert not app.exception, player_id
        if group == "GK":
            assert not app.get("image") and "Goalkeepers" in app.info[0].value
        else:
            assert len(app.get("image")) == 1, player_id


def test_players_under_threshold(app):
    """Повзунок на 0 відкриває всіх 493; у кого < 270 хв — пояснення замість радару."""
    totals = app_data.load_totals()
    app.slider(key="min_minutes").set_value(0).run()
    assert len(app.selectbox(key="player").options) == len(totals)

    low = totals[(totals["minutes"] < MIN_MINUTES) & (totals["position_group"] != "GK")]
    for player_id in low["player_id"]:
        app.selectbox(key="player").set_value(int(player_id)).run()
        assert not app.exception, player_id
        assert "below the" in app.info[0].value, player_id


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
