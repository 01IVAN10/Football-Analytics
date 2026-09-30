"""Euro 2024 Scout — веб-застосунок на Streamlit.

Запуск з кореня проєкту:
    streamlit run streamlit_app.py

Як думати про Streamlit (без цього код нижче незрозумілий):
  - Файл виконується ЗВЕРХУ ДОНИЗУ щоразу, коли користувач щось змінює
    (вибрав гравця, фільтр, вкладку). Колбеків "при кліку" немає: є повторний
    прогін скрипта, і кожен віджет просто повертає своє поточне значення.
  - Тому все дороге (читання файлів, перцентилі, малювання) кешуємо.
    Без кешу кожен клік перечитував би parquet і перемальовував радар.
  - Значення віджета з key=... живе в st.session_state. З bind="query-params"
    воно ще й записується в URL: посиланням на профіль гравця можна поділитися.

Інтерфейс англійською — як і підписи на графіках (портфоліо для міжнародної аудиторії).
"""
import threading

import matplotlib

# Бекенд Agg малює лише в пам'ять, без вікон. На Mac бекенд за замовчуванням
# відкриває вікна і може впасти, якщо малювати не з головного потоку, —
# а Streamlit виконує скрипт саме в окремих потоках. Має стояти ДО імпорту pyplot
# (його імпортують src.radar і src.style), тому ці імпорти нижче.
matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import app_data  # noqa: E402
from src.metrics import MIN_MINUTES, RATIO_COLS  # noqa: E402
from src.percentiles import (COMPARISON_POOLS, LABELS, POOL_NAMES,  # noqa: E402
                             RADAR_TEMPLATES, percentile_table)
from src.radar import format_value, plot_radar  # noqa: E402
from src.style import figure_to_png  # noqa: E402

GITHUB_URL = "https://github.com/01IVAN10/Football-Analytics"
STATSBOMB_URL = "https://github.com/statsbomb/open-data"

POOL_ORDER = ["FW", "AM/W", "MF", "FB", "CB", "GK"]   # у фільтрі: від атаки до воріт
DEFAULT_PLAYER = "Lamine Yamal"                        # хто відкривається без параметрів в URL
STRONG, WEAK = 80, 20                                  # межі "сильне / слабке місце" (перцентиль)

# Має бути першою командою Streamlit у скрипті
st.set_page_config(page_title="Euro 2024 Scout", page_icon=":material/sports_soccer:",
                   layout="wide")


# ---------- дані й кеш ----------

@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """per90 (191 гравець з 270+ хв), totals (усі 493) і перцентилі.

    st.cache_data запам'ятовує результат за аргументами функції (тут їх немає —
    отже, один запис на весь сервер: файли читаються один раз, а не на кожен клік).
    Кожен виклик повертає КОПІЮ, тож змінювати таблицю безпечно: кеш для інших
    відвідувачів не зіпсується. Для таблиць у сотні рядків копія майже безкоштовна.
    """
    per90 = app_data.load_per90()
    totals = app_data.load_totals()
    totals["pool"] = totals["position_group"].map(COMPARISON_POOLS)
    return per90, totals, percentile_table(per90)


# pyplot не розрахований на кілька потоків одночасно, а Streamlit обслуговує
# кожного відвідувача в окремому потоці. Якщо двоє відкриють радар в ту саму мить,
# фігури можуть "змішатися". Замок пускає малювати лише один потік за раз.
# Гальмом це не стане: завдяки кешу кожен радар малюється лише один раз.
_draw_lock = threading.Lock()


@st.cache_data(show_spinner="Drawing radar…", max_entries=500)
def radar_png(player_id: int, compare_id: int | None = None) -> bytes:
    """Радар як PNG-байти. Ключ кешу — (player_id, compare_id).

    Кешуємо байти, а не Figure: байти легкі, їх можна копіювати і віддавати
    будь-якому відвідувачу. max_entries — стеля (191 гравець + порівняння),
    щоб кеш не ріс безмежно.
    """
    per90, _, pct = load_tables()
    with _draw_lock:
        return figure_to_png(plot_radar(per90, pct, player_id, compare_id))


# ---------- допоміжне ----------

def ordinal(n: float) -> str:
    """97 -> '97th', 1 -> '1st', 22 -> '22nd', 13 -> '13th'."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def value_text(metric: str, value: float) -> str:
    """'8.10 per 90' або '90%' (частки не діляться на хвилини)."""
    text = format_value(metric, value)
    return text if metric in RATIO_COLS or text == "n/a" else f"{text} per 90"


def pool_label(pool: str) -> str:
    return "All roles" if pool == "All" else POOL_NAMES[pool].capitalize()


# ---------- сайдбар ----------

def sidebar(totals: pd.DataFrame) -> int:
    """Фільтри + вибір гравця. Повертає player_id.

    Усі чотири віджети прив'язані до URL (bind="query-params"): посилання
    ...?role=Forwards&player=... відкриє застосунок у тому самому стані.
    """
    with st.sidebar:
        st.title("Euro 2024 Scout")
        st.caption("Player profiles from StatsBomb event data: 51 matches, 24 teams, 493 players.")

        pool = st.selectbox("Role", ["All"] + POOL_ORDER, format_func=pool_label,
                            key="role", bind="query-params")
        team = st.selectbox("Team", ["All"] + sorted(totals["team"].unique()),
                            format_func=lambda t: "All teams" if t == "All" else t,
                            key="team", bind="query-params")
        min_minutes = st.slider(
            "Minimum minutes played", 0, 600, MIN_MINUTES, step=30,
            key="min_minutes", bind="query-params",
            help=f"Radar, percentiles and similar players need {MIN_MINUTES}+ minutes "
                 "(3 full matches). Lower it to find squad players.")

        # Фільтри — звичайні булеві маски pandas, як у CLI-скриптах
        players = totals[totals["minutes"] >= min_minutes]
        if pool != "All":
            players = players[players["pool"] == pool]
        if team != "All":
            players = players[players["team"] == team]

        if players.empty:
            st.warning("No players match these filters. Try lowering the minimum minutes.")
            st.stop()   # далі скрипт не виконується: показувати нічого

        players = players.sort_values("player")
        labels = dict(zip(players["player_id"], players["player"] + " (" + players["team"] + ")"))
        options = players["player_id"].tolist()     # .tolist() -> звичайні int, не numpy.int64

        # За замовчуванням — DEFAULT_PLAYER, якщо він пройшов фільтри, інакше перший у списку
        default = players["player"].str.startswith(DEFAULT_PLAYER)
        index = options.index(players.loc[default, "player_id"].iloc[0]) if default.any() else 0

        # filter_mode="fuzzy" (за замовчуванням): можна друкувати "yamal" або навіть
        # назву збірної — в підписі є команда, тож "ukraine" покаже всіх українців
        player_id = st.selectbox(f"Player ({len(options)})", options, index=index,
                                 format_func=labels.get, key="player", bind="query-params")

        st.divider()
        st.caption(f"Data: [StatsBomb Open Data]({STATSBOMB_URL}) · "
                   f"Code: [GitHub]({GITHUB_URL})")
    return player_id


# ---------- вкладки ----------

def header(player: pd.Series) -> None:
    st.title(player["player"])
    parts = [player["team"], player["main_position"],
             f"{player['minutes']:.0f} min",
             f"{player['matches']} match" + ("es" if player["matches"] != 1 else "")]
    # pd.notna: у 3 гравців без жодної дії на полі позиція невідома (NaN)
    st.caption(" · ".join(str(p) for p in parts if pd.notna(p)))


def metrics_table(raw: pd.Series, pct_row: pd.Series) -> pd.DataFrame:
    """Усі 22 метрики: значення і перцентиль. Таблиця = доступна альтернатива радару:
    точні числа, всі метрики, а не лише 10 на радарі."""
    return pd.DataFrame({
        "Metric": [LABELS[m] for m in LABELS],
        "Value": [format_value(m, raw[m]) for m in LABELS],
        "Percentile": [pct_row[m] for m in LABELS],
    })


def render_profile(player_id: int, player: pd.Series,
                   per90: pd.DataFrame, pct: pd.DataFrame) -> None:
    # Два випадки, коли профілю немає — пояснюємо чому, а не показуємо порожнечу
    if player["pool"] == "GK":
        st.info("Goalkeepers have no profile yet: goalkeeper-specific metrics "
                "(saves, claims, distribution) are not computed in this project.")
        return
    if not pct["player_id"].eq(player_id).any():
        # .1f, а не .0f: 269.6 хв округлилось би до "270 min — below the 270-minute threshold"
        st.info(f"{player['player']} played {player['minutes']:.1f} min — below the "
                f"{MIN_MINUTES}-minute threshold. Per-90 numbers on such a small sample "
                "are mostly noise, so there is no radar or percentile profile.")
        return

    pct_row = pct.set_index("player_id").loc[player_id]
    raw = per90.set_index("player_id").loc[player_id]
    pool = pct_row["pool"]
    pool_size = int((pct["pool"] == pool).sum())

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.image(radar_png(player_id), width="stretch")

    with right:
        # Сильні й слабкі місця — лише серед 10 метрик радару для цієї ролі:
        # 95-й перцентиль за виносами в нападника — не "сила", а дрібниця
        ranks = pct_row[RADAR_TEMPLATES[pool]].astype(float).dropna().sort_values(ascending=False)
        strong = ranks[ranks >= STRONG].head(3)
        weak = ranks[ranks <= WEAK].sort_values().head(3)

        st.subheader("Strengths")
        if strong.empty:
            st.write(f"No radar metric in the top {100 - STRONG}% of the pool.")
        for m, p in strong.items():
            st.markdown(f"**{LABELS[m]}** — {ordinal(p)} percentile · {value_text(m, raw[m])}")

        st.subheader("Weak spots")
        if weak.empty:
            st.write(f"No radar metric in the bottom {WEAK}% of the pool.")
        for m, p in weak.items():
            st.markdown(f"**{LABELS[m]}** — {ordinal(p)} percentile · {value_text(m, raw[m])}")

        st.caption(f"Percentiles vs {pool_size} {POOL_NAMES[pool]} with {MIN_MINUTES}+ min. "
                   "Defensive volume depends on team possession (no possession adjustment).")

    st.subheader("All metrics")
    table = metrics_table(raw, pct_row)
    st.dataframe(
        table, hide_index=True,
        height=(len(table) + 1) * 35 + 3,   # усі рядки без прокрутки (35 px на рядок)
        column_config={
            "Value": st.column_config.TextColumn(
                help="Per 90 minutes, penalties excluded. Pass completion and "
                     "npxG per shot are ratios."),
            "Percentile": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f",
                help=f"Share of the {pool_size} {POOL_NAMES[pool]} this player is ahead of."),
        },
    )


def render_about(per90: pd.DataFrame, pct: pd.DataFrame) -> None:
    sizes = pct["pool"].value_counts()
    pools = "\n".join(f"- {POOL_NAMES[p].capitalize()}: {sizes.get(p, 0)}"
                      + (" (no profile yet)" if p == "GK" else "") for p in POOL_ORDER)
    # Приклад "чому всередині ролі" рахуємо з даних, а не пишемо з голови
    clearances = per90.groupby(per90["position_group"].map(COMPARISON_POOLS))["clearances"].median()
    st.markdown(f"""
### What this is
A scouting tool built on **StatsBomb open event data for UEFA Euro 2024**
(51 matches, every on-ball action with coordinates). Pick a player to see
where they stand among players in the same role.

### How the numbers are made
- **Minutes** are counted from events (line-ups, substitutions, red cards) and
  include stoppage time. Penalty shoot-outs are excluded.
- **Per 90** metrics use players with **{MIN_MINUTES}+ minutes** (3 full matches);
  below that the sample is mostly noise.
- **Shooting** excludes penalties (npxG, non-penalty goals).
  **xA** = xG of the shots a player's passes led to.
- **Progressive** pass or carry: the ball ends at least 25% closer to the
  opponent's goal (carries: also at least 5 yards). Set pieces are not counted
  as progression.

### Comparison pools
Percentiles are computed **within a role**: 3 clearances per 90 is a lot for a
winger (pool median {clearances["AM/W"]:.1f}) and ordinary for a centre-back
(median {clearances["CB"]:.1f}). Eight position groups are merged
into six pools so that each has enough players ({MIN_MINUTES}+ min):

{pools}

A percentile of 80 means the player is ahead of about 80% of the pool.

### Limitations
- Small samples: 3–7 matches per player, only 17 forwards in their pool.
- No possession adjustment: defensive volume depends on how much the team defends.
- Goalkeepers have no specific metrics yet.

Data: [StatsBomb Open Data]({STATSBOMB_URL}) · Code: [GitHub]({GITHUB_URL})
""")


# ---------- сторінка ----------

per90, totals, pct = load_tables()
player_id = sidebar(totals)
player = totals.set_index("player_id").loc[player_id]

header(player)

# on_change="rerun" + .open = "ліниві" вкладки: виконується код лише відкритої.
# Без цього Streamlit рахує вміст УСІХ вкладок на кожен прогін і лише ховає зайві.
profile_tab, about_tab = st.tabs(["Profile", "About"], key="tab", on_change="rerun")
if profile_tab.open:
    with profile_tab:
        render_profile(player_id, player, per90, pct)
if about_tab.open:
    with about_tab:
        render_about(per90, pct)
