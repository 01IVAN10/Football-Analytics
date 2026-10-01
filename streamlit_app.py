"""Euro 2024 Scout — веб-застосунок на Streamlit.

Запуск з кореня проєкту:
    streamlit run streamlit_app.py

Як думати про Streamlit (без цього код нижче незрозумілий):
  - Файл виконується ЗВЕРХУ ДОНИЗУ щоразу, коли користувач щось змінює
    (вибрав гравця, фільтр, вкладку). Колбеків "при кліку" немає: є повторний
    прогін скрипта, і кожен віджет просто повертає своє поточне значення.
  - Тому все дороге (читання файлів, перцентилі, малювання) кешуємо.
    Без кешу кожен клік перечитував би parquet і перемальовував графіки.
  - Значення віджета з key=... живе в st.session_state. З bind="query-params"
    воно ще й записується в URL: посиланням на профіль гравця можна поділитися.

Інтерфейс англійською — як і підписи на графіках (портфоліо для міжнародної аудиторії).
"""
import threading

import matplotlib

# Бекенд Agg малює лише в пам'ять, без вікон. На Mac бекенд за замовчуванням
# відкриває вікна і може впасти, якщо малювати не з головного потоку, —
# а Streamlit виконує скрипт саме в окремих потоках. Має стояти ДО імпорту pyplot
# (його імпортують модулі src нижче), тому ці імпорти після нього.
matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import app_data  # noqa: E402
from src.metrics import MIN_MINUTES, RATIO_COLS  # noqa: E402
from src.paths import ASSETS  # noqa: E402
from src.percentiles import (COMPARISON_POOLS, LABELS, POOL_NAMES,  # noqa: E402
                             RADAR_TEMPLATES, percentile_table)
from src.pitch_maps import plot_heatmap, plot_pass_map, plot_shot_map, player_events  # noqa: E402
from src.radar import format_value, plot_radar  # noqa: E402
from src.similarity import FEATURES, best_matches, similar_players  # noqa: E402
from src.style import figure_to_png  # noqa: E402
from src.z_profile import plot_z_profile  # noqa: E402

GITHUB_URL = "https://github.com/01IVAN10/Football-Analytics"
STATSBOMB_URL = "https://github.com/hudl/open-data"   # репозиторій переїхав зі statsbomb/ у hudl/
# Умова StatsBomb Open Data: вказувати джерело і ставити їхній логотип.
# Файл — з їхнього репозиторію (img/), зменшений до 600 px.
STATSBOMB_LOGO = ASSETS / "statsbomb_logo.png"

TABS = ["Profile", "Pitch maps", "Similar players", "About"]
POOL_ORDER = ["FW", "AM/W", "MF", "FB", "CB", "GK"]   # у фільтрі: від атаки до воріт
DEFAULT_PLAYER = "Lamine Yamal"                        # хто відкривається без параметрів в URL
STRONG, WEAK = 80, 20                                  # межі "сильне / слабке місце" (перцентиль)
UNIQUE = 0.3                                           # найкращий збіг нижче — профіль унікальний
MAPS = {"Shots": plot_shot_map, "Passes": plot_pass_map, "Heatmap": plot_heatmap}

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


@st.cache_resource(show_spinner="Loading match events…")
def load_events() -> pd.DataFrame:
    """186 тис. подій (~73 МБ у пам'яті) — ОДИН спільний об'єкт на весь сервер.

    cache_resource, а не cache_data: cache_data на кожен виклик віддавав би копію,
    тобто 73 МБ і помітну затримку на кожен клік кожного відвідувача.
    cache_resource віддає всім той самий об'єкт — швидко, але його НЕ МОЖНА
    змінювати (зміну побачать усі). Тому далі лише фільтруємо: фільтр повертає
    новий DataFrame, а спільний лишається недоторканим.
    """
    return app_data.load_events()


@st.cache_data
def best_match_median() -> float:
    """Медіана найкращих збігів по турніру (~0.57) — орієнтир, щоб читати числа подібності."""
    per90, _, _ = load_tables()
    return float(best_matches(per90).median())


@st.cache_data(max_entries=500)
def similar_table(player_id: int, other_teams: bool) -> pd.DataFrame:
    per90, _, _ = load_tables()
    return similar_players(per90, player_id, n=10, other_teams=other_teams)


# pyplot не розрахований на кілька потоків одночасно, а Streamlit обслуговує
# кожного відвідувача в окремому потоці. Якщо двоє відкриють графік в ту саму мить,
# фігури можуть "змішатися". Замок пускає малювати лише один потік за раз.
# Гальмом це не стане: завдяки кешу кожен графік малюється лише один раз.
_draw_lock = threading.Lock()


def render_png(plot, *args, **kwargs) -> bytes:
    """Викликає функцію малювання під замком і повертає PNG-байти."""
    with _draw_lock:
        return figure_to_png(plot(*args, **kwargs))


# Кешуємо PNG-байти, а не Figure: байти легкі і їх можна віддавати будь-якому
# відвідувачу. Ключ кешу — аргументи функції. max_entries — стеля пам'яті:
# PNG важить 75–700 КБ (карта пасів найважча), а безкоштовний хостинг гарантує ~690 МБ.
@st.cache_data(show_spinner="Drawing radar…", max_entries=300)
def radar_png(player_id: int, compare_id: int | None = None) -> bytes:
    per90, _, pct = load_tables()
    return render_png(plot_radar, per90, pct, player_id, compare_id)


@st.cache_data(show_spinner="Drawing map…", max_entries=150)
def map_png(player_id: int, kind: str, open_play_only: bool = False) -> bytes:
    _, totals, _ = load_tables()
    player = totals.set_index("player_id").loc[player_id]
    ev = player_events(load_events(), player_id)
    extra = {"open_play_only": open_play_only} if kind == "Passes" else {}
    return render_png(MAPS[kind], ev, player, **extra)


@st.cache_data(show_spinner="Drawing comparison…", max_entries=200)
def z_profile_png(player_id: int, compare_id: int) -> bytes:
    per90, _, _ = load_tables()
    return render_png(plot_z_profile, per90, player_id, compare_id)


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


def no_profile_reason(player_id: int, player: pd.Series, pct: pd.DataFrame) -> str | None:
    """Чому в гравця немає профілю (радар, перцентилі, схожі) — або None, якщо він є."""
    if player["pool"] == "GK":
        return ("Goalkeepers have no profile yet: goalkeeper-specific metrics "
                "(saves, claims, distribution) are not computed in this project. "
                "Pitch maps are available.")
    if not pct["player_id"].eq(player_id).any():
        # .1f, а не .0f: 269.7 хв округлилось би до "270 min — below the 270-minute threshold"
        return (f"{player['player']} played {player['minutes']:.1f} min — below the "
                f"{MIN_MINUTES}-minute threshold. Per-90 numbers on such a small sample "
                "are mostly noise, so there is no profile or similarity search. "
                "Pitch maps are available.")
    return None


def open_player(player_id: int) -> None:
    """Колбек кнопки "Open profile" на вкладці схожих гравців.

    Колбек виконується ДО наступного прогону скрипта, тому тут можна змінити
    значення віджетів через st.session_state. Посеред прогону, коли віджет уже
    намальований, так робити не можна — Streamlit видасть помилку.
    Скидаємо фільтри, щоб новий гравець точно був у списку, і відкриваємо Profile.
    """
    st.session_state["role"] = "All"
    st.session_state["team"] = "All"
    st.session_state["min_minutes"] = min(st.session_state.get("min_minutes", MIN_MINUTES),
                                          MIN_MINUTES)
    st.session_state["player"] = player_id
    st.session_state["tab"] = "Profile"


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
        # Логотип — унизу як підпис "дані від", а не st.logo() угорі: st.logo — місце
        # для бренду самого застосунку, і там він виглядав би так, ніби це застосунок StatsBomb
        st.caption("Data provided by")
        st.image(str(STATSBOMB_LOGO), width=170, link=STATSBOMB_URL)
        st.caption(f"[StatsBomb Open Data]({STATSBOMB_URL}) · Code: [GitHub]({GITHUB_URL})")
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
    reason = no_profile_reason(player_id, player, pct)
    if reason:
        st.info(reason)   # пояснюємо чому, а не показуємо порожнечу
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


def render_maps(player_id: int, player: pd.Series) -> None:
    """Карти працюють для всіх 493 гравців: поріг хвилин тут не потрібен —
    карта просто показує, що гравець зробив, без ділення на 90."""
    kind = st.segmented_control("Map", list(MAPS), default="Shots", required=True,
                                key="map", bind="query-params", label_visibility="collapsed")
    open_play_only = False
    if kind == "Passes":
        open_play_only = st.toggle(
            "Open play only", key="open_play",
            help="Hide set pieces (corners, free kicks, throw-ins). Progressive passes are "
                 "open play anyway, so this changes key passes and the grey background.")

    # Ширина в пікселях: вертикальна карта ударів вужча за горизонтальні
    st.image(map_png(player_id, kind, open_play_only), width=620 if kind == "Shots" else 980)
    st.caption(f"All {player['minutes']:.0f} minutes at Euro 2024. "
               "Pitch maps use every minute played — no minutes threshold.")


def render_similar(player_id: int, player: pd.Series,
                   per90: pd.DataFrame, pct: pd.DataFrame) -> None:
    reason = no_profile_reason(player_id, player, pct)
    if reason:
        st.info(reason)
        return

    pool = player["pool"]
    pool_size = int((pct["pool"] == pool).sum())
    other_teams = st.toggle(
        "Only players from other teams", key="other_teams",
        help="The usual scouting question is who could replace this player — "
             "and a teammate is not an answer.")
    similar = similar_table(player_id, other_teams)

    best = similar["similarity"].iloc[0]
    if best < UNIQUE:
        st.warning(f"Even the best match is weak ({best:.2f}): this profile is unique "
                   f"among {pool_size} {POOL_NAMES[pool]}.")
    st.caption(f"Cosine similarity of {len(FEATURES)} per-90 metrics, z-scored within "
               f"{pool_size} {POOL_NAMES[pool]}. For reference, the median best match across "
               f"the tournament is {best_match_median():.2f}. Select a row to compare.")

    # on_select="rerun": клік по рядку перезапускає скрипт, а dataframe повертає,
    # які рядки вибрано. key містить гравця, тож для нового гравця таблиця "нова"
    # і знову стоїть вибір за замовчуванням — найсхожіший (рядок 0).
    # Коротка позиція (DM, CM, W...) замість повної "Left Defensive Midfield":
    # інакше таблиця не влазить і найважливіша колонка (різниця) ховається за прокруткою
    similar["pos"] = similar["player_id"].map(per90.set_index("player_id")["position_group"])
    event = st.dataframe(
        similar[["player", "team", "pos", "similarity", "shared", "difference"]],
        hide_index=True, on_select="rerun", selection_mode="single-row",
        selection_default={"selection": {"rows": [0]}},
        key=f"similar_{player_id}_{other_teams}",
        column_config={
            "player": st.column_config.TextColumn("Player", width=170),
            "team": st.column_config.TextColumn("Team", width=95),
            "pos": st.column_config.TextColumn(
                "Pos", width=42, help="Position group: CB, FB, DM, CM, AM, W, FW."),
            "similarity": st.column_config.ProgressColumn(
                "Similarity", min_value=0, max_value=1, format="%.2f", width=100),
            "shared": st.column_config.TextColumn(
                "Shared strengths",
                help="Top-2 metrics where both players are above the pool average "
                     "and that add most to the similarity."),
            "difference": st.column_config.TextColumn(
                "Main difference",
                help="Metric where one player is above and the other below the pool "
                     "average, with the biggest gap (z-scores)."),
        },
    )
    rows = event.selection.rows
    if not rows:   # користувач зняв вибір
        st.info("Select a player in the table to compare.")
        return
    other = similar.iloc[rows[0]]
    compare_id = int(other["player_id"])

    st.subheader(f"{player['player']} vs {other['player']}")
    # Те саме, що в колонках таблиці, але для вибраного гравця і завжди на виду:
    # на екрані ~1200 px остання колонка таблиці ховається за горизонтальною прокруткою
    st.markdown(f"**Shared strengths:** {other['shared']}  \n"
                f"**Main difference:** {other['difference']}")
    view = st.segmented_control("View", ["Z-profile", "Radar"], default="Z-profile",
                                required=True, key="compare_view", label_visibility="collapsed")
    if view == "Z-profile":
        st.image(z_profile_png(player_id, compare_id), width=820)
        st.caption("What the similarity search sees: all 18 metrics as z-scores "
                   "(0 = pool average). Similar players have dots on the same side of zero "
                   "in the same rows, even if one of them is further from zero.")
    else:
        st.image(radar_png(player_id, compare_id), width=640)
        st.caption("Percentiles of the 10 role metrics. The radar uses different numbers "
                   "than the similarity search, so a close match can look less similar here.")
    st.button(f"Open {other['player']}'s profile", on_click=open_player, args=(compare_id,),
              icon=":material/arrow_forward:")


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
where they stand among players in the same role, where they act on the pitch
and who plays a similar game.

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
(median {clearances["CB"]:.1f}). Eight position groups are merged into six pools
so that each has enough players ({MIN_MINUTES}+ min):

{pools}

A percentile of 80 means the player is ahead of about 80% of the pool.

### Pitch maps
- **Shots**: non-penalty shots, circle area = StatsBomb xG, goals in orange.
- **Passes**: progressive passes (blue) and key passes (orange) over all other
  completed passes (grey). "Open play only" hides set pieces.
- **Heatmap**: every action with a location, smoothed over about 5 yards.
  Carries are excluded because they start where the previous action ended.

### Similar players
Each player is a vector of **{len(FEATURES)} per-90 metrics** (shooting, creation,
passing, carrying, defending). Goals and assists are left out: over 3–7 matches
they are mostly luck, while npxG and xA describe the chances behind them.

1. Every metric is turned into a **z-score within the pool**: how many standard
   deviations above or below the average player in that role.
2. **Cosine similarity** compares the *shape* of two profiles, not their level:
   a player who does the same things slightly less often still matches.

**Validation.** Each player's matches were split into two halves, and the method had
to recognise the player's other half among everyone in the pool. It ranks the right
answer higher than about 75% of candidates on average (random guessing: 50%) and
puts it first 17% of the time (random: 3%). Cosine similarity beat Euclidean distance
with statistical confidence (paired bootstrap). Other choices (z-scores vs percentiles,
18 vs 10 metrics) made no measurable difference in this test.

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
# Без цього Streamlit рахує вміст УСІХ вкладок на кожен прогін і лише ховає зайві:
# радар + 3 карти + таблиця схожих на кожен клік.
#
# Вкладка в URL (?tab=Pitch+maps), щоб посилання вело одразу куди треба.
# У st.tabs немає bind="query-params", як у selectbox, тому синхронізуємо вручну:
# з URL читаємо лише стартову вкладку, а після кожного прогону записуємо активну.
# Стартову вкладку запам'ятовуємо в session_state ОДИН раз на сесію: default входить
# в "особу" віджета, і якби він мінявся разом з URL, Streamlit вважав би вкладки
# новим віджетом і скидав вибір користувача (клік по Profile повертав би на Pitch maps).
# Для першої вкладки параметр прибираємо, щоб звичайне посилання лишалось коротким.
if "start_tab" not in st.session_state:
    requested_tab = st.query_params.get("tab")
    st.session_state["start_tab"] = requested_tab if requested_tab in TABS else None
profile_tab, maps_tab, similar_tab, about_tab = st.tabs(
    TABS, key="tab", on_change="rerun", default=st.session_state["start_tab"])
if st.session_state["tab"] == TABS[0]:
    st.query_params.pop("tab", None)
else:
    st.query_params["tab"] = st.session_state["tab"]
if profile_tab.open:
    with profile_tab:
        render_profile(player_id, player, per90, pct)
if maps_tab.open:
    with maps_tab:
        render_maps(player_id, player)
if similar_tab.open:
    with similar_tab:
        render_similar(player_id, player, per90, pct)
if about_tab.open:
    with about_tab:
        render_about(per90, pct)
