"""Euro 2024 Scout: Streamlit web app.

Run from the project root:
    streamlit run streamlit_app.py

Streamlit reruns this file top to bottom on every interaction, so everything
expensive (reading files, percentiles, drawing) is cached. Widgets with
bind="query-params" are mirrored in the URL, so every view can be shared as a link.
"""
import threading

import matplotlib

# Agg draws to memory only. The default macOS backend opens windows and can crash when
# drawing outside the main thread, and Streamlit runs scripts in worker threads.
# Must be set before pyplot is imported (by the src modules below).
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
STATSBOMB_URL = "https://github.com/hudl/open-data"   # moved from statsbomb/ to hudl/
# StatsBomb Open Data terms: credit the source and show their logo
# (taken from img/ in their repository, resized to 600 px)
STATSBOMB_LOGO = ASSETS / "statsbomb_logo.png"

TABS = ["Profile", "Pitch maps", "Similar players", "About"]
POOL_ORDER = ["FW", "AM/W", "MF", "FB", "CB", "GK"]   # filter order: from attack to goal
DEFAULT_PLAYER = "Lamine Yamal"                        # shown when the URL names no player
STRONG, WEAK = 80, 20                                  # percentile bounds for strengths / weak spots
UNIQUE = 0.3                                           # best match below this = unique profile
MAPS = {"Shots": plot_shot_map, "Passes": plot_pass_map, "Heatmap": plot_heatmap}

# Must be the first Streamlit command
st.set_page_config(page_title="Euro 2024 Scout", page_icon=":material/sports_soccer:",
                   layout="wide")


# ---------- data and caching ----------

@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """per90 (191 players with 270+ minutes), totals (all 493) and percentiles."""
    per90 = app_data.load_per90()
    totals = app_data.load_totals()
    totals["pool"] = totals["position_group"].map(COMPARISON_POOLS)
    return per90, totals, percentile_table(per90)


@st.cache_resource(show_spinner="Loading match events…")
def load_events() -> pd.DataFrame:
    """186k events (~73 MB): one shared object (cache_data would copy it on every call).

    Must never be modified; the app only filters it.
    """
    return app_data.load_events()


@st.cache_data
def best_match_median() -> float:
    """Median best match across the tournament (~0.57): a scale for similarity numbers."""
    per90, _, _ = load_tables()
    return float(best_matches(per90).median())


@st.cache_data(max_entries=500)
def similar_table(player_id: int, other_teams: bool) -> pd.DataFrame:
    per90, _, _ = load_tables()
    return similar_players(per90, player_id, n=10, other_teams=other_teams)


# pyplot is not thread-safe and Streamlit serves each session in its own thread:
# two charts drawn at the same moment could get mixed up. The lock lets one thread
# draw at a time; with caching, each chart is drawn only once anyway.
_draw_lock = threading.Lock()


def render_png(plot, *args, **kwargs) -> bytes:
    """Call a plotting function under the lock and return PNG bytes."""
    with _draw_lock:
        return figure_to_png(plot(*args, **kwargs))


# Cache PNG bytes rather than Figures: bytes are light and can be served to any session.
# max_entries caps memory: a PNG is 75-700 KB (the pass map is the heaviest), and the
# free hosting guarantees ~690 MB.
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


# ---------- helpers ----------

def ordinal(n: float) -> str:
    """97 -> '97th', 1 -> '1st', 22 -> '22nd', 13 -> '13th'."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def value_text(metric: str, value: float) -> str:
    """'8.10 per 90', or '90%' for ratios (not divided by minutes)."""
    text = format_value(metric, value)
    return text if metric in RATIO_COLS or text == "n/a" else f"{text} per 90"


def pool_label(pool: str) -> str:
    return "All roles" if pool == "All" else POOL_NAMES[pool].capitalize()


def no_profile_reason(player_id: int, player: pd.Series, pct: pd.DataFrame) -> str | None:
    """Why the player has no profile (radar, percentiles, similar players), or None."""
    if player["pool"] == "GK":
        return ("Goalkeepers have no profile yet: goalkeeper-specific metrics "
                "(saves, claims, distribution) are not computed in this project. "
                "Pitch maps are available.")
    if not pct["player_id"].eq(player_id).any():
        # .1f: with .0f, 269.7 minutes would read "270 min — below the 270-minute threshold"
        return (f"{player['player']} played {player['minutes']:.1f} min — below the "
                f"{MIN_MINUTES}-minute threshold. Per-90 numbers on such a small sample "
                "are mostly noise, so there is no profile or similarity search. "
                "Pitch maps are available.")
    return None


def open_player(player_id: int) -> None:
    """Callback of the "Open profile" button. Runs before the rerun, when widget values
    can still be changed; filters are reset so the new player is in the list."""
    st.session_state["role"] = "All"
    st.session_state["team"] = "All"
    st.session_state["min_minutes"] = min(st.session_state.get("min_minutes", MIN_MINUTES),
                                          MIN_MINUTES)
    st.session_state["player"] = player_id
    st.session_state["tab"] = "Profile"


# ---------- sidebar ----------

def sidebar(totals: pd.DataFrame) -> int:
    """Filters + player choice, all bound to the URL. Returns player_id."""
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

        # Filters are plain boolean masks, as in the CLI scripts
        players = totals[totals["minutes"] >= min_minutes]
        if pool != "All":
            players = players[players["pool"] == pool]
        if team != "All":
            players = players[players["team"] == team]

        if players.empty:
            st.warning("No players match these filters. Try lowering the minimum minutes.")
            st.stop()

        players = players.sort_values("player")
        labels = dict(zip(players["player_id"], players["player"] + " (" + players["team"] + ")"))
        options = players["player_id"].tolist()     # plain Python ints, not numpy.int64

        # Default: DEFAULT_PLAYER if he passes the filters, otherwise the first in the list
        default = players["player"].str.startswith(DEFAULT_PLAYER)
        index = options.index(players.loc[default, "player_id"].iloc[0]) if default.any() else 0

        # Fuzzy search matches the label, which includes the team: "ukraine" lists all Ukrainians
        player_id = st.selectbox(f"Player ({len(options)})", options, index=index,
                                 format_func=labels.get, key="player", bind="query-params")

        st.divider()
        # Logo at the bottom as a "data provided by" credit, not in st.logo() at the top,
        # where it would look as if this were a StatsBomb app
        st.caption("Data provided by")
        st.image(str(STATSBOMB_LOGO), width=170, link=STATSBOMB_URL)
        st.caption(f"[StatsBomb Open Data]({STATSBOMB_URL}) · Code: [GitHub]({GITHUB_URL})")
    return player_id


# ---------- tabs ----------

def header(player: pd.Series) -> None:
    st.title(player["player"])
    parts = [player["team"], player["main_position"],
             f"{player['minutes']:.0f} min",
             f"{player['matches']} match" + ("es" if player["matches"] != 1 else "")]
    # 3 players without a single on-pitch event have no known position (NaN)
    st.caption(" · ".join(str(p) for p in parts if pd.notna(p)))


def metrics_table(raw: pd.Series, pct_row: pd.Series) -> pd.DataFrame:
    """All 22 metrics with value and percentile: exact numbers and every metric,
    not just the 10 on the radar (also an accessible alternative to the chart)."""
    return pd.DataFrame({
        "Metric": [LABELS[m] for m in LABELS],
        "Value": [format_value(m, raw[m]) for m in LABELS],
        "Percentile": [pct_row[m] for m in LABELS],
    })


def render_profile(player_id: int, player: pd.Series,
                   per90: pd.DataFrame, pct: pd.DataFrame) -> None:
    reason = no_profile_reason(player_id, player, pct)
    if reason:
        st.info(reason)   # explain why instead of showing an empty page
        return

    pct_row = pct.set_index("player_id").loc[player_id]
    raw = per90.set_index("player_id").loc[player_id]
    pool = pct_row["pool"]
    pool_size = int((pct["pool"] == pool).sum())

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.image(radar_png(player_id), width="stretch")

    with right:
        # Strengths and weak spots only among the 10 radar metrics of the role:
        # a forward's 95th percentile in clearances is not a strength
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
        height=(len(table) + 1) * 35 + 3,   # all rows without scrolling (35 px per row)
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
    """Pitch maps work for all 493 players: no minutes threshold, because a map shows
    what the player did, without dividing by 90."""
    kind = st.segmented_control("Map", list(MAPS), default="Shots", required=True,
                                key="map", bind="query-params", label_visibility="collapsed")
    open_play_only = False
    if kind == "Passes":
        open_play_only = st.toggle(
            "Open play only", key="open_play",
            help="Hide set pieces (corners, free kicks, throw-ins). Progressive passes are "
                 "open play anyway, so this changes key passes and the grey background.")

    # Width in pixels: the vertical shot map is narrower than the horizontal maps
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

    # on_select="rerun": selecting a row reruns the script and returns the selection.
    # The key includes the player, so a new player gets a fresh table with the default
    # selection (row 0, the closest match).
    # Short position (DM, CM, W...) instead of "Left Defensive Midfield": otherwise the
    # table does not fit and the most useful column (difference) scrolls out of view
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
    if not rows:   # the user cleared the selection
        st.info("Select a player in the table to compare.")
        return
    other = similar.iloc[rows[0]]
    compare_id = int(other["player_id"])

    st.subheader(f"{player['player']} vs {other['player']}")
    # Same as the table columns, for the selected player and always visible:
    # at ~1200 px the last column scrolls out of view
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
    # The "why within role" example is computed from the data
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


# ---------- page ----------

per90, totals, pct = load_tables()
player_id = sidebar(totals)
player = totals.set_index("player_id").loc[player_id]

header(player)

# Lazy tabs (on_change="rerun" + .open): only the open tab's code runs.
# st.tabs has no bind="query-params", so ?tab= is synced by hand. The starting tab is
# read from the URL once per session: `default` is part of the widget's identity, and
# changing it on every run would reset the user's choice.
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
