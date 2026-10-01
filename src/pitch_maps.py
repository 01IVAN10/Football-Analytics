"""Pitch maps of a player: shots, passes, action heatmap (mplsoccer).

Run from the project root (PNGs go to reports/figures/):
    python -m src.pitch_maps "kane"                  -> all three maps
    python -m src.pitch_maps "kroos" --only passes   -> one map

The maps use the definitions from src.metrics (non_penalty_shots, classify_passes),
so they show exactly the shots and passes counted in the metric tables.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter

from src.metrics import FINAL_THIRD_X, classify_passes, non_penalty_shots, open_play, split_xy
from src.paths import PROCESSED
from src.percentiles import find_player
from src.style import (BG, BLUE, CONTEXT, HEAT_CMAP, LINES, MUTED, ORANGE, TEXT,
                       draw_endnote, draw_header, player_subtitle, save_figure, slugify)

TOTALS_PATH = PROCESSED / "player_totals.parquet"

# StatsBomb pitch, 120x80 yards; every team attacks left to right
PITCH_STYLE = dict(pitch_type="statsbomb", pitch_color=BG, line_color=LINES, linewidth=1)
# One layout for all maps: title / pitch / endnote
GRID_STYLE = dict(title_height=0.1, title_space=0.02, grid_height=0.76,
                  endnote_height=0.06, endnote_space=0.05, axis=False)

SIZE_PER_XG = 1500   # shot map marker area (pt²) per 1.0 xG


def load_totals() -> pd.DataFrame:
    """All players, no minutes threshold: a squad player's shot map is still useful."""
    return pd.read_parquet(TOTALS_PATH)


def player_events(events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """All events of one player, shootout excluded."""
    return events[(events["player_id"] == player_id) & (events["period"] < 5)]


def legend_below(ax, handles: list, ncol: int, **kwargs) -> None:
    """One-row legend under the pitch."""
    style = dict(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False,
                 fontsize=10, labelcolor=TEXT, handletextpad=0.5, columnspacing=1.5)
    ax.legend(handles=handles, **(style | kwargs))


def plural(n: int, word: str) -> str:
    """plural(1, 'goal') -> '1 goal', plural(3, 'goal') -> '3 goals'."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------- shot map ----------

def plot_shot_map(ev: pd.DataFrame, player: pd.Series) -> plt.Figure:
    """Non-penalty shots on half a pitch. Marker area = xG, filled = goal."""
    shots = non_penalty_shots(ev)
    xy = split_xy(shots["location"])
    xg = shots["shot_statsbomb_xg"]
    goal = shots["shot_outcome"] == "Goal"

    # Half pitch, goal at the top. Four shots of the tournament came from the player's own
    # half (e.g. Kimmich from x=48): extend the pitch down so no shot falls off the image.
    min_x = xy["x"].min() if not shots.empty else 60
    pitch = VerticalPitch(half=True, pad_bottom=max(4, 60 - min_x + 4), **PITCH_STYLE)
    # More room under the pitch: the legend has a large "xG 0.5" marker
    fig, axs = pitch.grid(figheight=9, **(GRID_STYLE | dict(grid_height=0.73, endnote_space=0.08)))
    fig.set_facecolor(BG)
    ax = axs["pitch"]

    # s is the marker AREA, so area is proportional to xG. Scaling the diameter would make
    # an xG 0.4 shot look 4 times bigger than 0.2 instead of 2.
    pitch.scatter(xy.loc[~goal, "x"], xy.loc[~goal, "y"], s=xg[~goal] * SIZE_PER_XG,
                  facecolor=to_rgba(BLUE, 0.15), edgecolor=BLUE, lw=1.5, ax=ax, zorder=3)
    # Goals on top, with a white edge so they stand out from nearby shots
    pitch.scatter(xy.loc[goal, "x"], xy.loc[goal, "y"], s=xg[goal] * SIZE_PER_XG,
                  facecolor=ORANGE, edgecolor=BG, lw=2, ax=ax, zorder=4)
    if shots.empty:
        ax.text(40, 90, "No non-penalty shots", ha="center", va="center", fontsize=14, color=MUTED)

    # Legend: colour (goal / no goal) + size (xG scale)
    handles = [
        Line2D([], [], ls="", marker="o", ms=10, mfc=ORANGE, mec=BG, label="Goal"),
        Line2D([], [], ls="", marker="o", ms=10, mfc=to_rgba(BLUE, 0.15), mec=BLUE, label="No goal"),
    ]
    for v in (0.05, 0.2, 0.5):
        # Legend ms is a diameter in pt while scatter s is an area: hence the square root
        handles.append(Line2D([], [], ls="", marker="o", ms=(v * SIZE_PER_XG) ** 0.5,
                              mfc="none", mec=MUTED, label=f"xG {v}"))
    legend_below(ax, handles, ncol=5, handlelength=2.5, handletextpad=0.8)

    n, goals, total_xg = len(shots), int(goal.sum()), xg.sum()
    draw_header(axs["title"], (player["player"], player_subtitle(player)),
                ("Shot map", f"{plural(n, 'shot')} · {plural(goals, 'goal')} · {total_xg:.2f} npxG"),
                left_color=BLUE)
    draw_endnote(axs["endnote"], ["Non-penalty shots. Circle area = StatsBomb xG.",
                                  "Attacking upwards."])
    return fig


# ---------- pass map ----------

def plot_pass_map(ev: pd.DataFrame, player: pd.Series, open_play_only: bool = False) -> plt.Figure:
    """Progressive and key passes as arrows, other completed passes as grey context.

    open_play_only=True drops set pieces. Progressive passes are open play anyway, so this
    changes key passes (Kroos has many from corners), the grey context and the completion.
    The legend may then show fewer key passes than the key_passes metric, which includes
    set pieces.
    """
    passes = ev[ev["type"] == "Pass"]
    if open_play_only:
        passes = passes[open_play(passes)]
    flags = classify_passes(passes)
    start = split_xy(passes["location"])
    end = split_xy(passes["pass_end_location"])

    # A pass can be both progressive and key: both layers are drawn (key on top) and the
    # legend shows the same numbers as the metrics
    key = flags["key"]
    progressive = flags["progressive"]
    other = flags["completed"] & ~progressive & ~key

    pitch = Pitch(**PITCH_STYLE)
    fig, axs = pitch.grid(figheight=8, **GRID_STYLE)
    fig.set_facecolor(BG)
    ax = axs["pitch"]

    def draw(mask, **kwargs):
        pitch.arrows(start.loc[mask, "x"], start.loc[mask, "y"], end.loc[mask, "x"], end.loc[mask, "y"],
                     ax=ax, **kwargs)

    # Context: where the player passes at all (thin lines without heads, less noise)
    pitch.lines(start.loc[other, "x"], start.loc[other, "y"], end.loc[other, "x"], end.loc[other, "y"],
                color=CONTEXT, lw=0.6, ax=ax, zorder=1)
    draw(progressive, color=BLUE, width=1.5, headwidth=4, headlength=4, zorder=3)
    draw(key, color=ORANGE, width=2, headwidth=4, headlength=4, zorder=4)

    handles = [
        Line2D([], [], color=BLUE, lw=2.5, label=f"Progressive ({int(progressive.sum())})"),
        Line2D([], [], color=ORANGE, lw=2.5, label=f"Key pass ({int(key.sum())})"),
        Line2D([], [], color=CONTEXT, lw=2.5, label=f"Other completed ({int(other.sum())})"),
    ]
    legend_below(ax, handles, ncol=3)

    n = len(passes)
    completion = flags["completed"].mean() if n else float("nan")
    draw_header(axs["title"], (player["player"], player_subtitle(player)),
                ("Pass map" + (" · open play" if open_play_only else ""),
                 f"{n} passes · {completion:.0%} completed"), left_color=BLUE)
    set_pieces = "set pieces excluded" if open_play_only else "set pieces included"
    draw_endnote(axs["endnote"], [
        "Progressive: completed open-play pass, ball ≥25% closer to goal.",
        f"Key pass: led directly to a shot ({set_pieces}), drawn on top. Attacking left → right.",
    ])
    return fig


# ---------- heatmap ----------

def plot_heatmap(ev: pd.DataFrame, player: pd.Series) -> plt.Figure:
    """Where the player acts: smoothed density of all actions with a location."""
    # No carries: a carry starts at the reception point, which would count twice
    # (Ball Receipt + Carry)
    actions = ev[ev["location"].notna() & (ev["type"] != "Carry")]
    xy = split_xy(actions["location"])

    # Pitch lines above the colour; padding for the thirds labels and the colour bar
    pitch = Pitch(**PITCH_STYLE, line_zorder=2, pad_top=8, pad_bottom=8)
    fig, axs = pitch.grid(figheight=8, **GRID_STYLE)
    fig.set_facecolor(BG)
    ax = axs["pitch"]

    # Count actions on a 60x40 grid (2x2-yard cells), then apply a Gaussian filter
    # (sigma 2.5 cells ≈ 5 yards) so the picture does not depend on where the cell
    # borders fall and does not look like a chessboard.
    stats = pitch.bin_statistic(xy["x"], xy["y"], statistic="count", bins=(60, 40))
    stats["statistic"] = gaussian_filter(stats["statistic"], sigma=2.5)
    mesh = pitch.heatmap(stats, ax=ax, cmap=HEAT_CMAP, zorder=1)

    # Share of actions per third: a number that is easy to compare between players
    thirds = pd.cut(xy["x"], bins=[0, 40, FINAL_THIRD_X, 120], include_lowest=True,
                    labels=["Defensive third", "Middle third", "Final third"])
    shares = thirds.value_counts(normalize=True)
    for label, x in zip(shares.index.categories, (20, 60, 100)):
        ax.text(x, -3, f"{label}  {shares.get(label, 0):.0%}", ha="center", va="center",
                fontsize=11, color=TEXT)

    # After smoothing the values are not "numbers of actions", so the colour bar
    # only shows the direction (fewer -> more), without numbers
    cax = ax.inset_axes([0.8, 0.025, 0.16, 0.022])   # in the padding under the pitch
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    cax.text(-0.03, 0.5, "fewer actions", transform=cax.transAxes, ha="right", va="center",
             fontsize=9, color=MUTED)
    cax.text(1.03, 0.5, "more", transform=cax.transAxes, ha="left", va="center",
             fontsize=9, color=MUTED)

    draw_header(axs["title"], (player["player"], player_subtitle(player)),
                ("Heatmap", f"{len(actions)} actions"), left_color=BLUE)
    draw_endnote(axs["endnote"], [
        "All actions with a location, incl. pressures and duels.",
        "Carries excluded (they start at the reception point). Smoothed ≈5 yd. Attacking left → right.",
    ])
    return fig


MAPS = {"shots": plot_shot_map, "passes": plot_pass_map, "heatmap": plot_heatmap}


def main() -> None:
    parser = argparse.ArgumentParser(description="Euro 2024 player pitch maps")
    parser.add_argument("player", help="part of the name, e.g. 'kane' or 'mbappe'")
    parser.add_argument("--only", choices=list(MAPS), help="draw only one map")
    args = parser.parse_args()

    # Imported here: data_loader pulls in statsbombpy, and the app that uses the
    # plotting functions above does not need the API client
    from src.data_loader import load_events

    try:
        player = find_player(load_totals(), args.player)
    except ValueError as e:
        parser.error(str(e))

    ev = player_events(load_events(), player["player_id"])
    for name, plot in MAPS.items():
        if args.only and name != args.only:
            continue
        fig = plot(ev, player)
        print("Saved:", save_figure(fig, f"{name}_{slugify(player['player'])}"))


if __name__ == "__main__":
    main()
