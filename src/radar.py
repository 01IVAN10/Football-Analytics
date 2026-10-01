"""Player radar: percentiles within the comparison pool (mplsoccer.Radar).

Run from the project root (PNG goes to reports/figures/):
    python -m src.radar "kane"
    python -m src.radar "lamine yamal" --vs "saka"
"""
import argparse
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd
from mplsoccer import Radar, grid

from src.percentiles import (LABELS, POOL_NAMES, RADAR_TEMPLATES,
                             find_player, load_per90, percentile_table)
from src.style import (BG, BLUE, LINES, ORANGE, RING_FILL, TEXT, draw_endnote, draw_header,
                       player_subtitle, save_figure, slugify)


def polygon_style(color: str) -> dict:
    """Translucent fill with a solid outline.

    Transparency goes into the RGBA fill colour, not alpha=..., which would also fade
    the outline. The outline keeps both shapes visible when one polygon lies entirely
    inside the other.
    """
    return {"facecolor": to_rgba(color, 0.3), "edgecolor": color, "lw": 2.5}


def format_value(metric: str, value: float) -> str:
    """Raw value for the label: pass completion as a percentage, the rest with 2 decimals."""
    if pd.isna(value):
        return "n/a"
    if metric == "pass_completion":
        return f"{value:.0%}"
    return f"{value:.2f}"


def plot_radar(per90: pd.DataFrame, pct: pd.DataFrame, player_id: int,
               compare_id: int | None = None) -> plt.Figure:
    """Radar of one player, or of two players from the same pool."""
    pct = pct.set_index("player_id")
    raw = per90.set_index("player_id")

    pool = pct.loc[player_id, "pool"]
    if pool not in RADAR_TEMPLATES:
        raise ValueError(f"No radar for pool {pool}: there are no goalkeeping metrics")
    if compare_id is not None and pct.loc[compare_id, "pool"] != pool:
        raise ValueError("Only players from the same pool can be compared: "
                         "percentiles are computed within a pool")
    metrics = RADAR_TEMPLATES[pool]
    pool_size = (pct["pool"] == pool).sum()

    # --- axis labels ---
    # For a single player the raw per-90 value goes under the metric name:
    # the percentile says "how high among peers", the raw number says "how much".
    names = [textwrap.fill(LABELS[m], 14) for m in metrics]
    if compare_id is None:
        params = [f"{name}\n{format_value(m, raw.loc[player_id, m])}" for name, m in zip(names, metrics)]
    else:
        params = names

    # All axes are percentiles, 0-100; 4 rings = 25th, 50th, 75th, 100th percentile
    k = len(metrics)
    radar = Radar(params, min_range=[0] * k, max_range=[100] * k,
                  num_rings=4, ring_width=1, center_circle_radius=1)

    # mplsoccer grid: title, radar and endnote areas
    fig, axs = grid(figheight=10, grid_height=0.78, title_height=0.1, endnote_height=0.04,
                    title_space=0.02, endnote_space=0.04, grid_key="radar", axis=False)
    fig.set_facecolor(BG)
    ax = axs["radar"]
    radar.setup_axis(ax=ax, facecolor=BG)
    radar.draw_circles(ax=ax, facecolor=RING_FILL, edgecolor=LINES, lw=1)

    # NaN (no data) is drawn as 0 but labelled "n/a"
    pct_values = pct.loc[player_id, metrics].astype(float)
    values = pct_values.fillna(0).tolist()

    if compare_id is None:
        _, _, vertices = radar.draw_radar(
            values, ax=ax,
            kwargs_radar=polygon_style(BLUE),
            kwargs_rings={"facecolor": to_rgba(BLUE, 0.12)},
        )
        # Percentile in every vertex, so nobody has to estimate it by eye
        for (x, y), v in zip(vertices, pct_values):
            ax.text(x, y, "n/a" if pd.isna(v) else f"{v:.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc=BLUE, ec="none"))
    else:
        compare_values = pct.loc[compare_id, metrics].astype(float).fillna(0).tolist()
        radar.draw_radar_compare(
            values, compare_values, ax=ax,
            kwargs_radar=polygon_style(BLUE),
            kwargs_compare=polygon_style(ORANGE),
        )

    # wrap=None: lines are already wrapped with textwrap.fill above; mplsoccer's own
    # wrapping would turn our "\n" before the value into a space
    labels = radar.draw_param_labels(ax=ax, wrap=None, offset=1.25, fontsize=11, color=TEXT)
    for label in labels:
        label.set_rotation(0)   # horizontal labels are easier to read than rotated ones

    # --- header and endnote ---
    p = raw.loc[player_id]
    if compare_id is None:
        draw_header(axs["title"], (p["player"], player_subtitle(p)),
                    ("Euro 2024", f"vs {pool_size} {POOL_NAMES[pool]}"), left_color=BLUE)
    else:
        c = raw.loc[compare_id]
        draw_header(axs["title"], (p["player"], player_subtitle(p)),
                    (c["player"], player_subtitle(c)),
                    left_color=BLUE, right_color=ORANGE, left_size=17, right_size=17, sub_size=10)

    lines = [f"Percentile rank vs {pool_size} {POOL_NAMES[pool]} (270+ min, Euro 2024).",
             "Rings: 25th / 50th / 75th percentile."]
    if compare_id is None:
        lines[1] += " Values under labels: per 90 min (ratios as is), penalties excluded."
    draw_endnote(axs["endnote"], lines)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Euro 2024 player radar (percentiles within role)")
    parser.add_argument("player", help="part of the name, e.g. 'kane' or 'mbappe'")
    parser.add_argument("--vs", help="player to compare with (same pool)")
    args = parser.parse_args()

    per90 = load_per90()
    pct = percentile_table(per90)
    try:
        player = find_player(pct, args.player)
        other = find_player(pct, args.vs) if args.vs else None
        fig = plot_radar(per90, pct, player["player_id"],
                         None if other is None else other["player_id"])
    except ValueError as e:
        parser.error(str(e))   # short message instead of a traceback

    name = slugify(player["player"])
    if other is not None:
        name += "_vs_" + slugify(other["player"])
    print("Saved:", save_figure(fig, f"radar_{name}"))


if __name__ == "__main__":
    main()
