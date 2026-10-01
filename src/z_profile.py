"""Z-profile of two players: where each is above or below the average of his role.

The radar shows 10 metrics as percentiles, while the similarity search (src.similarity)
uses 18 metrics as z-scores, so a pair with cosine 0.7 may not look alike on the radar.
The z-profile shows exactly the vector the algorithm compares.

Reading it: 0 = pool average, +1 = one standard deviation above. Similar players have
dots on the same side of zero in the same rows, even if one is further from zero.

Run from the project root:
    python -m src.z_profile "rodri" --vs "xhaka"
    python -m src.z_profile "yamal"               -> vs the most similar player
"""
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import grid

from src.percentiles import COMPARISON_POOLS, LABELS, POOL_NAMES, find_player, load_per90
from src.similarity import FEATURES, cosine_to, similar_players, standardize
from src.style import (BG, BLUE, CONTEXT, LINES, MUTED, ORANGE, RING_FILL, TEXT,
                       draw_endnote, draw_header, player_subtitle, save_figure, slugify)

# The 18 FEATURES grouped into blocks (same order)
BLOCKS = {
    "Shooting": ["np_shots", "npxg"],
    "Creation": ["xa", "key_passes"],
    "Passing": ["passes", "pass_completion", "progressive_passes",
                "passes_final_third", "passes_into_box"],
    "Carrying": ["progressive_carries", "carries_into_box", "dribbles"],
    "Defending": ["tackles_won", "interceptions", "ball_recoveries",
                  "pressures", "clearances", "aerials_won"],
}
# Fail loudly if FEATURES changes and this grouping is not updated
assert [m for ms in BLOCKS.values() for m in ms] == FEATURES

BLOCK_GAP = 0.8   # extra space between blocks, in rows
OVERLAP_Z = 0.13  # dots closer than this (in z) overlap: a dot is ≈0.13 z wide
DODGE = 0.16      # ...so they are moved apart vertically by ±0.16 rows


def row_positions() -> dict[str, float]:
    """y position of every metric: top to bottom, with gaps between blocks."""
    y, positions = 0.0, {}
    for metrics in BLOCKS.values():
        for m in metrics:
            positions[m] = y
            y -= 1
        y -= BLOCK_GAP
    return positions


def plot_z_profile(per90: pd.DataFrame, player_id: int, compare_id: int) -> plt.Figure:
    """Dumbbell chart: 18 metrics, two dots per row (blue = player, orange = comparison)."""
    z = standardize(per90)
    for pid in (player_id, compare_id):
        if pid not in z.index:
            raise ValueError("Z-profiles are only built for outfield players with 270+ minutes")
    pool = z.loc[player_id, "pool"]
    if z.loc[compare_id, "pool"] != pool:
        raise ValueError("Only players from the same pool can be compared: z-scores are computed within a pool")

    pool_z = z[z["pool"] == pool]
    similarity = cosine_to(pool_z, player_id)[compare_id]
    za, zb = z.loc[player_id, FEATURES].astype(float), z.loc[compare_id, FEATURES].astype(float)

    # Same layout as the radar: title / chart / endnote; ax_aspect = width / height of the chart
    fig, axs = grid(figheight=9, ax_aspect=1.45, grid_height=0.76, title_height=0.09,
                    endnote_height=0.05, title_space=0.02, endnote_space=0.06,
                    grid_key="plot", axis=False)
    fig.set_facecolor(BG)
    ax = axs["plot"]
    # Narrow the chart from the left to make room for metric and block names
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left + width * 0.3, bottom, width * 0.7, height])
    ax.axis("on")

    ys = row_positions()
    y = [ys[m] for m in FEATURES]

    # Symmetric x axis: 0 (average) always in the middle, at least ±3
    limit = max(3, math.ceil(max(za.abs().max(), zb.abs().max()) + 0.3))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(min(y) - 0.8, max(y) + 0.8)

    # Blocks: shading on every other block + block name left of the metric names
    for i, (block, metrics) in enumerate(BLOCKS.items()):
        top, low = ys[metrics[0]] + 0.5, ys[metrics[-1]] - 0.5
        if i % 2 == 0:
            ax.axhspan(low, top, color=RING_FILL, zorder=0, lw=0)
        ax.text(-0.44, (top + low) / 2, block.upper(), transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=9, color=MUTED)

    # Light grid on whole z values; zero is darker
    for x in range(-limit, limit + 1):
        ax.axvline(x, color=TEXT if x == 0 else LINES, lw=1.2 if x == 0 else 0.8, zorder=1)

    # Grey line between the players: its length is the difference in that metric
    ax.hlines(y, za, zb, color=CONTEXT, lw=2.5, zorder=2)
    # Nearly equal values: the blue dot would hide the orange one completely
    # (e.g. Rodri and Xhaka clearances, both +0.95), so move them apart vertically
    dodge = ((za - zb).abs() < OVERLAP_Z).to_numpy() * DODGE
    # White edge so both dots stay visible when they partly overlap
    ax.scatter(zb, y - dodge, s=110, color=ORANGE, edgecolor=BG, lw=1.5, zorder=3)
    ax.scatter(za, y + dodge, s=110, color=BLUE, edgecolor=BG, lw=1.5, zorder=4)

    ax.set_yticks(y, [LABELS[m] for m in FEATURES], fontsize=11, color=TEXT)
    ax.set_xticks(range(-limit, limit + 1),
                  [f"{x:+d}" if x else "0" for x in range(-limit, limit + 1)], fontsize=10, color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("← below pool average          z-score          above pool average →",
                  fontsize=10, color=MUTED, labelpad=8)

    a, b = per90.set_index("player_id").loc[player_id], per90.set_index("player_id").loc[compare_id]
    draw_header(axs["title"], (a["player"], player_subtitle(a)), (b["player"], player_subtitle(b)),
                left_color=BLUE, right_color=ORANGE, left_size=17, right_size=17)
    draw_endnote(axs["endnote"], [
        f"Cosine similarity {similarity:.2f}: shape of the two profiles across these 18 metrics.",
        f"z-score within {len(pool_z)} {POOL_NAMES[pool]} (270+ min, Euro 2024), per 90, penalties excluded.",
    ])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-profile of two Euro 2024 players")
    parser.add_argument("player", help="part of the name, e.g. 'rodri' or 'kroos'")
    parser.add_argument("--vs", help="player to compare with (same pool); default: the most similar")
    args = parser.parse_args()

    per90 = load_per90()
    try:
        player = find_player(per90, args.player)
        if args.vs:
            other_id = find_player(per90, args.vs)["player_id"]
        else:
            other_id = similar_players(per90, player["player_id"], n=1)["player_id"].iloc[0]
        fig = plot_z_profile(per90, player["player_id"], other_id)
    except ValueError as e:
        parser.error(str(e))

    other = per90.set_index("player_id").loc[other_id, "player"]
    print("Saved:", save_figure(fig, f"zprofile_{slugify(player['player'])}_vs_{slugify(other)}"))


if __name__ == "__main__":
    main()
