"""Shared chart style: colours, header, footnote, saving.

The radar, the pitch maps and the app should look like one system.
"""
import io
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.paths import FIGURES, PROJECT_ROOT
from src.percentiles import normalize_name


# --- palette ---
# Blue and orange stay distinguishable with colour blindness
# (palette validator: ΔE 24.7 under protanopia, threshold 8).
# Text is always neutral: colour only encodes "whose / which category".
BG = "#FFFFFF"
TEXT = "#1F1F1F"
MUTED = "#52514E"
LINES = "#D6D6D6"       # pitch lines, radar ring borders
RING_FILL = "#F2F2F2"   # radar ring fill
CONTEXT = "#C9C9C6"     # background data: other passes etc.
BLUE = "#2A78D6"        # main player / main category
ORANGE = "#EB6834"      # comparison player / highlight (goals, key passes)

# Sequential single-hue scale for heatmaps, starting at the background colour:
# empty zones blend into the pitch instead of drawing attention.
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat_blue", [BG, "#CDE2FB", "#86B6EF", "#3987E5", "#256ABF", "#104281"])


def player_subtitle(row) -> str:
    """'England · Center Forward · 635 min'; skips a missing position."""
    parts = [row["team"], row["main_position"], f"{row['minutes']:.0f} min"]
    return " · ".join(str(p) for p in parts if pd.notna(p))


def draw_header(ax, left: tuple[str, str], right: tuple[str, str],
                left_color: str = TEXT, right_color: str = TEXT,
                left_size: int = 22, right_size: int = 16, sub_size: float = 12) -> None:
    """Two-column header: (title, subtitle) on the left and on the right.
    Use a smaller sub_size for two players, so long subtitles do not collide."""
    for (title, sub), x, ha, color, size in [
        (left, 0.01, "left", left_color, left_size),
        (right, 0.99, "right", right_color, right_size),
    ]:
        ax.text(x, 0.68, title, fontsize=size, fontweight="bold", color=color, ha=ha, va="center")
        ax.text(x, 0.22, sub, fontsize=sub_size, color=MUTED, ha=ha, va="center")


def draw_endnote(ax, lines: list[str]) -> None:
    """Footnote: how to read the chart + data source."""
    ax.text(0.01, 0.9, "\n".join(lines), fontsize=9, color=MUTED, ha="left", va="top")
    ax.text(0.99, 0.9, "Data: StatsBomb Open Data", fontsize=9, color=MUTED, ha="right", va="top")


def slugify(name: str) -> str:
    """'Kylian Mbappé Lottin' -> 'kylian_mbappe_lottin' (file names)."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(name)).strip("_")


def _savefig(fig: plt.Figure, target) -> None:
    """Same save settings for files and for the app. target: a path or a BytesIO."""
    fig.savefig(target, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)   # figures pile up in memory when looping over players


def save_figure(fig: plt.Figure, name: str) -> str:
    """Save a PNG to reports/figures and close the figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    _savefig(fig, path)
    return str(path.relative_to(PROJECT_ROOT))


def figure_to_png(fig: plt.Figure) -> bytes:
    """PNG bytes for Streamlit: cheap to cache, unlike a Figure."""
    buffer = io.BytesIO()
    _savefig(fig, buffer)
    return buffer.getvalue()
