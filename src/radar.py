"""Радар гравця: перцентилі всередині групи порівняння (mplsoccer.Radar).

Запуск з кореня проєкту:
    python -m src.radar "kane"
    python -m src.radar "lamine yamal" --vs "saka"
PNG зберігається в reports/figures/.
"""
import argparse
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd
from mplsoccer import Radar, grid

from src.data_loader import PROJECT_ROOT
from src.percentiles import (LABELS, POOL_NAMES, RADAR_TEMPLATES,
                             find_player, load_per90, normalize_name, percentile_table)

FIGURES = PROJECT_ROOT / "reports" / "figures"

BG = "#FFFFFF"
RING_FILL, RING_EDGE = "#F2F2F2", "#D6D6D6"
PLAYER_COLOR = "#1A78CF"    # синій — основний гравець
COMPARE_COLOR = "#E4572E"   # помаранчево-червоний — гравець для порівняння
TEXT, MUTED = "#1F1F1F", "#7A7A7A"


def polygon_style(color: str) -> dict:
    """Напівпрозора заливка + суцільний контур.

    Прозорість задаємо в самому кольорі (RGBA), а не через alpha=...,
    бо alpha зробила б прозорим і контур. Контур потрібен, щоб при
    порівнянні було видно форму обох гравців, навіть коли одна фігура
    повністю всередині іншої.
    """
    return {"facecolor": to_rgba(color, 0.3), "edgecolor": color, "lw": 2.5}


def format_value(metric: str, value: float) -> str:
    """Сире значення для підпису: точність пасу — у відсотках, решта — 2 знаки."""
    if pd.isna(value):
        return "n/a"
    if metric == "pass_completion":
        return f"{value:.0%}"
    return f"{value:.2f}"


def slugify(name: str) -> str:
    """'Kylian Mbappé Lottin' -> 'kylian_mbappe_lottin' (для імені файлу)."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(name)).strip("_")


def plot_radar(per90: pd.DataFrame, pct: pd.DataFrame, player_id: int,
               compare_id: int | None = None) -> plt.Figure:
    """Радар одного гравця або двох гравців одного пулу. Повертає Figure."""
    pct = pct.set_index("player_id")
    raw = per90.set_index("player_id")

    pool = pct.loc[player_id, "pool"]
    if pool not in RADAR_TEMPLATES:
        raise ValueError(f"Для пулу {pool} радар не будуємо: немає воротарських метрик")
    if compare_id is not None and pct.loc[compare_id, "pool"] != pool:
        raise ValueError("Порівнювати можна лише гравців одного пулу: "
                         "перцентилі рахуються всередині пулу і між пулами не порівнюються")
    metrics = RADAR_TEMPLATES[pool]
    pool_size = (pct["pool"] == pool).sum()

    # --- підписи осей ---
    # Для одного гравця під назвою метрики пишемо сире значення на 90:
    # перцентиль каже "наскільки високо серед колег", сире число — "скільки саме".
    names = [textwrap.fill(LABELS[m], 14) for m in metrics]
    if compare_id is None:
        params = [f"{name}\n{format_value(m, raw.loc[player_id, m])}" for name, m in zip(names, metrics)]
    else:
        params = names

    # Усі осі мають однакову шкалу 0–100, бо значення — перцентилі.
    # 4 кільця -> межі кілець на 25, 50, 75 і 100-му перцентилі.
    k = len(metrics)
    radar = Radar(params, min_range=[0] * k, max_range=[100] * k,
                  num_rings=4, ring_width=1, center_circle_radius=1)

    # grid() від mplsoccer: три області — заголовок, радар, підпис знизу
    fig, axs = grid(figheight=10, grid_height=0.78, title_height=0.1, endnote_height=0.04,
                    title_space=0.02, endnote_space=0.04, grid_key="radar", axis=False)
    fig.set_facecolor(BG)
    ax = axs["radar"]
    radar.setup_axis(ax=ax, facecolor=BG)
    radar.draw_circles(ax=ax, facecolor=RING_FILL, edgecolor=RING_EDGE, lw=1)

    # NaN (немає даних) малюємо як 0, але підписуємо "n/a"
    pct_values = pct.loc[player_id, metrics].astype(float)
    values = pct_values.fillna(0).tolist()

    if compare_id is None:
        _, _, vertices = radar.draw_radar(
            values, ax=ax,
            kwargs_radar=polygon_style(PLAYER_COLOR),
            kwargs_rings={"facecolor": to_rgba(PLAYER_COLOR, 0.12)},
        )
        # Число перцентиля в кожній вершині — щоб не вгадувати "на око"
        for (x, y), v in zip(vertices, pct_values):
            ax.text(x, y, "n/a" if pd.isna(v) else f"{v:.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc=PLAYER_COLOR, ec="none"))
    else:
        compare_values = pct.loc[compare_id, metrics].astype(float).fillna(0).tolist()
        radar.draw_radar_compare(
            values, compare_values, ax=ax,
            kwargs_radar=polygon_style(PLAYER_COLOR),
            kwargs_compare=polygon_style(COMPARE_COLOR),
        )

    # wrap=None: переноси рядків робимо самі (textwrap.fill вище), бо вбудований
    # wrap від mplsoccer склеїв би наш "\n" перед значенням у пробіл.
    labels = radar.draw_param_labels(ax=ax, wrap=None, offset=1.25, fontsize=11, color=TEXT)
    for label in labels:
        label.set_rotation(0)   # горизонтальні підписи читати легше, ніж повернуті вздовж осі

    # --- заголовок ---
    t = axs["title"]

    def player_header(pid, x, ha, color, size):
        info = f"{raw.loc[pid, 'team']} · {raw.loc[pid, 'main_position']} · {raw.loc[pid, 'minutes']:.0f} min"
        t.text(x, 0.68, raw.loc[pid, "player"], fontsize=size, fontweight="bold",
               color=color, ha=ha, va="center")
        t.text(x, 0.22, info, fontsize=12, color=MUTED, ha=ha, va="center")

    if compare_id is None:
        player_header(player_id, 0.01, "left", PLAYER_COLOR, 22)
        t.text(0.99, 0.68, "Euro 2024", fontsize=16, fontweight="bold", color=TEXT,
               ha="right", va="center")
        t.text(0.99, 0.22, f"vs {pool_size} {POOL_NAMES[pool]}", fontsize=12,
               color=MUTED, ha="right", va="center")
    else:
        player_header(player_id, 0.01, "left", PLAYER_COLOR, 17)
        player_header(compare_id, 0.99, "right", COMPARE_COLOR, 17)

    # --- підпис знизу: як читати графік ---
    lines = [f"Percentile rank vs {pool_size} {POOL_NAMES[pool]} (270+ min, Euro 2024).",
             "Rings: 25th / 50th / 75th percentile."]
    if compare_id is None:
        lines[1] += " Values under labels: per 90 min (ratios as is), penalties excluded."
    axs["endnote"].text(0.01, 0.9, "\n".join(lines), fontsize=9, color=MUTED, ha="left", va="top")
    axs["endnote"].text(0.99, 0.9, "Data: StatsBomb Open Data", fontsize=9, color=MUTED,
                        ha="right", va="top")
    return fig


def save_figure(fig: plt.Figure, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)   # звільняємо пам'ять: у циклі по гравцях фігури накопичуються
    return str(path.relative_to(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Радар гравця Євро 2024 (перцентилі всередині ролі)")
    parser.add_argument("player", help="частина імені, напр. 'kane' або 'mbappe'")
    parser.add_argument("--vs", help="гравець для порівняння (з того ж пулу)")
    args = parser.parse_args()

    per90 = load_per90()
    pct = percentile_table(per90)
    try:
        player = find_player(pct, args.player)
        other = find_player(pct, args.vs) if args.vs else None
        fig = plot_radar(per90, pct, player["player_id"],
                         None if other is None else other["player_id"])
    except ValueError as e:
        parser.error(str(e))   # коротке повідомлення замість traceback

    name = slugify(player["player"])
    if other is not None:
        name += "_vs_" + slugify(other["player"])
    print("Збережено:", save_figure(fig, f"radar_{name}"))


if __name__ == "__main__":
    main()
