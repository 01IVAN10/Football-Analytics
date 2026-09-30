"""Карти гравця на полі: удари, паси, теплова карта дій (mplsoccer).

Запуск з кореня проєкту:
    python -m src.pitch_maps "kane"                  -> усі три карти
    python -m src.pitch_maps "kroos" --only passes   -> лише одна
PNG зберігаються в reports/figures/.

Карти беруть визначення з src.metrics (non_penalty_shots, classify_passes),
тож на карті рівно ті удари й паси, що пораховані в таблиці метрик.
Змінимо визначення прогресивного пасу — зміниться і метрика, і карта.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter

from src.metrics import FINAL_THIRD_X, classify_passes, non_penalty_shots, split_xy
from src.paths import PROCESSED
from src.percentiles import find_player
from src.style import (BG, BLUE, CONTEXT, HEAT_CMAP, LINES, MUTED, ORANGE, TEXT,
                       draw_endnote, draw_header, player_subtitle, save_figure, slugify)

TOTALS_PATH = PROCESSED / "player_totals.parquet"

# Поле StatsBomb 120x80 ярдів; усі команди атакують зліва направо
PITCH_STYLE = dict(pitch_type="statsbomb", pitch_color=BG, line_color=LINES, linewidth=1)
# Одна розкладка для всіх карт: заголовок / поле / підпис
GRID_STYLE = dict(title_height=0.1, title_space=0.02, grid_height=0.76,
                  endnote_height=0.06, endnote_space=0.05, axis=False)

SIZE_PER_XG = 1500   # площа кружка на карті ударів (pt²) на 1.0 xG


def load_totals() -> pd.DataFrame:
    """Усі гравці турніру (без порогу хвилин): карту ударів можна глянути й у запасного."""
    return pd.read_parquet(TOTALS_PATH)


def player_events(events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Усі події одного гравця за турнір, без серії пенальті."""
    return events[(events["player_id"] == player_id) & (events["period"] < 5)]


def legend_below(ax, handles: list, ncol: int, **kwargs) -> None:
    """Легенда одним рядком під полем."""
    style = dict(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False,
                 fontsize=10, labelcolor=TEXT, handletextpad=0.5, columnspacing=1.5)
    ax.legend(handles=handles, **(style | kwargs))


def plural(n: int, word: str) -> str:
    """plural(1, 'goal') -> '1 goal', plural(3, 'goal') -> '3 goals'."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------- карта ударів ----------

def plot_shot_map(ev: pd.DataFrame, player: pd.Series) -> plt.Figure:
    """Удари без пенальті на половині поля. Площа кружка = xG, заливка = гол."""
    shots = non_penalty_shots(ev)
    xy = split_xy(shots["location"])
    xg = shots["shot_statsbomb_xg"]
    goal = shots["shot_outcome"] == "Goal"

    # VerticalPitch(half=True): ворота зверху, видно атакувальну половину (x від 56).
    # Якщо гравець бив з власної половини (за турнір таких ударів 4, напр. Кіммих з x=48),
    # розширюємо поле вниз, щоб жоден удар не "зник" за межами картинки.
    min_x = xy["x"].min() if not shots.empty else 60
    pitch = VerticalPitch(half=True, pad_bottom=max(4, 60 - min_x + 4), **PITCH_STYLE)
    # Більший відступ під полем: у легенді є великий кружок "xG 0.5"
    fig, axs = pitch.grid(figheight=9, **(GRID_STYLE | dict(grid_height=0.73, endnote_space=0.08)))
    fig.set_facecolor(BG)
    ax = axs["pitch"]

    # s — площа маркера, тому площа пропорційна xG (а не діаметр).
    # Якби xG задавав діаметр, удар з xG 0.4 виглядав би в 4 рази "важчим" за 0.2, а не в 2.
    pitch.scatter(xy.loc[~goal, "x"], xy.loc[~goal, "y"], s=xg[~goal] * SIZE_PER_XG,
                  facecolor=to_rgba(BLUE, 0.15), edgecolor=BLUE, lw=1.5, ax=ax, zorder=3)
    # Голи — поверх, з білим обідком, щоб не зливались із сусідніми ударами
    pitch.scatter(xy.loc[goal, "x"], xy.loc[goal, "y"], s=xg[goal] * SIZE_PER_XG,
                  facecolor=ORANGE, edgecolor=BG, lw=2, ax=ax, zorder=4)
    if shots.empty:
        ax.text(40, 90, "No non-penalty shots", ha="center", va="center", fontsize=14, color=MUTED)

    # Легенда: колір (гол / не гол) + розмір (шкала xG)
    handles = [
        Line2D([], [], ls="", marker="o", ms=10, mfc=ORANGE, mec=BG, label="Goal"),
        Line2D([], [], ls="", marker="o", ms=10, mfc=to_rgba(BLUE, 0.15), mec=BLUE, label="No goal"),
    ]
    for v in (0.05, 0.2, 0.5):
        # ms у легенді — діаметр у pt, а s у scatter — площа, тому корінь
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


# ---------- карта пасів ----------

def plot_pass_map(ev: pd.DataFrame, player: pd.Series) -> plt.Figure:
    """Прогресивні та ключові паси стрілками, решта точних пасів — сірим фоном."""
    passes = ev[ev["type"] == "Pass"]
    flags = classify_passes(passes)
    start = split_xy(passes["location"])
    end = split_xy(passes["pass_end_location"])

    # Пас може бути і прогресивним, і ключовим одночасно. Малюємо обидва шари
    # (ключові — зверху), а в легенді показуємо ті самі числа, що й у метриках.
    key = flags["key"]
    progressive = flags["progressive"]
    other = flags["completed"] & ~progressive & ~key   # сірий фон — лише "звичайні" паси

    pitch = Pitch(**PITCH_STYLE)
    fig, axs = pitch.grid(figheight=8, **GRID_STYLE)
    fig.set_facecolor(BG)
    ax = axs["pitch"]

    def draw(mask, **kwargs):
        pitch.arrows(start.loc[mask, "x"], start.loc[mask, "y"], end.loc[mask, "x"], end.loc[mask, "y"],
                     ax=ax, **kwargs)

    # Контекст: де гравець взагалі віддає паси (тонкі лінії без стрілок — менше шуму)
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
                ("Pass map", f"{n} passes · {completion:.0%} completed"), left_color=BLUE)
    draw_endnote(axs["endnote"], [
        "Progressive: completed open-play pass, ball ≥25% closer to goal.",
        "Key pass: led directly to a shot (set pieces included), drawn on top. Attacking left → right.",
    ])
    return fig


# ---------- теплова карта ----------

def plot_heatmap(ev: pd.DataFrame, player: pd.Series) -> plt.Figure:
    """Де гравець діє: щільність усіх дій з координатами, згладжена."""
    # Carry (ведення) не беремо: воно починається в точці прийому м'яча,
    # і та сама точка порахувалась би двічі (Ball Receipt + Carry).
    actions = ev[ev["location"].notna() & (ev["type"] != "Carry")]
    xy = split_xy(actions["location"])

    # line_zorder=2 — лінії поля поверх кольору; поля зверху/знизу — під підписи третин і шкалу
    pitch = Pitch(**PITCH_STYLE, line_zorder=2, pad_top=8, pad_bottom=8)
    fig, axs = pitch.grid(figheight=8, **GRID_STYLE)
    fig.set_facecolor(BG)
    ax = axs["pitch"]

    # 1) Рахуємо дії в сітці 60x40 клітинок (кожна 2x2 ярди).
    # 2) Згладжуємо фільтром Гауса (sigma=2.5 клітинки ≈ 5 ярдів): кожна дія "розтікається"
    #    на сусідні клітинки, тож картина не залежить від того, де пройшла межа сітки,
    #    і не виглядає як шахівниця.
    stats = pitch.bin_statistic(xy["x"], xy["y"], statistic="count", bins=(60, 40))
    stats["statistic"] = gaussian_filter(stats["statistic"], sigma=2.5)
    mesh = pitch.heatmap(stats, ax=ax, cmap=HEAT_CMAP, zorder=1)

    # Частка дій у кожній третині поля — число, яке легко порівнювати між гравцями
    thirds = pd.cut(xy["x"], bins=[0, 40, FINAL_THIRD_X, 120], include_lowest=True,
                    labels=["Defensive third", "Middle third", "Final third"])
    shares = thirds.value_counts(normalize=True)
    for label, x in zip(shares.index.categories, (20, 60, 100)):
        ax.text(x, -3, f"{label}  {shares.get(label, 0):.0%}", ha="center", va="center",
                fontsize=11, color=TEXT)

    # Шкала кольору: значення після згладжування не є "кількістю дій",
    # тож показуємо лише напрямок (менше -> більше), без чисел
    cax = ax.inset_axes([0.8, 0.025, 0.16, 0.022])   # у нижньому відступі під полем
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
    parser = argparse.ArgumentParser(description="Карти гравця Євро 2024 на полі")
    parser.add_argument("player", help="частина імені, напр. 'kane' або 'mbappe'")
    parser.add_argument("--only", choices=list(MAPS), help="намалювати лише одну карту")
    args = parser.parse_args()

    # Імпорт тут, а не вгорі: data_loader тягне statsbombpy (клієнт API), а функції
    # малювання вище використовує і веб-застосунок, якому API не потрібен
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
        print("Збережено:", save_figure(fig, f"{name}_{slugify(player['player'])}"))


if __name__ == "__main__":
    main()
