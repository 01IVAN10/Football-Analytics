"""Z-профіль двох гравців: де кожен вище чи нижче середнього по ролі.

Навіщо, якщо є радар: радар показує 10 метрик у перцентилях, а схожість
(src.similarity) рахується по 18 метриках у z-score. Тому пара з косинусом 0.7
на радарі може виглядати не дуже схожою: радар показує інші числа.
Z-профіль показує рівно те, що "бачить" алгоритм: той самий вектор з 18 чисел.

Як читати: 0 — середній гравець пулу, +1 — на одне стандартне відхилення вище.
Схожі гравці — ті, в кого точки по один бік від нуля в тих самих рядках
(форма профілю), навіть якщо один із них "гучніший" (далі від нуля).

Запуск з кореня проєкту:
    python -m src.z_profile "rodri" --vs "xhaka"
    python -m src.z_profile "yamal"               -> порівняння з найсхожішим
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

# Ті самі 18 метрик, що й FEATURES, згруповані за блоками (порядок збігається)
BLOCKS = {
    "Shooting": ["np_shots", "npxg"],
    "Creation": ["xa", "key_passes"],
    "Passing": ["passes", "pass_completion", "progressive_passes",
                "passes_final_third", "passes_into_box"],
    "Carrying": ["progressive_carries", "carries_into_box", "dribbles"],
    "Defending": ["tackles_won", "interceptions", "ball_recoveries",
                  "pressures", "clearances", "aerials_won"],
}
# Страховка: якщо колись змінимо FEATURES і забудемо тут — впаде одразу, а не мовчки
assert [m for ms in BLOCKS.values() for m in ms] == FEATURES

BLOCK_GAP = 0.8   # додатковий відступ між блоками (в "рядках")
OVERLAP_Z = 0.13  # ближче за це (в одиницях z) точки зливаються: діаметр точки ≈ 0.13 z
DODGE = 0.16      # тоді розводимо їх по вертикалі на ±0.16 рядка


def row_positions() -> dict[str, float]:
    """y-координата кожної метрики: зверху вниз, з проміжками між блоками."""
    y, positions = 0.0, {}
    for metrics in BLOCKS.values():
        for m in metrics:
            positions[m] = y
            y -= 1
        y -= BLOCK_GAP
    return positions


def plot_z_profile(per90: pd.DataFrame, player_id: int, compare_id: int) -> plt.Figure:
    """Dumbbell-графік: 18 метрик, дві точки на рядок (синя — гравець, помаранчева — порівняння)."""
    z = standardize(per90)
    for pid in (player_id, compare_id):
        if pid not in z.index:
            raise ValueError("Z-профіль будуємо лише для польових гравців з 270+ хв")
    pool = z.loc[player_id, "pool"]
    if z.loc[compare_id, "pool"] != pool:
        raise ValueError("Порівнювати можна лише гравців одного пулу: z-score рахується всередині пулу")

    pool_z = z[z["pool"] == pool]
    similarity = cosine_to(pool_z, player_id)[compare_id]
    za, zb = z.loc[player_id, FEATURES].astype(float), z.loc[compare_id, FEATURES].astype(float)

    # grid() як у радарі: заголовок / графік / підпис. ax_aspect — ширина до висоти області графіка
    fig, axs = grid(figheight=9, ax_aspect=1.45, grid_height=0.76, title_height=0.09,
                    endnote_height=0.05, title_space=0.02, endnote_space=0.06,
                    grid_key="plot", axis=False)
    fig.set_facecolor(BG)
    ax = axs["plot"]
    # Звужуємо область графіка зліва: там стоятимуть назви метрик і блоків
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left + width * 0.3, bottom, width * 0.7, height])
    ax.axis("on")

    ys = row_positions()
    y = [ys[m] for m in FEATURES]

    # Межі осі x симетричні: 0 (середнє) завжди посередині, мінімум ±3
    limit = max(3, math.ceil(max(za.abs().max(), zb.abs().max()) + 0.3))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(min(y) - 0.8, max(y) + 0.8)

    # Блоки: легка заливка через один + назва блоку ліворуч від назв метрик
    for i, (block, metrics) in enumerate(BLOCKS.items()):
        top, low = ys[metrics[0]] + 0.5, ys[metrics[-1]] - 0.5
        if i % 2 == 0:
            ax.axhspan(low, top, color=RING_FILL, zorder=0, lw=0)
        ax.text(-0.44, (top + low) / 2, block.upper(), transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=9, color=MUTED)

    # Сітка на цілих z: слабка, щоб не конкурувати з точками; нуль — темніший
    for x in range(-limit, limit + 1):
        ax.axvline(x, color=TEXT if x == 0 else LINES, lw=1.2 if x == 0 else 0.8, zorder=1)

    # "Гантель": сіра лінія між двома гравцями — довжина лінії і є різниця по метриці
    ax.hlines(y, za, zb, color=CONTEXT, lw=2.5, zorder=2)
    # Якщо значення майже однакові, синя точка повністю закрила б помаранчеву
    # (напр., виноси Родрі й Джаки: обидва +0.95). Тоді трохи розводимо їх по вертикалі.
    dodge = ((za - zb).abs() < OVERLAP_Z).to_numpy() * DODGE
    # Білий обідок навколо точок: при частковому перекритті видно межу обох
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
    parser = argparse.ArgumentParser(description="Z-профіль двох гравців Євро 2024")
    parser.add_argument("player", help="частина імені, напр. 'rodri' або 'kroos'")
    parser.add_argument("--vs", help="з ким порівняти (той самий пул); без --vs — найсхожіший")
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
    print("Збережено:", save_figure(fig, f"zprofile_{slugify(player['player'])}_vs_{slugify(other)}"))


if __name__ == "__main__":
    main()
