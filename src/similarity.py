"""Пошук схожих гравців: z-score всередині пулу + косинусна подібність.

Ідея: кожен гравець — це вектор із 18 чисел (метрики на 90 хв).
Схожі гравці — ті, чиї вектори "дивляться в той самий бік".

Кроки:
  1. Беремо 18 метрик стилю (FEATURES) для гравців з 270+ хв.
  2. Нормалізуємо всередині пулу (z-score): скільки стандартних відхилень
     гравець вище/нижче середнього серед колег по ролі.
  3. Рахуємо косинусну подібність між вектором цільового гравця і всіма іншими
     з того ж пулу. 1 = профіль тієї ж форми, 0 = не пов'язані, -1 = протилежні.

Запуск з кореня проєкту:
    python -m src.similarity "kane"
    python -m src.similarity "rodri" -n 5 --other-teams --radar
"""
import argparse

import numpy as np
import pandas as pd

from src.percentiles import (COMPARISON_POOLS, LABELS, POOL_NAMES,
                             find_player, load_per90, percentile_table)

# 18 метрик "стилю": що гравець робить і як часто.
# Порядок: удари -> створення -> паси -> ведення -> оборона.
# Чого тут НЕМАЄ і чому:
#   - голи та асисти: результат, а не стиль; на 3–7 матчах це здебільшого удача.
#     Натомість беремо npxG і xA — якість моментів, які гравець отримує/створює.
#   - dribbles_completed: кореляція з dribbles 0.92 — фактично та сама інформація
#     двічі, тобто подвійна вага. Лишаємо спроби (dribbles): це і є стиль.
#   - npxg_per_shot, dribble_success: частки з NaN (немає ударів/обводок = немає даних).
#   - passes_completed: це passes × pass_completion, обидві вже є.
FEATURES = [
    "np_shots", "npxg",
    "xa", "key_passes",
    "passes", "pass_completion", "progressive_passes", "passes_final_third", "passes_into_box",
    "progressive_carries", "carries_into_box", "dribbles",
    "tackles_won", "interceptions", "ball_recoveries", "pressures", "clearances", "aerials_won",
]

INFO_COLS = ["player", "team", "main_position", "minutes"]


def standardize(per90: pd.DataFrame) -> pd.DataFrame:
    """Z-score кожної метрики всередині пулу. Індекс — player_id, плюс колонка pool.

    z = (значення - середнє по пулу) / стандартне відхилення по пулу

    Навіщо нормалізувати: метрики мають різні шкали (≈50 пасів проти ≈0.3 npxG).
    Без нормалізації подібність визначали б майже лише паси — найбільші числа.
    Після z-score кожна метрика в однакових "одиницях": +1 = на одне стандартне
    відхилення вище за середнього колегу по ролі.

    Чому всередині пулу: 5 відборів — це багато для вінгера і мало для опорника.
    Нас цікавить "чим гравець виділяється серед своєї ролі", а не "чим CB
    відрізняється від FW" — це й так очевидно.
    """
    df = per90.set_index("player_id")
    pool = df["position_group"].map(COMPARISON_POOLS)
    df = df[pool != "GK"]              # воротарських метрик немає (як і для радару)
    pool = pool[pool != "GK"]

    # transform повертає таблицю того ж розміру: кожне значення замінюється
    # на z-score, порахований по його пулу. std — вибіркове (ddof=1, як у pandas за замовчуванням).
    z = df[FEATURES].groupby(pool).transform(lambda col: (col - col.mean()) / col.std())
    # Якщо в пулі всі однакові (std = 0), ділення дасть NaN. Така метрика нікого
    # не розрізняє, тож 0 ("як усі") — чесне значення. На наших даних цього немає.
    z = z.fillna(0)
    z["pool"] = pool
    return z


def cosine_to(z: pd.DataFrame, player_id: int) -> pd.Series:
    """Косинусна подібність гравця player_id з кожним рядком z.

    cos(a, b) = (a · b) / (|a| · |b|)
    a · b — скалярний добуток (сума попарних добутків), |a| — довжина вектора.

    Чому косинус, а не відстань: косинус порівнює НАПРЯМОК вектора — "в яких
    метриках гравець вище/нижче середнього" — і не залежить від довжини, тобто від
    того, наскільки різко виражений профіль. Молодий вінгер, який робить те саме, що
    й зірка, але трохи менше, буде схожим. Для скаутингу це якраз те, що треба:
    шукаємо тип гравця, а не його копію за рівнем.
    Зворотний бік: у гравця, близького до середнього по всіх метриках, вектор
    короткий і його напрямок "хиткий" — схожість для нього менш надійна.
    """
    features = z[FEATURES]
    target = features.loc[player_id]
    # features @ target — скалярний добуток кожного рядка з target (одна операція на всю таблицю)
    dots = features @ target
    norms = np.linalg.norm(features, axis=1) * np.linalg.norm(target)
    return dots / norms


def explain(z: pd.DataFrame, a: int, b: int, k: int = 2) -> tuple[str, str]:
    """Пояснення, ЧОМУ два гравці схожі і в чому головна різниця.

    Спільне: косинус розкладається на внески метрик: внесок = z_a · z_b / (|a|·|b|),
    і сума всіх внесків = косинус. Найбільші внески, де обидва вище середнього, —
    це спільні сильні сторони.

    Різниця: метрика, де гравці по РІЗНІ боки від середнього (у одного z > 0,
    у іншого z < 0), з найбільшим розривом. Це різниця у стилі ("один робить це
    багато, інший — середньо або мало"), а не у ступені ("обидва багато, один більше").
    Такі метрики ніколи не перетинаються зі спільними (там обидва z > 0).
    """
    # z[FEATURES] спершу, потім .loc: так рядок лишається float. z.loc[a] дав би
    # тип object, бо в рядку є ще й текстова колонка pool.
    features = z[FEATURES]
    za, zb = features.loc[a], features.loc[b]
    contrib = za * zb      # знаменник однаковий для всіх метрик, для сортування не потрібен

    both_strong = contrib[(za > 0) & (zb > 0)].nlargest(k)
    shared = ", ".join(LABELS[m] for m in both_strong.index) or "—"

    gaps = (za - zb).abs()
    # contrib < 0 <=> знаки різні. Саме ці метрики дають від'ємний внесок
    # у косинус, тобто "тягнуть" схожість донизу.
    opposite = gaps[contrib < 0]
    if opposite.empty:
        # Усі 18 метрик з одного боку від середнього — на наших даних такого немає
        # (у всіх 850 парах топ-5 є протилежні), але код не має падати.
        # Тоді беремо найбільший розрив серед метрик, яких немає в "спільному".
        opposite = gaps.drop(both_strong.index)
    gap = opposite.idxmax()
    difference = f"{LABELS[gap]} ({za[gap]:+.1f} vs {zb[gap]:+.1f})"
    return shared, difference


def similar_players(per90: pd.DataFrame, player_id: int, n: int = 10,
                    other_teams: bool = False) -> pd.DataFrame:
    """Топ-n найсхожіших гравців з того ж пулу.

    other_teams=True — лише з інших збірних. Типове скаутське питання: "хто може
    замінити нашого гравця" — партнери по команді тут не відповідь.
    Ми перевіряли, чи не "склеює" метод партнерів через стиль команди: частка
    партнерів у топ-5 — 5.1% проти 2.9% випадково, тобто ефект є, але слабкий.
    """
    z = standardize(per90)
    if player_id not in z.index:
        raise ValueError("Для воротарів пошук схожих не працює: немає воротарських метрик")

    info = per90.set_index("player_id")[INFO_COLS]
    pool_z = z[z["pool"] == z.loc[player_id, "pool"]]

    sim = cosine_to(pool_z, player_id).drop(player_id)   # сам із собою завжди 1.0
    result = info.loc[sim.index].assign(similarity=sim)
    if other_teams:
        result = result[result["team"] != info.loc[player_id, "team"]]
    result = result.nlargest(n, "similarity")

    reasons = [explain(z, player_id, other) for other in result.index]
    result["shared"] = [shared for shared, _ in reasons]
    result["difference"] = [diff for _, diff in reasons]
    return result.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Схожі гравці Євро 2024 (всередині ролі)")
    parser.add_argument("player", help="частина імені, напр. 'kane' або 'rodri'")
    parser.add_argument("-n", type=int, default=10, help="скільки гравців показати (10)")
    parser.add_argument("--other-teams", action="store_true", help="лише з інших збірних")
    parser.add_argument("--radar", action="store_true",
                        help="зберегти радар-порівняння з найсхожішим гравцем")
    args = parser.parse_args()

    per90 = load_per90()
    try:
        player = find_player(per90, args.player)
        result = similar_players(per90, player["player_id"], args.n, args.other_teams)
    except ValueError as e:
        parser.error(str(e))

    pool = COMPARISON_POOLS[player["position_group"]]
    pool_size = (per90["position_group"].map(COMPARISON_POOLS) == pool).sum()
    print(f"\nСхожі на {player['player']} ({player['team']}, {player['main_position']}, "
          f"{player['minutes']:.0f} хв)")
    print(f"Пул: {pool_size} {POOL_NAMES[pool]}, {len(FEATURES)} метрик на 90, "
          f"z-score + косинусна подібність\n")

    table = result.assign(minutes=result["minutes"].round(0).astype(int),
                          similarity=result["similarity"].round(2))
    table.index = table.index + 1
    print(table[["player", "team", "main_position", "minutes", "similarity",
                 "shared", "difference"]].to_string())

    # Орієнтир: у половини гравців турніру найкращий збіг ≥ 0.57 (див. stage4-decisions)
    if not result.empty and result["similarity"].iloc[0] < 0.3:
        print("\nНавіть найкращий збіг слабкий (< 0.3): профіль гравця унікальний для цього пулу.")

    if args.radar and not result.empty:
        from src.radar import plot_radar           # імпорт тут: matplotlib потрібен лише з --radar
        from src.style import save_figure, slugify
        best = result.iloc[0]
        fig = plot_radar(per90, percentile_table(per90), player["player_id"], best["player_id"])
        name = f"radar_{slugify(player['player'])}_vs_{slugify(best['player'])}"
        print("\nЗбережено:", save_figure(fig, name))


if __name__ == "__main__":
    main()
