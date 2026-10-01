"""Similar players: z-scores within the pool + cosine similarity.

Each player is a vector of 18 per-90 metrics (FEATURES), standardised within his
comparison pool. Similar players are those whose vectors point in the same direction:
1 = same profile shape, 0 = unrelated, -1 = opposite.
Validation of the method: src.validate_similarity.

Run from the project root:
    python -m src.similarity "kane"
    python -m src.similarity "rodri" -n 5 --other-teams --radar
"""
import argparse

import numpy as np
import pandas as pd

from src.percentiles import (COMPARISON_POOLS, LABELS, POOL_NAMES,
                             find_player, load_per90, percentile_table)

# 18 style metrics: what a player does and how often.
# Order: shooting -> creation -> passing -> carrying -> defending.
# Left out on purpose:
#   - goals and assists: outcomes, mostly luck over 3-7 matches; npxG and xA describe
#     the chances a player gets and creates instead.
#   - dribbles_completed: correlates 0.92 with dribbles, i.e. the same information
#     with double weight. Attempts are the style.
#   - npxg_per_shot, dribble_success: ratios with NaN (no shots / take-ons = no data).
#   - passes_completed: passes × pass_completion, both already included.
FEATURES = [
    "np_shots", "npxg",
    "xa", "key_passes",
    "passes", "pass_completion", "progressive_passes", "passes_final_third", "passes_into_box",
    "progressive_carries", "carries_into_box", "dribbles",
    "tackles_won", "interceptions", "ball_recoveries", "pressures", "clearances", "aerials_won",
]

INFO_COLS = ["player", "team", "main_position", "minutes"]


def standardize(per90: pd.DataFrame) -> pd.DataFrame:
    """Z-score of every metric within the pool. Index: player_id, plus a pool column.

    Without scaling, pass volume (≈50 vs ≈0.3 npxG) would decide the similarity.
    Within the pool, because 5 tackles is a lot for a winger and little for a holding
    midfielder.
    """
    df = per90.set_index("player_id")
    pool = df["position_group"].map(COMPARISON_POOLS)
    df = df[pool != "GK"]              # no goalkeeping metrics (same as the radar)
    pool = pool[pool != "GK"]

    z = df[FEATURES].groupby(pool).transform(lambda col: (col - col.mean()) / col.std())
    # std = 0 (everyone equal) gives NaN; such a metric separates nobody, so 0 = "average".
    # Does not happen on this data.
    z = z.fillna(0)
    z["pool"] = pool
    return z


def cosine_to(z: pd.DataFrame, player_id: int) -> pd.Series:
    """Cosine similarity of player_id with every row of z.

    Cosine compares the direction of the profile, not its length: a player who does the
    same things less often still counts as similar (a type of player, not his level).
    For a player close to average everywhere the direction is unstable.
    """
    features = z[FEATURES]
    target = features.loc[player_id]
    dots = features @ target
    norms = np.linalg.norm(features, axis=1) * np.linalg.norm(target)
    return dots / norms


def format_z(value: float) -> str:
    """+1.6, -1.1, but -0.04 rather than "-0.0": otherwise it is unclear why the metric
    counts as being on the other side of the average."""
    return f"{value:+.2f}" if abs(value) < 0.05 else f"{value:+.1f}"


def explain(z: pd.DataFrame, a: int, b: int, k: int = 2) -> tuple[str, str]:
    """Why two players are similar, and their main difference.

    Shared strengths: the largest per-metric contributions to the cosine (z_a·z_b)
    where both players are above average.
    Main difference: the largest gap among metrics where the players are on opposite
    sides of the average, i.e. a difference in style rather than in degree.
    """
    # Select FEATURES before .loc so the row stays float (the pool column is text)
    features = z[FEATURES]
    za, zb = features.loc[a], features.loc[b]
    contrib = za * zb      # the denominator is the same for every metric: not needed to rank

    both_strong = contrib[(za > 0) & (zb > 0)].nlargest(k)
    shared = ", ".join(LABELS[m] for m in both_strong.index) or "—"

    gaps = (za - zb).abs()
    # Opposite signs <=> negative contribution: these metrics pull the similarity down
    opposite = gaps[contrib < 0]
    if opposite.empty:
        # All 18 metrics on the same side of the average (never happens in the top-5 of
        # any player here): fall back to the largest gap outside the shared strengths
        opposite = gaps.drop(both_strong.index)
    gap = opposite.idxmax()
    difference = f"{LABELS[gap]} ({format_z(za[gap])} vs {format_z(zb[gap])})"
    return shared, difference


def similar_players(per90: pd.DataFrame, player_id: int, n: int = 10,
                    other_teams: bool = False) -> pd.DataFrame:
    """The n most similar players from the same pool.

    other_teams=True: "who could replace our player" is not answered by his teammates.
    """
    z = standardize(per90)
    if player_id not in z.index:
        raise ValueError("No similar players for goalkeepers: there are no goalkeeping metrics")

    info = per90.set_index("player_id")[INFO_COLS]
    pool_z = z[z["pool"] == z.loc[player_id, "pool"]]

    sim = cosine_to(pool_z, player_id).drop(player_id)   # always 1.0 with himself
    result = info.loc[sim.index].assign(similarity=sim)
    if other_teams:
        result = result[result["team"] != info.loc[player_id, "team"]]
    result = result.nlargest(n, "similarity")

    reasons = [explain(z, player_id, other) for other in result.index]
    result["shared"] = [shared for shared, _ in reasons]
    result["difference"] = [diff for _, diff in reasons]
    return result.reset_index()


def best_matches(per90: pd.DataFrame) -> pd.Series:
    """Each player's similarity to his closest match in the pool (median ≈0.57).

    A scale for reading the numbers: is 0.6 very similar or so-so?
    """
    z = standardize(per90)
    best = []
    for _, pool_z in z.groupby("pool"):
        x = pool_z[FEATURES].to_numpy()
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        cos = unit @ unit.T                  # cos[i, j] = similarity of players i and j
        np.fill_diagonal(cos, -np.inf)       # ignore each player's 1.0 with himself
        best.append(pd.Series(cos.max(axis=1), index=pool_z.index))
    return pd.concat(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Similar Euro 2024 players (within role)")
    parser.add_argument("player", help="part of the name, e.g. 'kane' or 'rodri'")
    parser.add_argument("-n", type=int, default=10, help="how many players to show (10)")
    parser.add_argument("--other-teams", action="store_true", help="only other national teams")
    parser.add_argument("--radar", action="store_true",
                        help="save a radar comparison with the most similar player")
    args = parser.parse_args()

    per90 = load_per90()
    try:
        player = find_player(per90, args.player)
        result = similar_players(per90, player["player_id"], args.n, args.other_teams)
    except ValueError as e:
        parser.error(str(e))

    pool = COMPARISON_POOLS[player["position_group"]]
    pool_size = (per90["position_group"].map(COMPARISON_POOLS) == pool).sum()
    print(f"\nSimilar to {player['player']} ({player['team']}, {player['main_position']}, "
          f"{player['minutes']:.0f} min)")
    print(f"Pool: {pool_size} {POOL_NAMES[pool]}, {len(FEATURES)} metrics per 90, "
          f"z-score + cosine similarity\n")

    table = result.assign(minutes=result["minutes"].round(0).astype(int),
                          similarity=result["similarity"].round(2))
    table.index = table.index + 1
    print(table[["player", "team", "main_position", "minutes", "similarity",
                 "shared", "difference"]].to_string())

    # For reference: half of the players have a best match of 0.57 or more
    if not result.empty and result["similarity"].iloc[0] < 0.3:
        print("\nEven the best match is weak (< 0.3): a unique profile in this pool.")

    if args.radar and not result.empty:
        from src.radar import plot_radar           # matplotlib is only needed with --radar
        from src.style import save_figure, slugify
        best = result.iloc[0]
        fig = plot_radar(per90, percentile_table(per90), player["player_id"], best["player_id"])
        name = f"radar_{slugify(player['player'])}_vs_{slugify(best['player'])}"
        print("\nSaved:", save_figure(fig, name))


if __name__ == "__main__":
    main()
