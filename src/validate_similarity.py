"""Validation of the similar-player search: the "recognise yourself" test.

There is no ground truth for who plays like whom, so the test checks a necessary
property: a good method recognises a player from his other matches.
Each player's matches are split alternately into halves A and B (so both mix group
stage and knockouts); every A profile is ranked against all B profiles of the pool and
vice versa. Each variant changes exactly ONE decision of the chosen method; a paired
bootstrap tells whether the difference is real.

Run from the project root (needs data/raw):
    python -m src.validate_similarity
"""
import numpy as np
import pandas as pd

from src.data_loader import load_events, load_matches
from src.metrics import COUNT_COLS, add_ratios, player_totals
from src.minutes import player_match_minutes
from src.percentiles import COMPARISON_POOLS, RADAR_TEMPLATES, load_per90
from src.similarity import FEATURES

# At least one full match in each half: a 20-minute profile is noise
MIN_HALF_MINUTES = 90


# ---------- 1. two profiles per player ----------

def match_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Metrics per match: one row per player per match.

    Calls the same player_totals as the tournament table, on one match at a time,
    so metric definitions are not duplicated.
    """
    per_match = []
    for match_id, ev in events.groupby("match_id"):
        totals = player_totals(ev).reset_index()
        totals["match_id"] = match_id
        per_match.append(totals)
    return pd.concat(per_match, ignore_index=True)


def half_profiles(events: pd.DataFrame, matches: pd.DataFrame,
                  per90: pd.DataFrame) -> pd.DataFrame:
    """Two per-90 profiles per player: half 0 (matches 1, 3, 5...) and 1 (2, 4, 6...).

    Same players as the search (270+ minutes, no goalkeepers), with at least
    MIN_HALF_MINUTES in each half.
    """
    totals = match_totals(events)
    metric_cols = [c for c in totals.columns if c not in ("player_id", "match_id")]

    # The per-match split must add up to the tournament totals; otherwise something gets
    # lost between matches (e.g. xA, where the pass and the shot are separate events)
    tournament = player_totals(events)
    summed = totals.groupby("player_id")[metric_cols].sum().loc[tournament.index]
    assert np.allclose(summed, tournament[metric_cols]), "sum over matches ≠ tournament"

    minutes = player_match_minutes(events)[["match_id", "player_id", "minutes"]]
    df = minutes.merge(totals, on=["match_id", "player_id"], how="left").fillna(0)
    df = df[df["minutes"] > 0]

    # Chronological order of each player's matches (ISO date and time strings sort correctly)
    df = df.merge(matches[["match_id", "match_date", "kick_off"]], on="match_id")
    df = df.sort_values(["player_id", "match_date", "kick_off"])
    # cumcount numbers each player's matches 0, 1, 2, 3...; % 2 -> 0, 1, 0, 1...
    df["half"] = df.groupby("player_id").cumcount() % 2

    halves = df.groupby(["player_id", "half"], as_index=False)[["minutes"] + metric_cols].sum()
    halves = add_ratios(halves)          # ratios from the sums, before converting to per 90
    halves[COUNT_COLS] = halves[COUNT_COLS].div(halves["minutes"], axis=0) * 90

    # The pool comes from the tournament table: a player's role over the whole tournament
    pools = per90.set_index("player_id")["position_group"].map(COMPARISON_POOLS)
    halves["pool"] = halves["player_id"].map(pools)
    halves = halves[halves["pool"].notna() & (halves["pool"] != "GK")]

    by_player = halves.groupby("player_id")
    enough = ((by_player["half"].transform("nunique") == 2)
              & (by_player["minutes"].transform("min") >= MIN_HALF_MINUTES))
    return halves[enough].reset_index(drop=True)


# ---------- 2. building blocks being compared ----------
# Scaling: takes the profiles of one pool, returns a table of the same shape.

def zscore(X: pd.DataFrame) -> pd.DataFrame:
    return (X - X.mean()) / X.std()


def percentile(X: pd.DataFrame) -> pd.DataFrame:
    # -0.5 centres the median at 0, like the mean in a z-score. Otherwise all values are
    # positive, every angle is below 90° and every cosine is high.
    return X.rank(pct=True) - 0.5


def no_scaling(X: pd.DataFrame) -> pd.DataFrame:
    return X


# Similarity: matrix S with S[i, j] = similarity of profile A_i to profile B_j.

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Unit vectors: then the dot product is the cosine, and A @ B.T gives all pairs at once
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def euclidean_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Broadcasting (n, 1, 18) - (1, m, 18) -> (n, m, 18): all pairs at once.
    # Negative, so that larger = more similar, as with the cosine.
    return -np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


# Feature sets: a function of the pool, because each pool has its own radar metrics.

def features_18(pool: str) -> list[str]:
    return FEATURES


def radar_10(pool: str) -> list[str]:
    return RADAR_TEMPLATES[pool]


def with_outcomes(pool: str) -> list[str]:
    return FEATURES + ["np_goals", "assists"]


# (name, scaling, similarity, features). The first row is the chosen method
# (as in src.similarity); every other row changes exactly one decision.
VARIANTS = [
    ("CHOSEN: z-score + cosine, 18 metrics", zscore, cosine_matrix, features_18),
    ("Euclidean distance instead of cosine", zscore, euclidean_similarity, features_18),
    ("percentiles instead of z-score", percentile, cosine_matrix, features_18),
    ("no scaling", no_scaling, cosine_matrix, features_18),
    ("10 radar metrics instead of 18", zscore, cosine_matrix, radar_10),
    ("18 metrics + goals and assists", zscore, cosine_matrix, with_outcomes),
]


# ---------- 3. the test ----------

def trial_ranks(profiles: pd.DataFrame, normalize, similarity, features_for) -> pd.DataFrame:
    """Run the test for one variant. One row per trial.

    rank  - how many OTHER players' profiles are more similar than his own (0 = recognised first)
    score - the same rank on a 0..1 scale: 1 = first, 0 = last, 0.5 = random
    """
    trials = []
    for pool, group in profiles.groupby("pool"):
        # Sorted, so that row i of A and row i of B are the same player
        group = group.sort_values(["player_id", "half"])
        # A and B are scaled TOGETHER so both halves are on the same scale.
        # fillna(0): NaN (std = 0) = average.
        X = normalize(group[features_for(pool)]).fillna(0).to_numpy()
        is_b = (group["half"] == 1).to_numpy()
        A, B = X[~is_b], X[is_b]
        players = group.loc[~is_b, "player_id"].to_numpy()

        for S in (similarity(A, B), similarity(B, A)):
            own = np.diag(S)                        # S[i, i]: the player vs his own other half
            trials.append(pd.DataFrame({
                "player_id": players,
                "rank": (S > own[:, None]).sum(axis=1),
                "pool_size": len(S),
            }))

    trials = pd.concat(trials, ignore_index=True)
    trials["score"] = 1 - trials["rank"] / (trials["pool_size"] - 1)
    return trials


def summarize(trials: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "top1": (trials["rank"] == 0).mean(),
        "top3": (trials["rank"] < 3).mean(),
        "mean_rank": trials["score"].mean(),
    })


def random_baseline(profiles: pd.DataFrame) -> pd.Series:
    """What random choice would give: in a pool of n players, 1/n chance to be first."""
    sizes = profiles.groupby("pool")["player_id"].nunique().to_numpy()
    per_trial = np.repeat(sizes, sizes)       # pool size for every player
    return pd.Series({"top1": (1 / per_trial).mean(),
                      "top3": (3 / per_trial).mean(),
                      "mean_rank": 0.5})


def paired_bootstrap(chosen: pd.DataFrame, other: pd.DataFrame,
                     n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in mean rank score (chosen - other) with a 95% confidence interval.

    Paired: both methods are scored on the same players, so how easy a player is to
    recognise cancels out. Players are resampled, not trials, because a player's two
    trials (A->B, B->A) are not independent.
    """
    assert (chosen["player_id"] == other["player_id"]).all(), "trials must be in the same order"
    diff = (chosen["score"] - other["score"]).groupby(chosen["player_id"]).mean().to_numpy()
    rng = np.random.default_rng(seed)
    # n_boot × 167 random indices: all 2000 resamples at once, no loop
    samples = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))]
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return diff.mean(), low, high


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "chosen is better"
    if high < 0:
        return "chosen is worse"
    return "no difference (noise)"


def formatted(stats: pd.Series) -> dict:
    """Numbers -> strings for printing. mean_rank is the main metric, so it comes first."""
    return {"mean_rank": f"{stats['mean_rank']:.3f}",
            "top1": f"{stats['top1']:.1%}",
            "top3": f"{stats['top3']:.1%}"}


def main() -> None:
    per90 = load_per90()
    profiles = half_profiles(load_events(), load_matches(), per90)
    n_players = profiles["player_id"].nunique()

    trials = {name: trial_ranks(profiles, *method) for name, *method in VARIANTS}
    chosen_name = VARIANTS[0][0]

    rows = {}
    for name, t in trials.items():
        row = formatted(summarize(t))
        if name != chosen_name:
            delta, low, high = paired_bootstrap(trials[chosen_name], t)
            row["Δ mean_rank [95% CI]"] = f"{delta:+.3f} [{low:+.3f}, {high:+.3f}]"
            row["verdict"] = verdict(low, high)
        rows[name] = row
    rows["random choice"] = formatted(random_baseline(profiles))
    # Empty cells (the chosen and random rows have no comparison) -> ""
    table = pd.DataFrame.from_dict(rows, orient="index").fillna("")

    print(f"\n\"Recognise yourself\" test: {n_players} players, {2 * n_players} trials (A→B and B→A)")
    print(f"Pools: {profiles.groupby('pool')['player_id'].nunique().to_dict()}\n")
    print(table.to_string())
    print("\nmean_rank: 1 = always recognised first, 0.5 = random. This is the main metric:")
    print("it uses the rank in every trial. top-1 / top-3 are intuitive but noisy (±4 points).")


if __name__ == "__main__":
    main()
