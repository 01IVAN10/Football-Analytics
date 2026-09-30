"""Перевірка пошуку схожих гравців: тест «упізнай себе».

Проблема: "правильних відповідей" немає — ніхто не розмітив, хто на кого схожий.
Тому перевіряємо непряму, але необхідну властивість: хороший метод має впізнати
гравця за його ж іншими матчами. Якщо не впізнає — він ловить шум, а не стиль.

Схема:
  1. Матчі кожного гравця ділимо через один: 1-й, 3-й, 5-й... -> половина A,
     2-й, 4-й, 6-й... -> половина B. Через один, а не "перші/останні", щоб в обидві
     половини потрапили і груповий етап, і плей-оф.
  2. Для кожної половини рахуємо ті самі метрики на 90 -> два незалежні профілі.
  3. Для профілю A шукаємо найсхожіші серед усіх профілів B того ж пулу.
     На першому місці має бути сам гравець. Потім навпаки: B -> A.
  4. Рахуємо: як часто "я" на 1-му місці (top-1), у топ-3, і середній ранг.

Кожен рядок таблиці змінює ОДНЕ рішення відносно вибраного методу,
тож видно, що саме дає кожне рішення.

Запуск з кореня проєкту:
    python -m src.validate_similarity
"""
import numpy as np
import pandas as pd

from src.data_loader import load_events, load_matches
from src.metrics import COUNT_COLS, add_ratios, player_totals
from src.minutes import player_match_minutes
from src.percentiles import COMPARISON_POOLS, RADAR_TEMPLATES, load_per90
from src.similarity import FEATURES

# Щонайменше один повний матч у кожній половині: на 20 хвилинах профіль — шум
MIN_HALF_MINUTES = 90


# ---------- 1. два профілі на гравця ----------

def match_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Метрики окремо в кожному матчі: один рядок = гравець × матч.

    Використовуємо той самий player_totals, що й для турніру, просто
    викликаємо його на подіях одного матчу. Визначення метрик не дублюється.
    """
    per_match = []
    for match_id, ev in events.groupby("match_id"):
        totals = player_totals(ev).reset_index()
        totals["match_id"] = match_id
        per_match.append(totals)
    return pd.concat(per_match, ignore_index=True)


def half_profiles(events: pd.DataFrame, matches: pd.DataFrame,
                  per90: pd.DataFrame) -> pd.DataFrame:
    """Два профілі на 90 для кожного гравця: half = 0 (матчі 1, 3, 5...) і 1 (2, 4, 6...).

    Лишаємо тих самих гравців, що й у пошуку (270+ хв, не воротарі),
    у яких в обох половинах щонайменше MIN_HALF_MINUTES.
    """
    totals = match_totals(events)
    metric_cols = [c for c in totals.columns if c not in ("player_id", "match_id")]

    # Перевірка розбиття: сума по матчах має дорівнювати підсумку за турнір.
    # Якщо ні — розбиття на матчі щось губить (напр., xA, де пас і удар у різних частинах).
    tournament = player_totals(events)
    summed = totals.groupby("player_id")[metric_cols].sum().loc[tournament.index]
    assert np.allclose(summed, tournament[metric_cols]), "сума по матчах ≠ турнір"

    minutes = player_match_minutes(events)[["match_id", "player_id", "minutes"]]
    df = minutes.merge(totals, on=["match_id", "player_id"], how="left").fillna(0)
    df = df[df["minutes"] > 0]

    # Хронологічний порядок матчів гравця. Дата і час — рядки ISO ("2024-06-15"),
    # тому звичайне сортування рядків = сортування в часі.
    df = df.merge(matches[["match_id", "match_date", "kick_off"]], on="match_id")
    df = df.sort_values(["player_id", "match_date", "kick_off"])
    # cumcount нумерує матчі гравця 0, 1, 2, 3...; % 2 перетворює на 0, 1, 0, 1...
    df["half"] = df.groupby("player_id").cumcount() % 2

    halves = df.groupby(["player_id", "half"], as_index=False)[["minutes"] + metric_cols].sum()
    halves = add_ratios(halves)          # точність пасів тощо — з сум, до переводу на 90
    halves[COUNT_COLS] = halves[COUNT_COLS].div(halves["minutes"], axis=0) * 90

    # Пул беремо з турнірної таблиці: роль гравця — за весь турнір, а не за половину
    pools = per90.set_index("player_id")["position_group"].map(COMPARISON_POOLS)
    halves["pool"] = halves["player_id"].map(pools)
    halves = halves[halves["pool"].notna() & (halves["pool"] != "GK")]

    by_player = halves.groupby("player_id")
    enough = ((by_player["half"].transform("nunique") == 2)
              & (by_player["minutes"].transform("min") >= MIN_HALF_MINUTES))
    return halves[enough].reset_index(drop=True)


# ---------- 2. складники методу, які порівнюємо ----------
# Нормалізація: приймає таблицю профілів одного пулу, повертає таблицю того ж розміру.

def zscore(X: pd.DataFrame) -> pd.DataFrame:
    return (X - X.mean()) / X.std()


def percentile(X: pd.DataFrame) -> pd.DataFrame:
    # -0.5 центрує: 0 = медіана пулу, як 0 = середнє у z-score. Без цього всі значення
    # додатні, кут між будь-якими векторами < 90° і косинус у всіх високий.
    return X.rank(pct=True) - 0.5


def no_scaling(X: pd.DataFrame) -> pd.DataFrame:
    return X


# Подібність: матриця S, де S[i, j] = подібність профілю A_i до профілю B_j.

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Ділимо кожен рядок на його довжину -> вектори довжини 1.
    # Тоді скалярний добуток = косинус, і A @ B.T рахує всі пари за одну операцію.
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def euclidean_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A[:, None, :] має форму (n, 1, 18), B[None, :, :] — (1, m, 18).
    # Різниця "розтягується" (broadcasting) до (n, m, 18): усі пари одразу.
    # Мінус — щоб "більше = схожіше", як у косинуса.
    return -np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


# Набори ознак: функція від пулу, бо метрики радару для кожного пулу свої.

def features_18(pool: str) -> list[str]:
    return FEATURES


def radar_10(pool: str) -> list[str]:
    return RADAR_TEMPLATES[pool]


def with_outcomes(pool: str) -> list[str]:
    return FEATURES + ["np_goals", "assists"]


# (назва, нормалізація, подібність, ознаки). Перший рядок — вибраний метод
# (як у src.similarity), кожен наступний змінює рівно одне рішення.
VARIANTS = [
    ("ВИБРАНИЙ: z-score + косинус, 18 метрик", zscore, cosine_matrix, features_18),
    ("евклідова відстань замість косинуса", zscore, euclidean_similarity, features_18),
    ("перцентилі замість z-score", percentile, cosine_matrix, features_18),
    ("без нормалізації", no_scaling, cosine_matrix, features_18),
    ("10 метрик радару замість 18", zscore, cosine_matrix, radar_10),
    ("18 метрик + голи й асисти", zscore, cosine_matrix, with_outcomes),
]


# ---------- 3. тест ----------

def trial_ranks(profiles: pd.DataFrame, normalize, similarity, features_for) -> pd.DataFrame:
    """Прогоняє тест для одного варіанту методу. Один рядок = одна спроба.

    rank  — скільки ЧУЖИХ профілів виявились схожішими за власний (0 = впізнав себе першим);
    score — той самий ранг у шкалі 0..1: 1 = перший, 0 = останній, 0.5 = як навмання.
    """
    trials = []
    for pool, group in profiles.groupby("pool"):
        # Сортування гарантує однаковий порядок гравців в A і B:
        # i-й рядок A і i-й рядок B — той самий гравець.
        group = group.sort_values(["player_id", "half"])
        # Нормалізуємо A і B РАЗОМ, щоб обидві половини були в одній шкалі.
        # fillna(0): NaN (npxG за удар без ударів, std = 0) = "як середній".
        X = normalize(group[features_for(pool)]).fillna(0).to_numpy()
        is_b = (group["half"] == 1).to_numpy()
        A, B = X[~is_b], X[is_b]
        players = group.loc[~is_b, "player_id"].to_numpy()

        for S in (similarity(A, B), similarity(B, A)):
            own = np.diag(S)                        # S[i, i] — гравець зі своєю ж іншою половиною
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
    """Що дав би випадковий вибір: у пулі з n гравців шанс вгадати себе першим = 1/n."""
    sizes = profiles.groupby("pool")["player_id"].nunique().to_numpy()
    per_trial = np.repeat(sizes, sizes)       # розмір пулу для кожного гравця
    return pd.Series({"top1": (1 / per_trial).mean(),
                      "top3": (3 / per_trial).mean(),
                      "mean_rank": 0.5})


def paired_bootstrap(chosen: pd.DataFrame, other: pd.DataFrame,
                     n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Різниця mean_rank (вибраний − інший) і її 95% довірчий інтервал.

    Навіщо: різниця 0.747 проти 0.742 може бути просто везінням на цих 167 гравцях.
    Бутстреп відповідає: "якби турнір зіграли інші, схожі гравці, чи лишилась би різниця?"

    Як:
      1. Для кожного гравця — середня різниця score двох методів (по його двох спробах).
         ПАРНИЙ: обидва методи на тих самих гравцях, тож "легкі" і "важкі" для
         впізнання гравці не додають шуму — віднімається лише ефект методу.
      2. 2000 разів випадково вибираємо 167 гравців З ПОВЕРНЕННЯМ (хтось двічі,
         хтось жодного) і рахуємо середню різницю.
      3. 2.5-й і 97.5-й перцентилі цих 2000 середніх = 95% інтервал.
    Вибираємо гравців, а не спроби: дві спроби одного гравця (A→B і B→A) пов'язані.
    Якщо інтервал не містить 0 — різниця реальна. seed фіксований -> результат відтворюється.
    """
    assert (chosen["player_id"] == other["player_id"]).all(), "спроби мають іти в одному порядку"
    diff = (chosen["score"] - other["score"]).groupby(chosen["player_id"]).mean().to_numpy()
    rng = np.random.default_rng(seed)
    # Матриця n_boot × 167 випадкових індексів: одразу всі 2000 вибірок, без циклу
    samples = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))]
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return diff.mean(), low, high


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "вибраний краще"
    if high < 0:
        return "вибраний гірше"
    return "різниця = шум"


def formatted(stats: pd.Series) -> dict:
    """Числа -> рядки для друку. Головна метрика — mean_rank, тому вона перша."""
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
            row["Δ mean_rank [95% ДІ]"] = f"{delta:+.3f} [{low:+.3f}, {high:+.3f}]"
            row["висновок"] = verdict(low, high)
        rows[name] = row
    rows["випадковий вибір"] = formatted(random_baseline(profiles))
    # orient="index": ключі словника -> рядки. Порожні клітинки (у вибраного і
    # випадкового немає порівняння) -> "".
    table = pd.DataFrame.from_dict(rows, orient="index").fillna("")

    print(f"\nТест «упізнай себе»: гравців {n_players}, спроб {2 * n_players} (A→B і B→A)")
    print(f"Пули: {profiles.groupby('pool')['player_id'].nunique().to_dict()}\n")
    print(table.to_string())
    print("\nmean_rank: 1 = завжди впізнає себе першим, 0.5 = як навмання. Це головна метрика:")
    print("вона враховує місце в кожній спробі. top-1 / top-3 наочні, але шумні (±4 п.п.).")


if __name__ == "__main__":
    main()
