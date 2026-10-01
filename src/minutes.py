"""Minutes played and main position of every player, derived from the events.

A player comes on with the Starting XI (minute 0) or as a substitute, and goes off
when substituted, sent off, or at the final whistle.
Lineups from the StatsBomb API are not used: their position timeline is broken
in extra-time matches.
"""
import pandas as pd

# StatsBomb's clock restarts at a fixed minute in every period
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105}

RED_CARDS = ["Red Card", "Second Yellow"]

# 25 StatsBomb positions -> 8 groups. Metrics are only comparable within a role:
# percentiles and similar players are computed within (merged) groups.
POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "AM", "Left Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Midfield": "W", "Right Midfield": "W", "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "FW", "Left Center Forward": "FW", "Right Center Forward": "FW",
    "Secondary Striker": "FW",
}


def add_elapsed(events: pd.DataFrame) -> pd.DataFrame:
    """Add `elapsed`: real time since kick-off, in minutes.

    StatsBomb's `minute` restarts every period: first-half stoppage time runs
    45, 46, 47... and the second half starts at 45 again. So elapsed = length of all
    previous periods + time into the current one. The penalty shootout (period 5)
    is not playing time and is dropped.
    """
    ev = events[events["period"] < 5].copy()
    ev["in_period"] = ev["minute"] + ev["second"] / 60 - ev["period"].map(PERIOD_START)

    # A period lasts until its last event
    duration = ev.groupby(["match_id", "period"])["in_period"].max()
    offset = (duration.groupby(level="match_id").cumsum() - duration).rename("offset")

    ev = ev.join(offset, on=["match_id", "period"])
    ev["elapsed"] = ev["offset"] + ev["in_period"]
    return ev


def player_match_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match: when he came on, went off, minutes played."""
    ev = add_elapsed(events)
    match_end = ev.groupby("match_id")["elapsed"].max().rename("match_end")

    # Starters are listed in the Starting XI event (tactics -> lineup)
    starters = [
        {"match_id": row.match_id, "team": row.team,
         "player_id": p["player"]["id"], "player": p["player"]["name"], "on": 0.0}
        for row in ev[ev["type"] == "Starting XI"].itertuples()
        for p in row.tactics["lineup"]
    ]

    subs = ev[ev["type"] == "Substitution"]

    # The player in substitution_replacement comes on
    subs_on = subs[["match_id", "team", "substitution_replacement_id",
                    "substitution_replacement", "elapsed"]]
    subs_on.columns = ["match_id", "team", "player_id", "player", "on"]

    on = pd.concat([pd.DataFrame(starters), subs_on], ignore_index=True)

    # Going off: substituted or sent off
    is_red = (ev["foul_committed_card"].isin(RED_CARDS)
              | ev["bad_behaviour_card"].isin(RED_CARDS))
    off = pd.concat([subs, ev[is_red]])[["match_id", "player_id", "elapsed"]]
    # Several exit events (e.g. a red card on the bench after being substituted): take the first
    off = off.groupby(["match_id", "player_id"], as_index=False)["elapsed"].min()
    off = off.rename(columns={"elapsed": "off"})

    # Nobody took him off -> played until the final whistle
    pm = on.merge(off, on=["match_id", "player_id"], how="left")
    pm = pm.join(match_end, on="match_id")
    pm["off"] = pm["off"].fillna(pm["match_end"])
    pm["minutes"] = (pm["off"] - pm["on"]).clip(lower=0)
    pm["player_id"] = pm["player_id"].astype(int)
    return pm.drop(columns="match_end")


def main_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Main position = the position with most of the player's events.

    Every StatsBomb event carries the player's position at that moment,
    so in-game tactical shifts are taken into account.
    """
    counts = (events.dropna(subset=["player_id", "position"])
              .groupby(["player_id", "position"]).size().rename("n").reset_index())
    # Most frequent position per player
    main = counts.sort_values("n", ascending=False).drop_duplicates("player_id")
    main = main.rename(columns={"position": "main_position"})
    main["player_id"] = main["player_id"].astype(int)
    main["position_group"] = main["main_position"].map(POSITION_GROUPS)
    return main[["player_id", "main_position", "position_group"]]


def player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Tournament totals: one row per player."""
    pm = player_match_minutes(events)
    totals = pm.groupby(["player_id", "player", "team"], as_index=False).agg(
        minutes=("minutes", "sum"),
        matches=("match_id", "nunique"),
    )
    result = totals.merge(main_positions(events), on="player_id", how="left")
    result["minutes"] = result["minutes"].round(1)
    return result.sort_values("minutes", ascending=False, ignore_index=True)


if __name__ == "__main__":
    from src.data_loader import load_events
    from src.paths import PROCESSED

    table = player_minutes(load_events())
    out = PROCESSED / "player_minutes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    print(table.head(10).to_string())
    print(f"\nPlayers: {len(table)}. Saved: {out}")
