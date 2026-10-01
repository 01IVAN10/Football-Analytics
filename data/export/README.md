# Euro 2024 player dataset

Player-level metrics for UEFA Euro 2024 (51 matches), derived from
[StatsBomb Open Data](https://github.com/hudl/open-data). These are the numbers behind
the [Euro 2024 Scout](https://euro2024-scout.streamlit.app) app.

| File | Rows | What |
|---|---|---|
| `euro2024_player_totals.csv` | 493 | Every player who appeared: tournament totals |
| `euro2024_player_per90.csv` | 191 | Players with 270+ minutes: values per 90 minutes and percentiles |

Regenerate from the repository root: `python -m src.export_data`
(a test checks that the committed files match the code).

## Player columns

| Column | Meaning |
|---|---|
| `player_id` | StatsBomb player id |
| `player`, `team` | Full name as in StatsBomb data, national team |
| `main_position` | The position the player was listed at most often in the events |
| `position_group` | GK, CB, FB, DM, CM, AM, W or FW |
| `pool` | *(per-90 file)* Comparison pool for percentiles: GK, CB, FB, MF (DM+CM), AM/W (AM+W), FW |
| `matches`, `minutes` | Appearances and minutes actually played, including stoppage time and extra time, excluding penalty shootouts. A full match is ≈94–100 minutes |

## Metrics

In the per-90 file every counting metric is divided by minutes and multiplied by 90.
Ratios (`pass_completion`, `dribble_success`, `npxg_per_shot`) are the same in both files.
An empty ratio means no attempts (not 0%).

| Column | Definition |
|---|---|
| `np_goals`, `np_shots`, `npxg` | Goals, shots and StatsBomb xG, excluding penalties |
| `assists` | Passes that led directly to a goal |
| `key_passes` | Passes that led directly to a shot (set pieces included) |
| `xa` | Expected assists: xG of the shots that followed the player's passes |
| `passes`, `passes_completed` | All passes, completed passes |
| `progressive_passes` | Completed open-play passes that moved the ball at least 25% closer to the opponent's goal |
| `passes_final_third` | Completed open-play passes from outside into the final third |
| `passes_into_box` | Completed open-play passes from outside into the penalty area |
| `progressive_carries` | Carries that moved the ball at least 25% and at least 5 yards closer to goal |
| `carries_into_box` | Carries from outside into the penalty area |
| `dribbles`, `dribbles_completed` | Take-ons attempted and completed |
| `tackles_won` | Tackles that won the ball or put it out of play |
| `interceptions`, `ball_recoveries`, `pressures`, `clearances` | StatsBomb events of that type (failed recoveries excluded) |
| `aerials_won` | Headers won |
| `pass_completion`, `dribble_success` | Share of completed passes / take-ons (0–1) |
| `npxg_per_shot` | Average non-penalty xG per shot |
| `*_pct` | *(per-90 file)* Percentile (0–100) within the player's comparison pool: the share of players in the same pool with a lower or equal value (ties share the average rank) |

Open play = everything except corners, free kicks, throw-ins, goal kicks and kick-offs.

## Caveats

- 3–7 matches per player: per-90 values for players near the 270-minute threshold are noisy.
- Defensive metrics are not adjusted for possession: players in teams that defend a lot
  get more chances to tackle and intercept.
- Goalkeepers have no goalkeeping metrics; their rows contain outfield metrics only.

## Source and license

Data: [StatsBomb Open Data](https://github.com/hudl/open-data). If you use this dataset,
credit StatsBomb as the data source and follow
[their terms](https://github.com/hudl/open-data/blob/master/LICENSE.pdf).

<img src="../../assets/statsbomb_logo.png" alt="StatsBomb" width="170">
