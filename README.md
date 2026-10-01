# Euro 2024 Scout

A scouting tool built on StatsBomb open event data for UEFA Euro 2024 (51 matches, 493 players).

**Live app: [euro2024-scout.streamlit.app](https://euro2024-scout.streamlit.app)**

- **Player profiles**: per-90 metrics and percentile radars, compared within role
  (6 comparison pools, players with 270+ minutes), strengths and weak spots.
- **Pitch maps** for every player: shots sized by xG, progressive and key passes
  (with an open-play filter), smoothed heatmaps.
- **Similar players**: 18-metric profiles z-scored within role and compared with cosine
  similarity. Validated with a split-half test: the method finds a player's own "other half"
  first in 17% of cases (random: 3%) and ranks it above ~75% of candidates on average.

Every view has a shareable URL, e.g.
[Toni Kroos' similar players](https://euro2024-scout.streamlit.app/?player=Toni+Kroos+%28Germany%29&tab=Similar+players).

> The app runs on Streamlit Community Cloud and falls asleep after 12 hours without visitors.
> If you see "This app has gone to sleep", wake it up and give it ~30 seconds.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
python -m pytest tests/ -q          # smoke tests of every page (Streamlit AppTest)
```

The app reads the slim data committed in `data/app/` (~1 MB). To rebuild everything from the StatsBomb API:

```bash
python -m src.data_loader   # download 51 matches -> data/raw
python -m src.metrics       # minutes, per-90 metrics -> data/processed
python -m src.app_data      # slim app data -> data/app (with a consistency check)
```

## Structure

| Module | What it does |
|---|---|
| `src/minutes.py` | Minutes played from events (substitutions, red cards, stoppage time) |
| `src/metrics.py` | 21 counting metrics + ratios, per 90 minutes |
| `src/percentiles.py` | Comparison pools and percentiles within role |
| `src/radar.py`, `src/pitch_maps.py`, `src/z_profile.py` | Charts (matplotlib + mplsoccer) |
| `src/similarity.py` | Similar players: z-scores + cosine similarity |
| `src/validate_similarity.py` | Split-half validation with a paired bootstrap |
| `streamlit_app.py` | Web app |

Each module also runs from the command line, e.g. `python -m src.radar "kane"` or
`python -m src.similarity "xhaka" --other-teams`.

## Data

<img src="assets/statsbomb_logo.png" alt="StatsBomb" width="170">

Data: [StatsBomb Open Data](https://github.com/hudl/open-data), used under their terms
(state the source and show the StatsBomb logo).
