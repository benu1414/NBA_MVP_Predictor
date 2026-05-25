# NBA MVP Predictor Platform

This project started as a notebook-based NBA MVP predictor using Basketball Reference data and Ridge regression. It is being upgraded into a full analytics platform with a reusable ML pipeline, model comparison, FastAPI backend, and React dashboard.

## Current Capabilities

- Builds a cleaned player-season feature dataset from Basketball Reference CSV/HTML sources.
- Adds Basketball Reference advanced stats such as PER, TS%, WS, BPM, and VORP.
- Trains and backtests Ridge, Random Forest, Gradient Boosting, and a two-stage candidate/share model.
- Saves model metrics, backtested predictions, and a best-model artifact.
- Exposes predictions, actual MVP results, players, model metrics, QA checks, explanations, and what-if simulations through FastAPI.
- Provides a React dashboard for historical MVP race exploration, model comparison, QA, player profiles, explanations, and simulation.

## Build Plan

See [BUILD_SCRIPT.md](BUILD_SCRIPT.md) for the complete phased roadmap.

## ML Commands

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Build features:

```bash
python -m ml.features
```

Refresh Basketball Reference source data through 2026:

```bash
python -m ml.scrape --start-year 2022 --end-year 2026 --force
python -m ml.scrape --start-year 2003 --end-year 2026 --types advanced
python -m ml.clean --start-year 2003 --end-year 2026
python -m ml.features
python -m ml.qa
python -m ml.train
```

Train and evaluate models:

```bash
python -m ml.train
```

Predict a historical season:

```bash
python -m ml.predict --year 2021
```

## API

Start the API after training models:

```bash
uvicorn backend.app.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /seasons`
- `GET /players`
- `GET /players/{player_name}/seasons/{year}/explanation`
- `GET /seasons/{year}/predictions`
- `GET /seasons/{year}/actual-results`
- `GET /models`
- `GET /qa`
- `GET /narrative/status`
- `GET /narrative/schema`
- `POST /simulate`

FastAPI docs are available at `http://localhost:8000/docs`.

## Frontend

The dashboard lives in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

By default, it expects the API at `http://localhost:8000`. Set `VITE_API_BASE` to point it somewhere else.

## Narrative Layer

The narrative/media layer is scaffolded as an optional experiment. Create the starter CSV with:

```bash
python -m ml.narrative
```

See [docs/NARRATIVE_LAYER.md](docs/NARRATIVE_LAYER.md) for the feature schema and ablation-test plan.

After filling the narrative feature CSV, run:

```bash
python -m ml.narrative_experiment
```

## Testing

```bash
python -m unittest discover -s tests
```

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
