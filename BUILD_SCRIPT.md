# NBA MVP Platform Build Script

This file is the working script for turning the existing notebook project into a complete MVP analytics platform. It is intentionally phased so each milestone produces something runnable before the next layer is added.

## Product Goal

Build an NBA MVP analytics platform that:

- Produces reproducible historical MVP predictions.
- Compares baseline and stronger ML models.
- Exposes player, season, prediction, model, and simulation data through an API.
- Powers an interactive dashboard for exploring MVP races.
- Leaves room for a narrative/media layer that can be evaluated statistically instead of guessed into the model.

## Phase 1: Project Foundation

Goals:

- Preserve the original notebook/data work.
- Create a clean project structure for ML, backend, frontend, data, models, and docs.
- Convert notebook logic into reusable Python modules.

Deliverables:

- `ml/` package with feature, training, evaluation, and prediction modules.
- `data/processed/player_season_features.csv`.
- `models/` directory for trained model artifacts and metrics.
- Updated README with setup and run commands.

Acceptance criteria:

- A single command can build the feature dataset.
- A single command can train and evaluate models.
- Generated artifacts are saved outside notebooks.

## Phase 1B: Data Refresh Through 2026

Goals:

- Replace one-off notebook scraping with repeatable refresh commands.
- Download Basketball Reference MVP, player, and team pages through the current season.
- Normalize schema changes between older and newer Basketball Reference tables.
- Rebuild all downstream CSVs, features, predictions, and model artifacts.

Commands:

```bash
python -m ml.scrape --start-year 2022 --end-year 2026 --force
python -m ml.scrape --start-year 2003 --end-year 2026 --types advanced
python -m ml.clean --start-year 2003 --end-year 2026
python -m ml.features
python -m ml.qa
python -m ml.train
```

Acceptance criteria:

- `MVP_Web_Scraper/player_mvp_stats.csv` includes seasons 2003-2026.
- `data/processed/player_season_features.csv` includes seasons 2003-2026.
- `data/processed/predictions.csv` includes backtested predictions through 2026.
- API `GET /seasons` returns `2026`.

## Phase 2: Better ML Architecture

Goals:

- Keep Ridge regression as the baseline.
- Add stronger models for comparison.
- Evaluate the project as a ranking problem, not only a regression problem.

Models:

- Ridge regression baseline.
- Random forest regressor.
- Gradient boosting regressor.
- Two-stage candidate classifier plus vote-share regressor.
- Optional XGBoost/LightGBM later if installed.

Metrics:

- Mean squared error.
- Top-1 MVP winner accuracy.
- Top-3 recall.
- Top-5 average precision.
- Actual winner predicted rank.
- Spearman rank correlation when possible.

Acceptance criteria:

- `models/metrics.json` summarizes every trained model.
- `models/best_model.joblib` points to the selected model.
- `data/processed/predictions.csv` stores backtested predictions.

## Phase 2B: Advanced Stats and QA

Goals:

- Add Basketball Reference advanced stats to the model.
- Validate the refreshed dataset before training.
- Expose QA status and model explanations through the API and dashboard.

Deliverables:

- `MVP_Web_Scraper/advanced.csv`.
- `ml.qa` command and `data/processed/qa_report.json`.
- Advanced features including PER, TS%, WS, WS/48, BPM, VORP, and percentile/rank variants.
- Player-season explanation endpoint.

Acceptance criteria:

- QA report status is `pass`.
- `GET /qa` returns row counts, coverage, duplicates, missing team joins, and MVP vote counts.
- `GET /players/{player}/seasons/{year}/explanation` returns lift/drag factors.

## Phase 3: API Backend

Goals:

- Build a FastAPI backend around the processed data and trained model.
- Make the model usable by a dashboard or external clients.
- Include a simulation endpoint for what-if MVP cases.

Core endpoints:

- `GET /health`
- `GET /seasons`
- `GET /players`
- `GET /players/{player_name}`
- `GET /players/{player_name}/seasons/{year}`
- `GET /players/{player_name}/seasons/{year}/explanation`
- `GET /seasons/{year}/predictions`
- `GET /seasons/{year}/actual-results`
- `GET /models`
- `GET /models/{model_name}/metrics`
- `GET /qa`
- `POST /simulate`

Acceptance criteria:

- API starts locally with `uvicorn backend.app.main:app --reload`.
- Swagger docs are available at `/docs`.
- Dashboard can load season predictions from the API.

## Phase 4: Dashboard

Goals:

- Build a React dashboard that consumes the API.
- Make historical MVP races easy to inspect and compare.

Core views:

- Season leaderboard.
- Actual vs predicted rank table.
- Player profile cards.
- Model comparison panel.
- What-if simulator.
- Narrative/media experiment panel placeholder.

Acceptance criteria:

- User can pick a season and see predicted rankings.
- User can compare actual MVP voting with model ranks.
- User can run a basic simulation.

## Phase 5: Explainability

Goals:

- Explain why a player is ranked where they are.
- Start with model-agnostic feature deltas and Ridge coefficients.
- Add SHAP later if dependency/runtime allows.

Deliverables:

- `GET /players/{player_name}/seasons/{year}/explanation`.
- Dashboard explanation panel.

Acceptance criteria:

- Each top player has positive and negative contributing factors.

## Phase 6: Narrative/Media Layer

Goals:

- Add narrative features without leaking future information.
- Compare basketball-only models against basketball-plus-media models.

Candidate features:

- Media mention count.
- Search trend percentile.
- Reddit mention count.
- Sentiment score.
- Team surprise score.
- Breakout score.
- Prior MVP fatigue.

Statistical method:

- Aggregate narrative data by player-season or player-week.
- Normalize within season.
- Train two feature sets:
  - `basketball_only`
  - `basketball_plus_narrative`
- Compare with the same backtest metrics.

Acceptance criteria:

- Narrative features are optional and clearly flagged.
- Ablation metrics show whether narrative features help.

## Phase 7: Deployment and Automation

Goals:

- Deploy backend and frontend.
- Add scheduled updates.

Later deliverables:

- GitHub Actions or cron job for updates.
- Postgres database.
- Hosted API.
- Hosted dashboard.

Acceptance criteria:

- Fresh predictions can be regenerated with documented commands.
- Deployment instructions are repeatable.
