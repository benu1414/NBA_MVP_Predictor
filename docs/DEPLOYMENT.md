# Deployment Notes

This project has two deployable services:

- FastAPI backend in `backend/`.
- Vite React frontend in `frontend/`.

## Backend

Install dependencies and build artifacts:

```bash
python -m pip install -r requirements.txt
python -m ml.features
python -m ml.qa
python -m ml.train
```

Run locally:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Start command for many hosted providers:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

## Frontend

Install and build:

```bash
cd frontend
npm install
npm run build
```

Set `VITE_API_BASE` to the deployed backend URL.

## Refresh Workflow

For a manual data refresh:

```bash
python -m ml.scrape --start-year 2022 --end-year 2026 --force
python -m ml.scrape --start-year 2003 --end-year 2026 --types advanced
python -m ml.clean --start-year 2003 --end-year 2026
python -m ml.features
python -m ml.qa
python -m ml.train
```

For automation, schedule that sequence weekly or monthly, then redeploy the backend artifacts.

