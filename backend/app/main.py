from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.services import data_service
from ml.narrative import NARRATIVE_SCHEMA, narrative_status


class SimulationRequest(BaseModel):
    player: str | None = None
    year: int | None = None
    pts: float | None = None
    ast: float | None = None
    trb: float | None = None
    reb: float | None = None
    wins: float | None = None
    losses: float | None = None
    games: float | None = None
    srs: float | None = None
    fg_pct: float | None = None
    three_pct: float | None = None
    ft_pct: float | None = None


app = FastAPI(
    title="NBA MVP Predictor API",
    description="Historical MVP predictions, model metrics, player data, and what-if simulations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/seasons")
def get_seasons() -> list[int]:
    return data_service.seasons()


@app.get("/players")
def get_players() -> list[dict]:
    return data_service.players()


@app.get("/players/{player_name}")
def get_player(player_name: str) -> list[dict]:
    history = data_service.player_history(player_name)
    if not history:
        raise HTTPException(status_code=404, detail="Player not found")
    return history


@app.get("/players/{player_name}/seasons/{year}")
def get_player_season(player_name: str, year: int) -> dict:
    row = data_service.player_season(player_name, year)
    if row is None:
        raise HTTPException(status_code=404, detail="Player season not found")
    return row


@app.get("/players/{player_name}/seasons/{year}/explanation")
def get_player_explanation(player_name: str, year: int) -> dict:
    explanation = data_service.explain_player_season(player_name, year)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return explanation


@app.get("/seasons/{year}/predictions")
def get_predictions(year: int, model: str | None = Query(default=None)) -> list[dict]:
    rows = data_service.season_predictions(year, model=model)
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions found for this season")
    return rows


@app.get("/seasons/{year}/actual-results")
def get_actual_results(year: int) -> list[dict]:
    rows = data_service.actual_results(year)
    if not rows:
        raise HTTPException(status_code=404, detail="No actual MVP results found for this season")
    return rows


@app.get("/models")
def get_models() -> dict:
    return data_service.metrics()


@app.get("/qa")
def get_qa_report() -> dict:
    return data_service.qa_report()


@app.get("/models/{model_name}/metrics")
def get_model_metrics(model_name: str) -> dict:
    model_metrics = data_service.metrics().get("models", {}).get(model_name)
    if model_metrics is None:
        raise HTTPException(status_code=404, detail="Model metrics not found")
    return model_metrics


@app.get("/narrative/status")
def get_narrative_status() -> dict:
    return narrative_status()


@app.get("/narrative/schema")
def get_narrative_schema() -> dict:
    return NARRATIVE_SCHEMA


@app.post("/simulate")
def post_simulate(request: SimulationRequest) -> dict:
    try:
        if hasattr(request, "model_dump"):
            payload = request.model_dump(exclude_none=True)
        else:
            payload = request.dict(exclude_none=True)
        return data_service.simulate(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
