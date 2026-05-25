import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.features import MODEL_FEATURES, build_and_save_features
from ml.metrics import add_prediction_ranks
from ml.paths import BEST_MODEL_PATH, FEATURES_CSV, MODEL_METRICS_JSON, PREDICTIONS_CSV, QA_REPORT_JSON


@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        build_and_save_features()
    return pd.read_csv(FEATURES_CSV)


@lru_cache(maxsize=1)
def predictions() -> pd.DataFrame:
    if PREDICTIONS_CSV.exists():
        return pd.read_csv(PREDICTIONS_CSV)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def metrics() -> dict:
    if MODEL_METRICS_JSON.exists():
        return json.loads(Path(MODEL_METRICS_JSON).read_text(encoding="utf-8"))
    return {"best_model": None, "features": MODEL_FEATURES, "models": {}}


@lru_cache(maxsize=1)
def model_artifact() -> dict | None:
    if not BEST_MODEL_PATH.exists():
        return None
    return joblib.load(BEST_MODEL_PATH)


@lru_cache(maxsize=1)
def qa_report() -> dict:
    if QA_REPORT_JSON.exists():
        return json.loads(Path(QA_REPORT_JSON).read_text(encoding="utf-8"))
    from ml.qa import save_qa_report

    return save_qa_report()


def seasons() -> list[int]:
    return sorted(int(year) for year in features()["Year"].unique())


def players() -> list[dict]:
    cols = ["Player", "Pos", "Tm", "Team"]
    player_frame = features()[cols].drop_duplicates("Player").sort_values("Player")
    return player_frame.to_dict(orient="records")


def player_history(player_name: str) -> list[dict]:
    frame = features()
    history = frame[frame["Player"].str.lower() == player_name.lower()].sort_values("Year")
    return history.to_dict(orient="records")


def player_season(player_name: str, year: int) -> dict | None:
    frame = features()
    row = frame[
        (frame["Player"].str.lower() == player_name.lower()) & (frame["Year"] == year)
    ]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def season_predictions(year: int, model: str | None = None) -> list[dict]:
    prediction_frame = predictions()
    if not prediction_frame.empty:
        season = prediction_frame[prediction_frame["Year"] == year]
        if model:
            season = season[season["model"] == model]
        elif "model" in season.columns and not season.empty:
            best = metrics().get("best_model")
            if best:
                season = season[season["model"] == best]
        if not season.empty:
            return season.sort_values("Predicted_Rk").to_dict(orient="records")

    artifact = model_artifact()
    if artifact is None:
        return []

    frame = features()
    season = frame[frame["Year"] == year].copy()
    season["prediction"] = artifact["model"].predict(season[artifact["features"]])
    ranked = add_prediction_ranks(
        season[["Player", "Pos", "Tm", "Team", "Year", "Share", "prediction"]]
    )
    ranked["model"] = artifact["model_name"]
    return ranked.sort_values("Predicted_Rk").to_dict(orient="records")


def actual_results(year: int) -> list[dict]:
    frame = features()
    season = frame[(frame["Year"] == year) & (frame["Share"] > 0)].copy()
    season = season.sort_values("Share", ascending=False)
    season["Rk"] = range(1, season.shape[0] + 1)
    return season[
        ["Rk", "Player", "Pos", "Tm", "Team", "Year", "Share", "Pts Won", "Pts Max"]
    ].to_dict(orient="records")


def simulate(payload: dict) -> dict:
    artifact = model_artifact()
    if artifact is None:
        raise RuntimeError("No model is trained yet. Run `python -m ml.train` first.")

    frame = features()
    year = int(payload.get("year", frame["Year"].max()))
    base_name = payload.get("player")
    if base_name:
        candidates = frame[frame["Player"].str.lower() == str(base_name).lower()]
        base = candidates.sort_values("Year").tail(1)
    else:
        base = pd.DataFrame()

    if base.empty:
        base = frame[frame["Year"] == year].sort_values("Share", ascending=False).head(1)
    row = base.iloc[0].copy()
    row["Year"] = year

    field_map = {
        "pts": "PTS",
        "ast": "AST",
        "trb": "TRB",
        "reb": "TRB",
        "wins": "W",
        "losses": "L",
        "games": "G",
        "srs": "SRS",
        "fg_pct": "FG%",
        "three_pct": "3P%",
        "ft_pct": "FT%",
    }
    for incoming, feature_name in field_map.items():
        if incoming in payload and payload[incoming] is not None:
            row[feature_name] = payload[incoming]

    season_context = frame[frame["Year"] == year].copy()
    if not season_context.empty:
        player_mask = season_context["Player"].str.lower() == str(row["Player"]).lower()
        if player_mask.any():
            season_context.loc[player_mask, row.index] = row.values
        else:
            season_context = pd.concat([season_context, pd.DataFrame([row])], ignore_index=True)
        from ml.features import build_features

        simulated_features = build_features(season_context)
        sample = simulated_features[
            simulated_features["Player"].str.lower() == str(row["Player"]).lower()
        ].tail(1)
    else:
        sample = pd.DataFrame([row])

    for feature in artifact["features"]:
        if feature not in sample.columns:
            sample[feature] = 0
    prediction = float(artifact["model"].predict(sample[artifact["features"]])[0])

    season = pd.DataFrame(season_predictions(year))
    if not season.empty:
        predicted_rank = int((season["prediction"] > prediction).sum() + 1)
    else:
        predicted_rank = None

    return {
        "player": str(row["Player"]),
        "year": year,
        "model": artifact["model_name"],
        "predicted_share": prediction,
        "estimated_rank": predicted_rank,
        "inputs": payload,
    }


def explain_player_season(player_name: str, year: int) -> dict | None:
    artifact = model_artifact()
    row = player_season(player_name, year)
    if artifact is None or row is None:
        return None

    sample = pd.DataFrame([row])
    feature_names = artifact["features"]
    for feature in feature_names:
        if feature not in sample.columns:
            sample[feature] = 0

    model = artifact["model"]
    prediction = float(model.predict(sample[feature_names])[0])
    contributions = []

    if hasattr(model, "named_steps") and "scaler" in model.named_steps and "model" in model.named_steps:
        scaler = model.named_steps["scaler"]
        regressor = model.named_steps["model"]
        transformed = scaler.transform(sample[feature_names])[0]
        for name, value, contribution in zip(
            feature_names, sample[feature_names].iloc[0], transformed * regressor.coef_
        ):
            contributions.append(
                {
                    "feature": name,
                    "value": float(value),
                    "contribution": float(contribution),
                }
            )
    elif hasattr(model, "feature_importances_"):
        for name, value, contribution in zip(
            feature_names, sample[feature_names].iloc[0], model.feature_importances_
        ):
            contributions.append(
                {
                    "feature": name,
                    "value": float(value),
                    "contribution": float(contribution),
                }
            )
    else:
        for name, value in sample[feature_names].iloc[0].items():
            contributions.append(
                {
                    "feature": name,
                    "value": float(value),
                    "contribution": 0.0,
                }
            )

    positives = sorted(contributions, key=lambda item: item["contribution"], reverse=True)[:8]
    negatives = sorted(contributions, key=lambda item: item["contribution"])[:8]

    return {
        "player": row["Player"],
        "year": int(year),
        "model": artifact["model_name"],
        "prediction": prediction,
        "actual_share": float(row.get("Share", 0)),
        "positive_factors": positives,
        "negative_factors": negatives,
    }
