import argparse
import json

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import MODEL_FEATURES
from ml.metrics import add_prediction_ranks, evaluate_prediction_frame
from ml.narrative import NARRATIVE_FEATURES, NARRATIVE_FEATURES_CSV
from ml.paths import FEATURES_CSV, PROCESSED_DATA_DIR


NARRATIVE_EXPERIMENT_JSON = PROCESSED_DATA_DIR / "narrative_experiment.json"


def _backtest(frame: pd.DataFrame, features: list[str], start_year: int) -> dict:
    yearly = {}
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.1))])
    for year in sorted(year for year in frame["Year"].unique() if year >= start_year):
        train = frame[frame["Year"] < year]
        test = frame[frame["Year"] == year]
        if train.empty or test.empty:
            continue
        model.fit(train[features], train["Share"])
        prediction_frame = test[
            ["Player", "Pos", "Tm", "Team", "Year", "Share", "Pts Won", "Pts Max"]
        ].copy()
        prediction_frame["prediction"] = model.predict(test[features])
        prediction_frame = add_prediction_ranks(prediction_frame)
        yearly[str(year)] = evaluate_prediction_frame(prediction_frame)

    summary = {}
    for metric in ["mse", "top_1_accuracy", "top_3_recall", "top_5_average_precision"]:
        values = [payload[metric] for payload in yearly.values()]
        summary[metric] = float(sum(values) / len(values)) if values else 0.0
    return {"yearly": yearly, "summary": summary}


def run_experiment(start_year: int = 2008) -> dict:
    if not NARRATIVE_FEATURES_CSV.exists():
        return {
            "status": "missing_narrative_data",
            "message": "Run `python -m ml.narrative` and fill data/processed/narrative_features.csv first.",
        }

    features = pd.read_csv(FEATURES_CSV)
    narrative = pd.read_csv(NARRATIVE_FEATURES_CSV)
    if narrative.empty:
        return {
            "status": "empty_narrative_data",
            "message": "Narrative feature template exists but has no player-season rows yet.",
        }

    merged = features.merge(narrative, how="left", on=["Player", "Year"])
    for column in NARRATIVE_FEATURES:
        if column not in merged.columns:
            merged[column] = 0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    basketball_only = _backtest(merged, MODEL_FEATURES, start_year)
    basketball_plus_narrative = _backtest(
        merged,
        MODEL_FEATURES + NARRATIVE_FEATURES,
        start_year,
    )

    payload = {
        "status": "complete",
        "features": NARRATIVE_FEATURES,
        "basketball_only": basketball_only,
        "basketball_plus_narrative": basketball_plus_narrative,
    }
    NARRATIVE_EXPERIMENT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run narrative feature ablation experiment.")
    parser.add_argument("--start-year", type=int, default=2008)
    args = parser.parse_args()
    print(json.dumps(run_experiment(start_year=args.start_year), indent=2))


if __name__ == "__main__":
    main()

