import argparse
import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.estimators import TwoStageMvpModel
from ml.features import MODEL_FEATURES, build_and_save_features
from ml.metrics import add_prediction_ranks, evaluate_prediction_frame
from ml.paths import BEST_MODEL_PATH, FEATURES_CSV, MODEL_METRICS_JSON, MODELS_DIR, PREDICTIONS_CSV


@dataclass
class ModelSpec:
    name: str
    estimator: object


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "ridge",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=0.1)),
                ]
            ),
        ),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=10,
                random_state=1,
                min_samples_split=5,
                max_depth=5,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            "gradient_boosting",
            GradientBoostingRegressor(random_state=1),
        ),
        ModelSpec(
            "two_stage_ridge",
            TwoStageMvpModel(alpha=0.1),
        ),
    ]


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        return build_and_save_features()
    return pd.read_csv(FEATURES_CSV)


def backtest_model(frame: pd.DataFrame, spec: ModelSpec, start_year: int) -> tuple[pd.DataFrame, dict]:
    years = sorted(year for year in frame["Year"].unique() if year >= start_year)
    all_predictions = []
    yearly_metrics = {}

    for year in years:
        train = frame[frame["Year"] < year]
        test = frame[frame["Year"] == year]
        if train.empty or test.empty:
            continue

        spec.estimator.fit(train[MODEL_FEATURES], train["Share"])
        predictions = spec.estimator.predict(test[MODEL_FEATURES])
        year_predictions = test[
            ["Player", "Pos", "Tm", "Team", "Year", "Share", "Pts Won", "Pts Max"]
        ].copy()
        year_predictions["model"] = spec.name
        year_predictions["prediction"] = predictions
        year_predictions = add_prediction_ranks(year_predictions)
        all_predictions.append(year_predictions)
        yearly_metrics[str(year)] = evaluate_prediction_frame(year_predictions)

    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_metrics(yearly_metrics)
    return predictions_frame, {"yearly": yearly_metrics, "summary": summary}


def summarize_metrics(yearly_metrics: dict) -> dict:
    keys = ["mse", "top_1_accuracy", "top_3_recall", "top_5_average_precision"]
    summary = {}
    for key in keys:
        values = [metrics[key] for metrics in yearly_metrics.values() if metrics[key] is not None]
        summary[key] = float(sum(values) / len(values)) if values else 0.0

    winner_ranks = [
        metrics["winner_predicted_rank"]
        for metrics in yearly_metrics.values()
        if metrics["winner_predicted_rank"] is not None
    ]
    summary["average_winner_predicted_rank"] = (
        float(sum(winner_ranks) / len(winner_ranks)) if winner_ranks else None
    )
    return summary


def select_best_model(metrics: dict) -> str:
    return max(
        metrics,
        key=lambda name: (
            metrics[name]["summary"]["top_5_average_precision"],
            metrics[name]["summary"]["top_1_accuracy"],
            -metrics[name]["summary"]["mse"],
        ),
    )


def train_final_model(frame: pd.DataFrame, spec: ModelSpec) -> object:
    spec.estimator.fit(frame[MODEL_FEATURES], frame["Share"])
    return spec.estimator


def run_training(start_year: int = 2008) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_features()
    all_predictions = []
    all_metrics = {}

    specs = model_specs()
    for spec in specs:
        print(f"Backtesting {spec.name}...", flush=True)
        predictions, metrics = backtest_model(frame, spec, start_year=start_year)
        all_predictions.append(predictions)
        all_metrics[spec.name] = metrics

    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    predictions_frame.to_csv(PREDICTIONS_CSV, index=False)

    best_model_name = select_best_model(all_metrics)
    best_spec = next(spec for spec in model_specs() if spec.name == best_model_name)
    final_model = train_final_model(frame, best_spec)

    artifact = {
        "model_name": best_model_name,
        "model": final_model,
        "features": MODEL_FEATURES,
        "trained_years": [int(frame["Year"].min()), int(frame["Year"].max())],
        "metrics": all_metrics[best_model_name],
    }
    joblib.dump(artifact, BEST_MODEL_PATH)

    payload = {
        "best_model": best_model_name,
        "features": MODEL_FEATURES,
        "models": all_metrics,
    }
    MODEL_METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and backtest MVP prediction models.")
    parser.add_argument("--start-year", type=int, default=2008)
    args = parser.parse_args()
    metrics = run_training(start_year=args.start_year)
    print(f"Best model: {metrics['best_model']}")
    print(f"Saved metrics to {MODEL_METRICS_JSON}")
    print(f"Saved best model to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
