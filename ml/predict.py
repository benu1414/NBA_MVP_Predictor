import argparse
import sys

import joblib
import pandas as pd

from ml.features import MODEL_FEATURES, build_and_save_features
from ml.metrics import add_prediction_ranks
from ml.paths import BEST_MODEL_PATH, FEATURES_CSV, PREDICTIONS_CSV


def load_model_artifact() -> dict:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python -m ml.train` first.")
    return joblib.load(BEST_MODEL_PATH)


def predict_season(year: int, use_backtest: bool = True) -> pd.DataFrame:
    if use_backtest and PREDICTIONS_CSV.exists():
        predictions = pd.read_csv(PREDICTIONS_CSV)
        if not predictions.empty:
            if "model" in predictions.columns:
                artifact = load_model_artifact() if BEST_MODEL_PATH.exists() else None
                model_name = artifact["model_name"] if artifact else predictions["model"].iloc[0]
                predictions = predictions[predictions["model"] == model_name]
            season_predictions = predictions[predictions["Year"] == year]
            if not season_predictions.empty:
                return season_predictions.sort_values("Predicted_Rk")

    if not FEATURES_CSV.exists():
        build_and_save_features()
    artifact = load_model_artifact()
    frame = pd.read_csv(FEATURES_CSV)
    season = frame[frame["Year"] == year].copy()
    if season.empty:
        raise ValueError(f"No feature rows found for season {year}.")

    season["prediction"] = artifact["model"].predict(season[MODEL_FEATURES])
    ranked = add_prediction_ranks(
        season[["Player", "Pos", "Tm", "Team", "Year", "Share", "prediction"]]
    )
    return ranked.sort_values("Predicted_Rk")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predict and rank MVP candidates for a season.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--final-model",
        action="store_true",
        help="Use the final model trained on all available seasons instead of saved backtest rows.",
    )
    args = parser.parse_args()
    print(predict_season(args.year, use_backtest=not args.final_model).head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
