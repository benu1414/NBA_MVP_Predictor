import math

import pandas as pd
from sklearn.metrics import mean_squared_error


def add_prediction_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked = ranked.sort_values("prediction", ascending=False)
    ranked["Predicted_Rk"] = range(1, ranked.shape[0] + 1)
    ranked = ranked.sort_values("Share", ascending=False)
    ranked["Rk"] = range(1, ranked.shape[0] + 1)
    ranked["Rank_Diff"] = ranked["Rk"] - ranked["Predicted_Rk"]
    return ranked


def top_k_average_precision(frame: pd.DataFrame, k: int = 5) -> float:
    actual = frame.sort_values("Share", ascending=False).head(k)
    predicted = frame.sort_values("prediction", ascending=False)
    actual_players = set(actual["Player"].values)

    precisions = []
    found = 0
    for seek, (_, row) in enumerate(predicted.iterrows(), start=1):
        if row["Player"] in actual_players:
            found += 1
            precisions.append(found / seek)

    return sum(precisions) / len(precisions) if precisions else 0.0


def top_k_recall(frame: pd.DataFrame, k: int = 3) -> float:
    actual = set(frame.sort_values("Share", ascending=False).head(k)["Player"])
    predicted = set(frame.sort_values("prediction", ascending=False).head(k)["Player"])
    return len(actual & predicted) / k if k else 0.0


def winner_rank(frame: pd.DataFrame) -> int | None:
    ranked = add_prediction_ranks(frame)
    winner = ranked.sort_values("Share", ascending=False).head(1)
    if winner.empty:
        return None
    return int(winner.iloc[0]["Predicted_Rk"])


def spearman_rank(frame: pd.DataFrame) -> float | None:
    ranked = add_prediction_ranks(frame)
    corr = ranked[["Rk", "Predicted_Rk"]].corr(method="spearman").iloc[0, 1]
    if corr is None or math.isnan(corr):
        return None
    return float(corr)


def evaluate_prediction_frame(frame: pd.DataFrame) -> dict:
    winner_predicted_rank = winner_rank(frame)
    return {
        "mse": float(mean_squared_error(frame["Share"], frame["prediction"])),
        "top_1_accuracy": 1.0 if winner_predicted_rank == 1 else 0.0,
        "top_3_recall": top_k_recall(frame, 3),
        "top_5_average_precision": top_k_average_precision(frame, 5),
        "winner_predicted_rank": winner_predicted_rank,
        "spearman_rank": spearman_rank(frame),
    }

