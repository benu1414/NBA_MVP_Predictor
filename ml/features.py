import argparse

import pandas as pd

from ml.paths import FEATURES_CSV, PLAYER_MVP_STATS_CSV, PROCESSED_DATA_DIR


BASE_FEATURES = [
    "Age",
    "G",
    "GS",
    "MP",
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "2P",
    "2PA",
    "2P%",
    "eFG%",
    "FT",
    "FTA",
    "FT%",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "Year",
    "W",
    "L",
    "W/L%",
    "GB",
    "PS/G",
    "PA/G",
    "SRS",
]

ENGINEERED_FEATURES = [
    "Games_Missed",
    "PTS_Rank",
    "AST_Rank",
    "TRB_Rank",
    "STL_Rank",
    "BLK_Rank",
    "W_Rank",
    "SRS_Rank",
    "PTS_Pctile",
    "AST_Pctile",
    "TRB_Pctile",
    "W_Pctile",
    "SRS_Pctile",
    "Team_Win_Share",
    "Usage_Proxy",
    "Scoring_Efficiency_Proxy",
    "PER_Rank",
    "BPM_Rank",
    "VORP_Rank",
    "WS_Rank",
    "PER_Pctile",
    "BPM_Pctile",
    "VORP_Pctile",
    "WS_Pctile",
]

ADVANCED_FEATURES = [
    "PER",
    "TS%",
    "3PAr",
    "FTr",
    "ORB%",
    "DRB%",
    "TRB%",
    "AST%",
    "STL%",
    "BLK%",
    "TOV%",
    "USG%",
    "OWS",
    "DWS",
    "WS",
    "WS/48",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",
]

MODEL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES + ENGINEERED_FEATURES


def load_source_stats(path=PLAYER_MVP_STATS_CSV) -> pd.DataFrame:
    stats = pd.read_csv(path)
    if "Unnamed: 0" in stats.columns:
        stats = stats.drop(columns=["Unnamed: 0"])
    return stats


def _rank_within_year(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    return frame.groupby("Year")[column].rank(method="min", ascending=ascending)


def _pctile_within_year(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("Year")[column].rank(pct=True)


def build_features(stats: pd.DataFrame) -> pd.DataFrame:
    features = stats.copy()

    numeric_columns = features.select_dtypes(include=["number"]).columns
    features[numeric_columns] = features[numeric_columns].fillna(0)
    features = features.fillna({"Pos": "UNK", "Tm": "UNK", "Team": "Unknown"})

    for column in ADVANCED_FEATURES:
        if column not in features.columns:
            features[column] = 0

    season_games = features.groupby("Year")["G"].transform("max")
    features["Games_Missed"] = season_games - features["G"]

    for column in ["PTS", "AST", "TRB", "STL", "BLK", "W", "SRS"]:
        features[f"{column}_Rank"] = _rank_within_year(features, column, ascending=False)

    for column in ["PTS", "AST", "TRB", "W", "SRS"]:
        features[f"{column}_Pctile"] = _pctile_within_year(features, column)

    for column in ["PER", "BPM", "VORP", "WS"]:
        features[f"{column}_Rank"] = _rank_within_year(features, column, ascending=False)
        features[f"{column}_Pctile"] = _pctile_within_year(features, column)

    features["Team_Win_Share"] = features["W"] / (features["W"] + features["L"]).replace(0, 1)
    features["Usage_Proxy"] = features["FGA"] + 0.44 * features["FTA"] + features["TOV"]
    features["Scoring_Efficiency_Proxy"] = features["PTS"] / features["Usage_Proxy"].replace(0, 1)
    features["MVP_Candidate"] = (features["Share"] > 0).astype(int)

    for column in MODEL_FEATURES + ["Share"]:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

    return features


def build_and_save_features() -> pd.DataFrame:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features(load_source_stats())
    features.to_csv(FEATURES_CSV, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MVP player-season feature dataset.")
    parser.parse_args()
    features = build_and_save_features()
    print(f"Saved {features.shape[0]} rows and {features.shape[1]} columns to {FEATURES_CSV}")


if __name__ == "__main__":
    main()
