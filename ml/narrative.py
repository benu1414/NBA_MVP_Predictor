from pathlib import Path

import pandas as pd

from ml.paths import PROCESSED_DATA_DIR


NARRATIVE_FEATURES_CSV = PROCESSED_DATA_DIR / "narrative_features.csv"

NARRATIVE_FEATURES = [
    "media_mentions_count",
    "search_trend_score",
    "reddit_mentions_count",
    "sentiment_score",
    "team_surprise_score",
    "breakout_score",
    "prior_mvp_fatigue",
]

NARRATIVE_SCHEMA = {
    "Player": "Player name matching the cleaned feature dataset.",
    "Year": "Season year, such as 2021.",
    "media_mentions_count": "Count of player mentions from approved media sources before voting.",
    "search_trend_score": "Season-normalized search interest score.",
    "reddit_mentions_count": "Count of relevant Reddit mentions before voting.",
    "sentiment_score": "Average normalized sentiment, ideally from -1 to 1.",
    "team_surprise_score": "How much the team outperformed preseason or prior-year expectation.",
    "breakout_score": "Signal for first major leap or career-best season.",
    "prior_mvp_fatigue": "Penalty-style signal for recent repeated MVP wins or voter fatigue.",
}


def narrative_status() -> dict:
    exists = NARRATIVE_FEATURES_CSV.exists()
    rows = 0
    years = []
    if exists:
        frame = pd.read_csv(NARRATIVE_FEATURES_CSV)
        rows = int(frame.shape[0])
        if "Year" in frame.columns:
            years = sorted(int(year) for year in frame["Year"].dropna().unique())

    return {
        "enabled": exists and rows > 0,
        "path": str(NARRATIVE_FEATURES_CSV),
        "rows": rows,
        "years": years,
        "features": NARRATIVE_FEATURES,
        "method": "Train basketball_only and basketball_plus_narrative feature sets, then compare backtest metrics.",
    }


def create_template(path: Path = NARRATIVE_FEATURES_CSV) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["Player", "Year"] + NARRATIVE_FEATURES
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    return path


def main() -> None:
    path = create_template()
    print(f"Saved narrative feature template to {path}")


if __name__ == "__main__":
    main()

