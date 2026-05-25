import argparse
import json

import pandas as pd

from ml.paths import FEATURES_CSV, PLAYER_MVP_STATS_CSV, PROCESSED_DATA_DIR


QA_REPORT_JSON = PROCESSED_DATA_DIR / "qa_report.json"


def build_qa_report() -> dict:
    stats = pd.read_csv(PLAYER_MVP_STATS_CSV)
    features = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.exists() else pd.DataFrame()

    duplicate_player_years = int(stats.duplicated(["Player", "Year"]).sum())
    missing_team_rows = int(stats[["Team", "W", "L", "SRS"]].isna().any(axis=1).sum())
    years = sorted(int(year) for year in stats["Year"].dropna().unique())
    vote_counts = (
        stats[stats["Share"] > 0]
        .groupby("Year")["Player"]
        .count()
        .reindex(years, fill_value=0)
        .astype(int)
        .to_dict()
    )
    row_counts = stats.groupby("Year")["Player"].count().astype(int).to_dict()

    feature_missing = {}
    if not features.empty:
        feature_missing = {
            column: int(count)
            for column, count in features.isna().sum().items()
            if int(count) > 0
        }

    issues = []
    if duplicate_player_years:
        issues.append(f"{duplicate_player_years} duplicate Player-Year rows")
    if missing_team_rows:
        issues.append(f"{missing_team_rows} rows missing joined team context")
    if feature_missing:
        issues.append("Feature dataset contains missing values")

    return {
        "status": "pass" if not issues else "warn",
        "issues": issues,
        "source_rows": int(stats.shape[0]),
        "feature_rows": int(features.shape[0]) if not features.empty else 0,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "years": years,
        "row_counts_by_year": {str(k): int(v) for k, v in row_counts.items()},
        "mvp_vote_counts_by_year": {str(k): int(v) for k, v in vote_counts.items()},
        "duplicate_player_years": duplicate_player_years,
        "missing_team_rows": missing_team_rows,
        "feature_missing_values": feature_missing,
    }


def save_qa_report() -> dict:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    report = build_qa_report()
    QA_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data quality checks.")
    parser.parse_args()
    report = save_qa_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

