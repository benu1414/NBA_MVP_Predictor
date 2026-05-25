from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = ROOT_DIR / "MVP_Web_Scraper"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

PLAYER_MVP_STATS_CSV = SOURCE_DATA_DIR / "player_mvp_stats.csv"
FEATURES_CSV = PROCESSED_DATA_DIR / "player_season_features.csv"
PREDICTIONS_CSV = PROCESSED_DATA_DIR / "predictions.csv"
MODEL_METRICS_JSON = MODELS_DIR / "metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
QA_REPORT_JSON = PROCESSED_DATA_DIR / "qa_report.json"
