import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TwoStageMvpModel(BaseEstimator, RegressorMixin):
    """Classify MVP candidates, then estimate vote share for likely candidates."""

    def __init__(self, alpha: float = 0.1, candidate_threshold: float = 0.0):
        self.alpha = alpha
        self.candidate_threshold = candidate_threshold

    def fit(self, X, y):
        candidate_y = (y > self.candidate_threshold).astype(int)
        self.classifier_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=1,
                    ),
                ),
            ]
        )
        self.regressor_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )

        self.classifier_.fit(X, candidate_y)
        candidate_mask = candidate_y == 1
        if candidate_mask.sum() >= 5:
            self.regressor_.fit(X[candidate_mask], y[candidate_mask])
        else:
            self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        probability = self.classifier_.predict_proba(X)[:, 1]
        share = self.regressor_.predict(X)
        return np.clip(probability * np.clip(share, 0, None), 0, None)

