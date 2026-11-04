"""Shared helpers: load the panel, split by time, score models, log results.

Use these in every notebook so all models are compared on exactly the same data.
"""
from datetime import datetime

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from config import MODEL_LOG, PROCESSED, TEST_START_YEAR, TRAIN_END_YEAR


def load_panel(target="stress_next"):
    """Panel rows that have a target value."""
    panel = pd.read_csv(PROCESSED / "panel.csv")
    return panel.dropna(subset=[target]).reset_index(drop=True)


def time_split(df, features, target="stress_next"):
    """Train on years <= TRAIN_END_YEAR, test on years >= TEST_START_YEAR.

    No random shuffling: we must not use the future to predict the past.
    """
    train = df[df["year"] <= TRAIN_END_YEAR]
    test = df[df["year"] >= TEST_START_YEAR]
    return train[features], test[features], train[target], test[target]


def evaluate(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return {k: round(v, 4) for k, v in out.items()}


def log_result(model_name, params, metrics, notes=""):
    """Append one experiment to results/model_log.csv (our 'effort' record)."""
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "model": model_name, "params": str(params),
           **metrics, "notes": notes}
    new = pd.DataFrame([row])
    if MODEL_LOG.exists():
        new = pd.concat([pd.read_csv(MODEL_LOG), new], ignore_index=True)
    new.to_csv(MODEL_LOG, index=False)
    return row
