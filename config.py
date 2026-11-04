"""Shared settings. Everyone imports from here so results are comparable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
MODEL_LOG = ROOT / "results" / "model_log.csv"

RANDOM_STATE = 42

# Time-based split: never train on years after the test period starts.
TRAIN_END_YEAR = 2021   # train: 2010-2021
TEST_START_YEAR = 2022  # test:  2022-2024 (2025 has no next-year label)

# Target definition (agree as a team, then don't change silently).
# Classification: "stress year" = next-year inflation above this threshold.
INFLATION_STRESS_THRESHOLD = 10.0
