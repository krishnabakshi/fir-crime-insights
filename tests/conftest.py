"""Shared test setup: repo root on sys.path, temp prediction log, model present."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # keep test predictions out of the real drift log
    os.environ.setdefault("PREDICTION_LOG", str(ROOT / "logs" / "test_predictions.jsonl"))
    # a fresh clone without model/model.joblib still gets a working test run
    if not (ROOT / "model" / "model.joblib").exists():
        subprocess.run([sys.executable, str(ROOT / "train.py"), "--algorithm", "logreg", "--no-mlflow"],
                       check=True, cwd=ROOT)
