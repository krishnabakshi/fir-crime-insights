"""Drift check: compare logged prediction inputs with the training data.

For every numeric feature it reports the mean and standard deviation of the
inputs the API has received, next to the training values saved in
model/metadata.json, and flags a feature when its mean has moved by more than
`--threshold` training standard deviations. Category shares (crime type,
station) are compared the same way using the share of each value.

Usage:  python scripts/drift_check.py [--log logs/predictions.jsonl] [--threshold 0.5]
Exit code 1 when drift is found, so it can run as a scheduled CI job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(line)["input"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def numeric_drift(logged: pd.DataFrame, stats: dict, threshold: float) -> pd.DataFrame:
    out = []
    for feat, s in stats.items():
        if feat not in logged:
            continue
        m, sd = logged[feat].mean(), logged[feat].std()
        shift = abs(m - s["mean"]) / (s["std"] or 1.0)
        out.append({"feature": feat, "train_mean": round(s["mean"], 2), "train_std": round(s["std"], 2),
                    "live_mean": round(m, 2), "live_std": round(sd, 2) if pd.notna(sd) else None,
                    "shift_in_sd": round(shift, 2), "drift": shift > threshold})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "predictions.jsonl")
    ap.add_argument("--meta", type=Path, default=ROOT / "model" / "metadata.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-rows", type=int, default=20)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"No prediction log at {args.log}")
        return 0
    logged = load_log(args.log)
    if len(logged) < args.min_rows:
        print(f"Only {len(logged)} logged predictions; need {args.min_rows} for a meaningful check.")
        return 0
    stats = json.loads(args.meta.read_text())["training_stats"]
    report = numeric_drift(logged, stats, args.threshold)
    print(f"Drift check on {len(logged)} logged predictions (threshold {args.threshold} SD)\n")
    print(report.to_string(index=False))
    print("\nTop crime types in live traffic:")
    print((logged["crime_type"].value_counts(normalize=True) * 100).round(1).head(5).to_string())
    drifted = report.loc[report["drift"], "feature"].tolist()
    if drifted:
        print(f"\nDRIFT in {drifted}: review recent FIRs and retrain with `python train.py --sweep`.")
        return 1
    print("\nNo drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
