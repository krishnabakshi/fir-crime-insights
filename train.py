"""Train the chargesheet-delay classifier with MLflow tracking.

Examples
--------
python train.py --n_estimators 100 --max_depth 6           # random forest (default)
python train.py --algorithm logreg --C 0.5
python train.py --algorithm decision_tree --max_depth 5 --min_samples_leaf 20
python train.py --algorithm knn --n_neighbors 25
python train.py --sweep                                     # 6 configs, 4 algorithms, keeps the best
mlflow ui --backend-store-uri sqlite:///mlflow.db

The model from the run is saved to model/model.joblib only if its test F1 is
at least as good as the model already there (use --force to always save).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import data_loader as dl  # noqa: E402
from src.contract import MODEL_FEATURES, TARGET  # noqa: E402
from src.model_pipeline import (ALGORITHMS, build_pipeline, evaluate, get_xy,  # noqa: E402
                                raw_feature_importance, split, training_stats)
from src.preprocessing import prepare_cases  # noqa: E402

MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metadata.json"

SWEEP = [
    {"algorithm": "logreg", "C": 1.0},
    {"algorithm": "logreg", "C": 0.1},
    {"algorithm": "decision_tree", "max_depth": 5, "min_samples_leaf": 20},
    {"algorithm": "random_forest", "n_estimators": 100, "max_depth": 6},
    {"algorithm": "random_forest", "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5},
    {"algorithm": "knn", "n_neighbors": 25},
]


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    src = dl.load_sources()
    cases, _ = prepare_cases(dl.load_cases(), src["criminal_history"])
    return get_xy(cases)


def run_one(cfg: dict, X_train, X_test, y_train, y_test, use_mlflow: bool = True) -> tuple[object, dict]:
    cfg = dict(cfg)
    algorithm = cfg.pop("algorithm")
    pipe = build_pipeline(algorithm, **cfg)
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    metrics["cv_f1_mean"] = round(float(cv_f1.mean()), 4)
    metrics["train_seconds"] = round(time.time() - t0, 2)
    params = {"algorithm": algorithm, **cfg}
    if use_mlflow:
        import mlflow
        with mlflow.start_run(run_name=f"{algorithm}-" + "-".join(f"{k}{v}" for k, v in cfg.items())):
            mlflow.log_params(params)
            mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test), "n_features_in": X_train.shape[1]})
            mlflow.log_metrics(metrics)
    print(f"{params} -> {metrics}")
    return pipe, {"params": params, "metrics": metrics}


def save_if_better(pipe, result: dict, X_train, X_test, y_test, force: bool = False) -> bool:
    current = json.loads(META_PATH.read_text())["metrics"]["f1"] if META_PATH.exists() else -1
    if not force and MODEL_PATH.exists() and result["metrics"]["f1"] < current:
        print(f"Kept existing model (F1 {current}) - this run F1 {result['metrics']['f1']}")
        return False
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    imp = raw_feature_importance(pipe, X_test, y_test)
    meta = {**result, "features": MODEL_FEATURES, "target": TARGET,
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "feature_importance": imp.head(8).to_dict(orient="records"),
            "training_stats": training_stats(X_train)}
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {MODEL_PATH} (F1 {result['metrics']['f1']})")
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the chargesheet-delay classifier")
    ap.add_argument("--algorithm", choices=ALGORITHMS, default="random_forest")
    ap.add_argument("--n_estimators", type=int, default=100)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_samples_leaf", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--sweep", action="store_true", help="run all configs in SWEEP and keep the best")
    ap.add_argument("--force", action="store_true", help="save this run's model even if F1 is lower")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging (quick CI/test runs)")
    ap.add_argument("--experiment", default="chargesheet-delay")
    args = ap.parse_args(argv)

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(args.experiment)

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = split(X, y)   # split BEFORE fitting anything
    print(f"Rows: train {len(X_train)}, test {len(X_test)}; delayed share {y.mean():.2%}")

    if args.sweep:
        results = [run_one(cfg, X_train, X_test, y_train, y_test, use_mlflow) for cfg in SWEEP]
        # choose on cross-validated F1 (training data only), not on the test set
        best_pipe, best = max(results, key=lambda r: (r[1]["metrics"]["cv_f1_mean"], r[1]["metrics"]["roc_auc"]))
        table = pd.DataFrame([{**r[1]["params"], **r[1]["metrics"]} for r in results])
        print("\n" + table.to_string(index=False))
        print(f"\nBest: {best['params']}")
        save_if_better(best_pipe, best, X_train, X_test, y_test, force=True)
        return

    alg = args.algorithm
    cfg = {"algorithm": alg}
    if alg == "random_forest":
        cfg.update(n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    elif alg == "decision_tree":
        cfg.update(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    elif alg == "logreg":
        cfg.update(C=args.C)
    elif alg == "knn":
        cfg.update(n_neighbors=args.n_neighbors)
    pipe, result = run_one(cfg, X_train, X_test, y_train, y_test, use_mlflow)
    save_if_better(pipe, result, X_train, X_test, y_test, force=args.force)


if __name__ == "__main__":
    main()
