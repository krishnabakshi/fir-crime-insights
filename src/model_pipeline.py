"""Supervised model: will this FIR's chargesheet miss the 90-day deadline?

Every learned preprocessing step (imputers, encoders, scaler, log transform,
variance filter) is inside one sklearn Pipeline. The Pipeline is fitted on the
training split only, so there is no leakage from the test split, and the same
object is saved to model/model.joblib and used by the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.contract import (BINARY_FEATURES, GRAVITY_ORDER, MODEL_FEATURES, NOMINAL_FEATURES,
                          NUMERIC_FEATURES, OFFENCE_GRAVITY, ORDINAL_FEATURES, RANDOM_STATE,
                          SEASON_OF_MONTH, SKEWED_FEATURES, TARGET, TIME_OF_DAY_BINS,
                          TIME_OF_DAY_LABELS)

ALGORITHMS = ["logreg", "decision_tree", "random_forest", "knn"]


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Labelled rows only; columns in the contract order with plain dtypes."""
    lab = df[df[TARGET].notna()]
    return to_model_frame(lab), lab[TARGET].astype(int)


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[MODEL_FEATURES].copy()
    for c in NUMERIC_FEATURES + SKEWED_FEATURES + BINARY_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in ORDINAL_FEATURES + NOMINAL_FEATURES:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def api_record_to_frame(rec: dict) -> pd.DataFrame:
    """Turn one validated API request into the model's feature frame."""
    hour = rec["hour_of_occurrence"]
    tod = pd.cut([hour], bins=TIME_OF_DAY_BINS, labels=TIME_OF_DAY_LABELS, right=False)[0]
    row = {
        "num_accused": rec["num_accused"], "num_witnesses": rec["num_witnesses"],
        "io_caseload": rec["io_caseload"], "registration_delay_days": rec["registration_delay_days"],
        "property_value_inr": rec["property_value_inr"],
        **{b: int(bool(rec[b])) for b in BINARY_FEATURES},
        "offence_gravity": OFFENCE_GRAVITY[rec["crime_type"]],
        "crime_type": rec["crime_type"], "police_station": rec["police_station"],
        "season": SEASON_OF_MONTH[rec["month"]], "time_of_day": str(tod),
    }
    return to_model_frame(pd.DataFrame([row]))


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill missing values with the median of the row's group, learned on the
    training split only (e.g. property value by crime type, IO caseload by
    police station). Falls back to the overall training median."""

    def __init__(self, groups: dict | None = None):
        self.groups = groups

    def _groups(self) -> dict:
        return self.groups if self.groups is not None else {
            "property_value_inr": "crime_type", "io_caseload": "police_station"}

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_, self.global_ = {}, {}
        for col, grp in self._groups().items():
            self.medians_[col] = X.groupby(grp)[col].median().to_dict()
            self.global_[col] = float(X[col].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, grp in self._groups().items():
            fill = X[grp].map(self.medians_[col]).astype(float).fillna(self.global_[col])
            X[col] = X[col].fillna(fill)
        return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    skewed = Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                       ("scale", StandardScaler())])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    ordinal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(categories=[GRAVITY_ORDER]))])
    nominal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES), ("skew", skewed, SKEWED_FEATURES),
        ("bin", binary, BINARY_FEATURES), ("ord", ordinal, ORDINAL_FEATURES),
        ("nom", nominal, NOMINAL_FEATURES),
    ], verbose_feature_names_out=False)


def make_classifier(algorithm: str, **params):
    if algorithm == "logreg":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=3000, class_weight="balanced")
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", 6),
                                      min_samples_leaf=params.get("min_samples_leaf", 10),
                                      class_weight="balanced", random_state=RANDOM_STATE)
    if algorithm == "random_forest":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 200),
                                      max_depth=params.get("max_depth", 8),
                                      min_samples_leaf=params.get("min_samples_leaf", 3),
                                      class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    if algorithm == "knn":
        return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 15), weights="distance")
    raise ValueError(f"Unknown algorithm {algorithm}; choose from {ALGORITHMS}")


def build_pipeline(algorithm: str = "random_forest", **params) -> Pipeline:
    return Pipeline([
        ("group_impute", GroupMedianImputer()),
        ("prep", build_preprocessor()),
        ("variance", VarianceThreshold(threshold=0.0)),  # drop constant columns
        ("clf", make_classifier(algorithm, **params)),
    ])


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified split BEFORE any imputer/encoder/scaler is fitted."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4)}


def raw_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on the original input columns (model-agnostic)."""
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean.round(4)})
              .sort_values("importance", ascending=False).reset_index(drop=True))


def training_stats(X_train: pd.DataFrame) -> dict:
    """Mean/std of numeric inputs, saved for the drift check."""
    num = X_train[NUMERIC_FEATURES + SKEWED_FEATURES]
    return {c: {"mean": float(num[c].mean()), "std": float(num[c].std())} for c in num.columns}
