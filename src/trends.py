"""Crime trends (rise/fall by area, month, season, time of day) and forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.contract import TIME_OF_DAY_LABELS


# ---------------------------------------------------------------------------
# Descriptive trends
# ---------------------------------------------------------------------------
def monthly_counts(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    m = df.assign(month_start=df["date_occurred"].dt.to_period("M").dt.start_time)
    if by is None:
        return m.groupby("month_start").size().rename("firs").to_frame()
    return m.pivot_table(index="month_start", columns=by, values="fir_no",
                         aggfunc="count", fill_value=0)


def trend_changes(df: pd.DataFrame, by: list[str] | str = "crime_type",
                  recent_months: int = 6, min_count: int = 8,
                  threshold_pct: float = 20.0) -> pd.DataFrame:
    """Year-on-year change: last `recent_months` months vs the same months a year earlier.

    Comparing with the same calendar months removes seasonality, so a rise
    in October-November theft is not mistaken for a trend.
    """
    by = [by] if isinstance(by, str) else list(by)
    end = df["date_occurred"].max().to_period("M")
    recent = pd.period_range(end - recent_months + 1, end, freq="M")
    prior = recent - 12
    p = df["date_occurred"].dt.to_period("M")
    cur = df[p.isin(recent)].groupby(by).size().rename("recent")
    old = df[p.isin(prior)].groupby(by).size().rename("year_before")
    out = pd.concat([cur, old], axis=1).fillna(0).astype(int)
    out = out[(out["recent"] + out["year_before"]) >= min_count].copy()
    out["change_pct"] = ((out["recent"] - out["year_before"]) / out["year_before"].clip(lower=1) * 100).round(1)
    out["direction"] = np.select([out["change_pct"] >= threshold_pct, out["change_pct"] <= -threshold_pct],
                                 ["Rise", "Fall"], "Stable")
    out.attrs["window"] = f"{recent[0]}..{recent[-1]} vs {prior[0]}..{prior[-1]}"
    return out.sort_values("change_pct", ascending=False).reset_index()


def season_profile(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
    t = pd.crosstab(df["crime_type"], df["season"])
    t = t[[c for c in order if c in t.columns]]
    return (t.div(t.sum(axis=1), axis=0) * 100).round(1)


def time_of_day_profile(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(df["crime_type"], df["time_of_day"])
    t = t[[c for c in TIME_OF_DAY_LABELS if c in t.columns]]
    return (t.div(t.sum(axis=1), axis=0) * 100).round(1)


def peak_times(df: pd.DataFrame, by: str = "police_station") -> pd.DataFrame:
    """Busiest time-of-day band, weekday and hour for each area (from history)."""
    rows = []
    for key, g in df.groupby(by):
        tod = g["time_of_day"].value_counts()
        rows.append({by: key, "peak_time_of_day": tod.index[0],
                     "peak_share_%": round(tod.iloc[0] / tod.sum() * 100, 1),
                     "peak_weekday": g["weekday"].value_counts().index[0],
                     "peak_hour": int(g["hour_of_occurrence"].mode().iloc[0])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forecasting weekly FIR counts per police station
# ---------------------------------------------------------------------------
def weekly_panel(df: pd.DataFrame, by: str = "police_station") -> pd.DataFrame:
    """Station x week matrix of FIR counts (complete weeks only, zeros filled)."""
    last_full = df["date_occurred"].max() - pd.Timedelta(days=6)
    weeks = pd.date_range(df["week"].min(), last_full, freq="W-MON")
    panel = df.pivot_table(index="week", columns=by, values="fir_no", aggfunc="count", fill_value=0)
    return panel.reindex(weeks, fill_value=0)


def _design(t: np.ndarray, weeks: pd.DatetimeIndex) -> np.ndarray:
    woy = weeks.isocalendar().week.to_numpy().astype(float)
    return np.column_stack([t, np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52),
                            np.sin(4 * np.pi * woy / 52), np.cos(4 * np.pi * woy / 52)])


def _fit_predict(name: str, y: pd.Series, horizon: int) -> np.ndarray:
    """Fit one model on history y and return `horizon` future values."""
    if name == "naive_mean4":
        return np.repeat(y.iloc[-4:].mean(), horizon)
    if name == "holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y.to_numpy(dtype=float), trend="add", damped_trend=True,
                                       initialization_method="estimated").fit()
        return np.clip(fit.forecast(horizon), 0, None)
    if name == "linear_fourier":
        t = np.arange(len(y), dtype=float)
        future_idx = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
        model = LinearRegression().fit(_design(t, y.index), y.to_numpy(dtype=float))
        tf = np.arange(len(y), len(y) + horizon, dtype=float)
        return np.clip(model.predict(_design(tf, future_idx)), 0, None)
    raise ValueError(name)


FORECAST_MODELS = ["naive_mean4", "holt_winters", "linear_fourier"]


def backtest(panel: pd.DataFrame, holdout: int = 12, horizon: int = 4) -> pd.DataFrame:
    """Rolling-origin backtest: forecast `horizon` weeks from several origins in
    the last `holdout` weeks; MAE per station and model."""
    rows = []
    for station in panel.columns:
        y = panel[station]
        for model in FORECAST_MODELS:
            errs = []
            for origin in range(len(y) - holdout, len(y) - horizon + 1, horizon):
                pred = _fit_predict(model, y.iloc[:origin], horizon)
                errs.append(mean_absolute_error(y.iloc[origin:origin + horizon], pred))
            rows.append({"police_station": station, "model": model, "mae": float(np.mean(errs))})
    return pd.DataFrame(rows)


def forecast_hotspots(df: pd.DataFrame, horizon: int = 4, holdout: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the best model per station by backtest MAE and forecast `horizon` weeks.

    Returns (forecast table, backtest table)."""
    panel = weekly_panel(df)
    bt = backtest(panel, holdout=holdout, horizon=horizon)
    best = bt.loc[bt.groupby("police_station")["mae"].idxmin()].set_index("police_station")
    peaks = peak_times(df).set_index("police_station")
    recent_avg = panel.iloc[-12:].mean()
    rows = []
    for station in panel.columns:
        model = best.loc[station, "model"]
        pred = _fit_predict(model, panel[station], horizon)
        rows.append({
            "police_station": station,
            "forecast_next_weeks": round(float(pred.sum()), 1),
            "forecast_per_week": round(float(pred.mean()), 2),
            "recent_12wk_avg": round(float(recent_avg[station]), 2),
            "best_model": model, "backtest_mae": round(float(best.loc[station, "mae"]), 2),
            "peak_time_of_day": peaks.loc[station, "peak_time_of_day"],
            "peak_weekday": peaks.loc[station, "peak_weekday"],
            "forecast_start": (panel.index[-1] + pd.Timedelta(weeks=1)).date(),
        })
    fc = pd.DataFrame(rows).sort_values("forecast_next_weeks", ascending=False).reset_index(drop=True)
    fc["risk_rank"] = np.arange(1, len(fc) + 1)
    return fc, bt
