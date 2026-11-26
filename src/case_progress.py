"""Case progress: pendency, chargesheet delays, conviction rates, IO workload."""

from __future__ import annotations

import pandas as pd

from src.contract import CHARGESHEET_DEADLINE_DAYS


def overdue_chargesheets(df: pd.DataFrame, days: int = CHARGESHEET_DEADLINE_DAYS) -> pd.DataFrame:
    """Cases still under investigation for more than `days` days (oldest first)."""
    p = df[(df["case_status"] == "Under Investigation") & (df["case_age_days"] > days)]
    cols = ["fir_no", "police_station", "crime_type", "date_registered", "case_age_days", "io_id", "num_accused"]
    return p.sort_values("case_age_days", ascending=False)[cols].reset_index(drop=True)


def pendency_by_station(df: pd.DataFrame) -> pd.DataFrame:
    pend = df["case_status"] == "Under Investigation"
    out = (df.assign(pending=pend, overdue=pend & (df["case_age_days"] > CHARGESHEET_DEADLINE_DAYS))
             .groupby("police_station")
             .agg(total_firs=("fir_no", "count"), pending=("pending", "sum"), overdue_90d=("overdue", "sum"))
             .reset_index())
    out["pending_%"] = (out["pending"] / out["total_firs"] * 100).round(1)
    return out.sort_values("overdue_90d", ascending=False).reset_index(drop=True)


def chargesheet_delay_by_crime(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["days_to_chargesheet"].notna()]
    out = (f.groupby("crime_type")["days_to_chargesheet"]
             .agg(filed="count", median_days="median", mean_days="mean",
                  within_90d=lambda s: (s <= CHARGESHEET_DEADLINE_DAYS).mean() * 100)
             .round(1).reset_index())
    out = out.rename(columns={"within_90d": "filed_within_90d_%"})
    return out.sort_values("median_days", ascending=False).reset_index(drop=True)


def conviction_rates(df: pd.DataFrame, min_decided: int = 10) -> pd.DataFrame:
    """Conviction rate = convicted / (convicted + acquitted) among decided cases."""
    d = df[df["court_status"].isin(["Convicted", "Acquitted"])]
    out = (d.groupby("crime_type")["court_status"]
             .agg(decided="count", convicted=lambda s: (s == "Convicted").sum())
             .reset_index())
    out["conviction_rate_%"] = (out["convicted"] / out["decided"] * 100).round(1)
    out = out[out["decided"] >= min_decided]
    overall = out["convicted"].sum() / out["decided"].sum() * 100
    out.attrs["overall_rate"] = round(float(overall), 1)
    return out.sort_values("conviction_rate_%").reset_index(drop=True)


def io_workload(df: pd.DataFrame) -> pd.DataFrame:
    pend = df["case_status"] == "Under Investigation"
    out = (df.assign(pending=pend, overdue=pend & (df["case_age_days"] > CHARGESHEET_DEADLINE_DAYS))
             .groupby(["io_id", "police_station"])
             .agg(cases=("fir_no", "count"), pending=("pending", "sum"), overdue_90d=("overdue", "sum"),
                  median_caseload=("io_caseload", "median"))
             .reset_index())
    return out.sort_values(["overdue_90d", "pending"], ascending=False).reset_index(drop=True)


def status_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["case_status", "court_status"]).size().rename("firs").reset_index()
              .sort_values("firs", ascending=False).reset_index(drop=True))
