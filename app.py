"""FastAPI service: predict whether an FIR's chargesheet will miss the 90-day deadline.

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from src.contract import CRIME_TYPES, OFFENCE_GRAVITY, STATION_NAMES
from src.model_pipeline import api_record_to_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model" / "model.joblib"))
META_PATH = MODEL_PATH.with_name("metadata.json")
LOG_PATH = Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))

# Loaded ONCE at start-up, not per request.
MODEL = joblib.load(MODEL_PATH)
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

CrimeType = Literal[tuple(CRIME_TYPES)]  # type: ignore[valid-type]
Station = Literal[tuple(STATION_NAMES)]  # type: ignore[valid-type]


class CaseInput(BaseModel):
    """Data contract. Any value outside these rules is rejected with HTTP 422."""
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={"example": {
        "crime_type": "Cyber Fraud", "police_station": "Vijay Nagar", "hour_of_occurrence": 14, "month": 8,
        "num_accused": 2, "num_witnesses": 1, "property_value_inr": 75000.0, "io_caseload": 38,
        "registration_delay_days": 3, "has_cctv": False, "has_cdr_evidence": True,
        "has_financial_trail": True, "forensic_required": False, "repeat_offender_involved": False}})

    crime_type: CrimeType
    police_station: Station
    hour_of_occurrence: int = Field(ge=0, le=23, description="Hour the offence occurred (0-23)")
    month: int = Field(ge=1, le=12)
    num_accused: int = Field(ge=1, le=15)
    num_witnesses: int = Field(ge=0, le=30)
    property_value_inr: float = Field(ge=0, le=50_000_000)
    io_caseload: int = Field(ge=1, le=60, description="Open cases held by the investigating officer")
    registration_delay_days: int = Field(ge=0, le=365, description="Days between offence and FIR")
    has_cctv: bool
    has_cdr_evidence: bool
    has_financial_trail: bool
    forensic_required: bool
    repeat_offender_involved: bool


class Prediction(BaseModel):
    delay_probability: float
    predicted_delayed: bool
    risk_band: Literal["High", "Medium", "Low"]
    tips: list[str]
    model: str


def rule_based_tips(c: CaseInput, p: float) -> list[str]:
    """2-3 practical tips for the supervising officer."""
    tips: list[str] = []
    if c.io_caseload >= 35:
        tips.append(f"The IO holds {c.io_caseload} open cases; reassign or add a second IO to meet the 90-day deadline.")
    if c.forensic_required:
        tips.append("Send exhibits to the forensic lab this week and track the report date; lab delays are a common cause of late chargesheets.")
    if c.has_financial_trail:
        tips.append("Request bank statements and KYC from all accounts in the first 15 days.")
    if c.has_cdr_evidence:
        tips.append("Apply for CDR / tower data early; telecom replies often take 2-4 weeks.")
    if c.num_accused >= 3:
        tips.append("Several accused: record arrests and statements on one timeline so a single chargesheet can be filed.")
    if c.registration_delay_days > 7:
        tips.append("The FIR was registered late; record the reason for the delay in the case diary.")
    if OFFENCE_GRAVITY[c.crime_type] == "High" and p >= 0.5:
        tips.append("Serious offence with high delay risk: put it on the SHO's weekly review list.")
    if not tips:
        tips.append("Routine case: follow the standard investigation checklist and file within 60 days.")
    return tips[:3]


def log_prediction(payload: dict, result: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "input": payload, "output": result}) + "\n")


app = FastAPI(title="FIR Chargesheet-Delay Predictor",
              description="Predicts whether an FIR's chargesheet is likely to miss the 90-day deadline. "
                          "Trained on synthetic data for CSL236.", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None,
            "model": META.get("params", {}).get("algorithm", "unknown"),
            "test_f1": META.get("metrics", {}).get("f1")}


@app.get("/model-info")
def model_info() -> dict:
    """Metrics and global feature importance (permutation importance on the test set)."""
    return {"params": META.get("params"), "metrics": META.get("metrics"),
            "feature_importance": META.get("feature_importance"), "trained_at": META.get("trained_at")}


@app.post("/predict", response_model=Prediction)
def predict(case: CaseInput) -> Prediction:
    payload = case.model_dump()
    p = float(MODEL.predict_proba(api_record_to_frame(payload))[0, 1])
    band = "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"
    result = {"delay_probability": round(p, 4), "predicted_delayed": p >= 0.5, "risk_band": band,
              "tips": rule_based_tips(case, p), "model": META.get("params", {}).get("algorithm", "unknown")}
    log_prediction(payload, result)
    return Prediction(**result)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
