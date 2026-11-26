"""API tests required by the CSL236 end-term guidelines (health, UI, domain logic, 422)."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

SIMPLE_CASE = {  # petty theft, CCTV, one accused, light IO workload
    "crime_type": "Theft", "police_station": "Rajwada", "hour_of_occurrence": 14, "month": 3,
    "num_accused": 1, "num_witnesses": 4, "property_value_inr": 8000.0, "io_caseload": 10,
    "registration_delay_days": 0, "has_cctv": True, "has_cdr_evidence": False,
    "has_financial_trail": False, "forensic_required": False, "repeat_offender_involved": False}

COMPLEX_CASE = {  # money laundering, many accused, forensics + money trail, overloaded IO
    "crime_type": "Money Laundering", "police_station": "Rau", "hour_of_occurrence": 12, "month": 8,
    "num_accused": 6, "num_witnesses": 0, "property_value_inr": 4500000.0, "io_caseload": 50,
    "registration_delay_days": 20, "has_cctv": False, "has_cdr_evidence": True,
    "has_financial_trail": True, "forensic_required": True, "repeat_offender_involved": True}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_ui_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Chargesheet-Delay Predictor" in r.text


def test_predict_valid_returns_probability_and_tips():
    r = client.post("/predict", json=SIMPLE_CASE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_band"] in {"High", "Medium", "Low"}
    assert 1 <= len(body["tips"]) <= 3


def test_domain_logic_complex_case_riskier_than_simple_case():
    """A complex, under-resourced case must get a higher delay risk than a simple one."""
    p_simple = client.post("/predict", json=SIMPLE_CASE).json()["delay_probability"]
    p_complex = client.post("/predict", json=COMPLEX_CASE).json()["delay_probability"]
    assert p_complex > p_simple
    assert p_complex >= 0.5 > p_simple


def test_domain_logic_higher_caseload_raises_risk():
    low = client.post("/predict", json={**COMPLEX_CASE, "io_caseload": 5}).json()["delay_probability"]
    high = client.post("/predict", json={**COMPLEX_CASE, "io_caseload": 55}).json()["delay_probability"]
    assert high > low


@pytest.mark.parametrize("field,value", [
    ("hour_of_occurrence", 24),        # above 23
    ("month", 0),                      # below 1
    ("num_accused", 0),                # below 1
    ("io_caseload", 61),               # above 60
    ("property_value_inr", -1.0),      # negative
    ("registration_delay_days", 400),  # above 365
])
def test_out_of_range_returns_422(field, value):
    r = client.post("/predict", json={**SIMPLE_CASE, field: value})
    assert r.status_code == 422


@pytest.mark.parametrize("field,value", [
    ("num_accused", "three"),          # wrong type
    ("has_cctv", "yes"),               # string instead of boolean
    ("crime_type", "Murder"),          # not in the allowed list
    ("police_station", "Unknown PS"),  # not in the allowed list
])
def test_wrong_type_or_value_returns_422(field, value):
    r = client.post("/predict", json={**SIMPLE_CASE, field: value})
    assert r.status_code == 422


def test_missing_and_extra_fields_return_422():
    missing = {k: v for k, v in SIMPLE_CASE.items() if k != "io_caseload"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**SIMPLE_CASE, "religion": "x"}).status_code == 422


def test_prediction_input_is_logged():
    log = Path(os.environ["PREDICTION_LOG"])
    before = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
    client.post("/predict", json=SIMPLE_CASE)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == before + 1
    assert json.loads(lines[-1])["input"]["crime_type"] == "Theft"
