"""Shared constants and the data contract.

This module has no third-party imports so that the data generator, the
training script, the notebook and the FastAPI app can all import it without
pulling in heavy libraries.
"""

from __future__ import annotations

# Reference date used to decide whether a case is "pending" and for how long.
# Fixed (not "today") so every run of the notebook gives the same answers.
AS_OF_DATE = "2026-09-30"

# Chargesheet deadline used for the supervised target (CrPC 173 / BNSS 193
# practice: 60/90 days depending on the offence; 90 is the outer limit).
CHARGESHEET_DEADLINE_DAYS = 90

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Police stations. Names come from the places in the 5 provided files
# (tower locations, addresses, FIR text). Coordinates are approximate area
# centroids used only for the synthetic map.
# ---------------------------------------------------------------------------
STATIONS = {
    "Bhawarkuan":    {"code": "BHK", "district": "Indore", "lat": 22.6960, "lon": 75.8640},
    "Rajwada":       {"code": "RJW", "district": "Indore", "lat": 22.7186, "lon": 75.8553},
    "Chandan Nagar": {"code": "CHN", "district": "Indore", "lat": 22.7175, "lon": 75.8190},
    "Rau":           {"code": "RAU", "district": "Indore", "lat": 22.6310, "lon": 75.8110},
    "Palasia":       {"code": "PLS", "district": "Indore", "lat": 22.7245, "lon": 75.8850},
    "Vijay Nagar":   {"code": "VJN", "district": "Indore", "lat": 22.7510, "lon": 75.8950},
    "Sector 12":     {"code": "S12", "district": "Indore", "lat": 22.6830, "lon": 75.8330},
    "Dewas Naka":    {"code": "DWN", "district": "Indore", "lat": 22.7640, "lon": 75.9010},
    "Karol Bagh":    {"code": "KRB", "district": "Delhi",  "lat": 28.6519, "lon": 77.1909},
}
STATION_NAMES = list(STATIONS)

CRIME_TYPES = [
    "Theft", "Vehicle Theft", "Chain Snatching", "Burglary", "Robbery",
    "Assault", "Extortion", "Cyber Fraud", "Narcotics", "Money Laundering",
    "Illegal Arms", "Sexual Offence",
]

# Ordinal gravity of each offence (Low < Medium < High). Used as the
# ordinal-encoded feature required by the mid-term guidelines.
GRAVITY_ORDER = ["Low", "Medium", "High"]
OFFENCE_GRAVITY = {
    "Theft": "Low", "Vehicle Theft": "Low", "Chain Snatching": "Medium",
    "Burglary": "Medium", "Robbery": "High", "Assault": "Medium",
    "Extortion": "High", "Cyber Fraud": "Medium", "Narcotics": "High",
    "Money Laundering": "High", "Illegal Arms": "High", "Sexual Offence": "High",
}

PROPERTY_CRIMES = {"Theft", "Vehicle Theft", "Chain Snatching", "Burglary",
                   "Robbery", "Cyber Fraud", "Extortion", "Money Laundering"}

SEXUAL_OFFENCES = {"Sexual Offence"}

# Attributes that must never be used as model features (bias / profiling).
PROTECTED_ATTRIBUTES = ["caste", "religion", "community", "tribe", "ethnicity",
                        "accused_caste", "accused_religion", "complainant_caste",
                        "complainant_religion"]

# Indian seasons (domain-defined bins for month).
SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Summer", 4: "Summer", 5: "Summer",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                   10: "Post-monsoon", 11: "Post-monsoon"}

# Domain-defined bins for hour of day (policing shifts).
TIME_OF_DAY_BINS = [0, 6, 12, 18, 24]
TIME_OF_DAY_LABELS = ["Night (00-06)", "Morning (06-12)",
                      "Afternoon (12-18)", "Evening (18-24)"]

# ---------------------------------------------------------------------------
# Model feature contract (enforced by the Pydantic schema in app.py)
# ---------------------------------------------------------------------------
TARGET = "chargesheet_delayed"

NUMERIC_FEATURES = ["num_accused", "num_witnesses", "io_caseload",
                    "registration_delay_days"]
SKEWED_FEATURES = ["property_value_inr"]
BINARY_FEATURES = ["has_cctv", "has_cdr_evidence", "has_financial_trail",
                   "forensic_required", "repeat_offender_involved"]
ORDINAL_FEATURES = ["offence_gravity"]
NOMINAL_FEATURES = ["crime_type", "police_station", "season", "time_of_day"]

MODEL_FEATURES = (NUMERIC_FEATURES + SKEWED_FEATURES + BINARY_FEATURES
                  + ORDINAL_FEATURES + NOMINAL_FEATURES)

# Raw inputs accepted by the API (derived features are computed from these).
API_LIMITS = {
    "hour_of_occurrence": (0, 23),
    "month": (1, 12),
    "num_accused": (1, 15),
    "num_witnesses": (0, 30),
    "property_value_inr": (0.0, 50_000_000.0),
    "io_caseload": (1, 60),
    "registration_delay_days": (0, 365),
}
