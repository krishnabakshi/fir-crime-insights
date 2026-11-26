"""Unit tests for cleaning, anonymisation, entity resolution and text normalisation."""

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src.contract import TARGET
from src.mo_matching import MOMatcher, detect_script, normalize_text
from src.preprocessing import (PersonRegistry, add_case_outcomes, add_repeat_offender_flag,
                               anonymize_cases, drop_protected_attributes, extract_entities,
                               ipc_to_bns, mask_phone, normalize_station, parse_mixed_dates,
                               prepare_cases, pseudonym)


@pytest.fixture(scope="module")
def sources():
    return dl.load_sources()


@pytest.fixture(scope="module")
def cases(sources):
    df, _ = prepare_cases(dl.load_cases(), sources["criminal_history"])
    return df


def test_all_five_source_files_load(sources):
    assert set(sources) == {"fir_reports", "social_media", "cdr", "criminal_history", "financial"}
    assert len(sources["criminal_history"]) == 8 and len(sources["cdr"]) == 18


def test_case_register_meets_size_contract(cases):
    assert len(cases) >= 1000
    assert cases["fir_no"].is_unique


@pytest.mark.parametrize("raw,expected", [
    ("BHAWARKUAN P.S.", "Bhawarkuan"), (" bhawarkuan ", "Bhawarkuan"), ("Bhanwarkuan", "Bhawarkuan"),
    ("Vijaynagar", "Vijay Nagar"), ("Rajwada PS", "Rajwada"), ("Dewas naka", "Dewas Naka"),
])
def test_station_names_are_standardised(raw, expected):
    assert normalize_station(raw) == expected


def test_mixed_dates_do_not_swap_day_and_month():
    s = pd.Series(["2026-08-12", "12/08/2026", "03/01/2025"])
    out = parse_mixed_dates(s)
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2026-08-12", "2026-08-12", "2025-01-03"]


def test_cleaning_removes_duplicates_and_impossible_values(cases):
    assert not cases.duplicated(subset=["fir_no"]).any()
    assert cases["num_accused"].max() <= 15
    assert (cases["registration_delay_days"].dropna() >= 0).all()
    assert cases["police_station"].nunique() == 9


def test_ipc_to_bns_mapping():
    assert ipc_to_bns("IPC 379") == "BNS 303(2)"
    assert ipc_to_bns("IPC 384; IPC 506; IPC 34") == "BNS 308(2); BNS 351(2); BNS 3(5)"
    assert ipc_to_bns("NDPS 8/20") == "NDPS 8/20"  # special Acts unchanged


def test_protected_attributes_are_dropped():
    df = pd.DataFrame({"crime_type": ["Theft"], "religion": ["x"], "caste": ["y"]})
    assert list(drop_protected_attributes(df).columns) == ["crime_type"]


def test_target_definition():
    df = pd.DataFrame({
        "date_registered": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01", "2026-09-01"]),
        "chargesheet_date": pd.to_datetime(["2026-02-01", "2026-06-01", None, None]),
        "case_status": ["Chargesheet Filed", "Chargesheet Filed", "Under Investigation", "Under Investigation"],
    })
    y = add_case_outcomes(df, as_of="2026-09-30")[TARGET]
    assert y.iloc[0] == 0 and y.iloc[1] == 1 and y.iloc[2] == 1 and np.isnan(y.iloc[3])


def test_repeat_offender_flag_uses_only_earlier_records():
    cases = pd.DataFrame({"fir_no": ["A", "B", "C"], "accused_names": ["Ravi K", "Ravi K", "New Person"],
                          "date_registered": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])})
    hist = pd.DataFrame({"name": ["Someone"], "prior_cases": ["FIR 1/2020"]})
    flags = add_repeat_offender_flag(cases, hist)["repeat_offender_involved"].tolist()
    assert flags == [0, 1, 0]


def test_alias_and_name_variants_resolve(sources):
    reg = PersonRegistry(sources["criminal_history"], extra_names=["Anita Deshmukh"])
    assert reg.resolve("Ramesh Kr.") == "Ramesh Kumar"
    assert reg.resolve("Rathore Vikram") == "Vikram Rathore"
    assert reg.resolve("Vicky") == "Vikram Rathore"
    assert reg.resolve("Chatterjee Traders") is None


def test_entity_extraction_from_fir_text(sources):
    reg = PersonRegistry(sources["criminal_history"], extra_names=["Anita Deshmukh"])
    ents = extract_entities(sources["fir_reports"].iloc[0]["text"], reg)
    assert {"Vikram Rathore", "Ramesh Kumar", "Suresh Yadav"} <= set(ents["persons"])
    assert "9876543210" in ents["phones"] and "MP09AB1234" in ents["vehicles"]


def test_anonymisation_protects_sexual_offence_victims(cases):
    anon, lookup = anonymize_cases(cases, salt="test-salt")
    for col in ["accused_names", "complainant_name", "complainant_address", "victim_name"]:
        assert col not in anon.columns
    sx = anon[anon["crime_type"] == "Sexual Offence"]
    assert (sx["complainant_id"] == "[PROTECTED]").all()
    assert (sx["victim_identity"] == "[PROTECTED - sexual offence]").all()
    assert not anon["complainant_phone"].str.fullmatch(r"\d{10}").any()
    # same person -> same pseudonym, different salt -> different pseudonym
    assert pseudonym("Vikram Rathore", "s1") == pseudonym("vikram rathore ", "s1") != pseudonym("Vikram Rathore", "s2")
    assert set(lookup["role"]) == {"accused"}


def test_mask_phone_keeps_last_two_digits():
    assert mask_phone("9876543210") == "XXXXXXXX10"


def test_hindi_and_hinglish_are_normalised_to_english():
    hi = "बिना नंबर की काली मोटरसाइकिल पर आए दो युवक महिला की चेन छीनकर भाग गए।"
    hinglish = "Dukaandar se hafta vasooli, paise na dene par dhamki di."
    assert detect_script(hi) == "Hindi (Devanagari)"
    assert detect_script(hinglish) == "Hinglish (romanised)"
    assert {"chain", "snatch", "motorcycle", "black"} <= set(normalize_text(hi).split())
    assert {"shopkeeper", "extortion", "threat"} <= set(normalize_text(hinglish).split())


def test_mo_matcher_scores_same_mo_higher_than_different_mo(cases):
    m = MOMatcher().fit(cases["modus_operandi"])
    texts = pd.Series([
        "Shopkeeper near Rajwada threatened by men in vehicle MP09AB1234; they demanded weekly protection money.",
        "Dukaandar ko Rajwada par MP09AB1234 gaadi mein aaye logon ne hafta vasooli ke liye dhamkaya.",
        "Complainant received a phone call from a person posing as bank officer, shared OTP and lost Rs 45,000.",
    ])
    sim = m.similarity(texts)
    assert sim[0, 1] > sim[0, 2]
