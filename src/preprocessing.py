"""Cleaning, merging, IPC-to-BNS mapping, entity resolution and anonymisation."""

from __future__ import annotations

import difflib
import hashlib
import re

import numpy as np
import pandas as pd

from src.contract import (
    AS_OF_DATE, CHARGESHEET_DEADLINE_DAYS, OFFENCE_GRAVITY, PROPERTY_CRIMES,
    PROTECTED_ATTRIBUTES, SEASON_OF_MONTH, SEXUAL_OFFENCES, STATION_NAMES,
    TARGET, TIME_OF_DAY_BINS, TIME_OF_DAY_LABELS,
)

BINARY_YES_NO = ["has_cctv", "has_cdr_evidence", "has_financial_trail", "forensic_required"]

# ---------------------------------------------------------------------------
# IPC -> BNS (Bharatiya Nyaya Sanhita, 2023, in force from 1 July 2024).
# Common sections only. Verify against the official MHA comparison table
# before any real use. Special Acts (NDPS, Arms, PMLA, IT Act) are unchanged.
# ---------------------------------------------------------------------------
IPC_TO_BNS = {
    "IPC 34": "BNS 3(5)", "IPC 120B": "BNS 61(2)", "IPC 302": "BNS 103(1)",
    "IPC 307": "BNS 109", "IPC 323": "BNS 115(2)", "IPC 324": "BNS 118(1)",
    "IPC 354": "BNS 74", "IPC 356": "BNS 304", "IPC 376": "BNS 64",
    "IPC 379": "BNS 303(2)", "IPC 380": "BNS 305", "IPC 384": "BNS 308(2)",
    "IPC 392": "BNS 309(4)", "IPC 395": "BNS 310(2)", "IPC 406": "BNS 316(2)",
    "IPC 420": "BNS 318(4)", "IPC 457": "BNS 331(4)", "IPC 506": "BNS 351(2)",
}


def ipc_to_bns(sections: str) -> str:
    """Map a '; '-separated list of IPC sections to BNS (other Acts kept as-is)."""
    if not isinstance(sections, str) or not sections.strip():
        return ""
    parts = [p.strip() for p in sections.split(";")]
    return "; ".join(IPC_TO_BNS.get(p, p) for p in parts)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
_STATION_KEYS = {re.sub(r"[^a-z]", "", s.lower()): s for s in STATION_NAMES}


def normalize_station(raw) -> str | float:
    """'BHAWARKUAN P.S.' / ' bhawarkuan ' / 'Bhanwarkuan' -> 'Bhawarkuan'."""
    if not isinstance(raw, str) or not raw.strip():
        return np.nan
    key = re.sub(r"\b(p\.?s\.?|police station|thana)\b", "", raw.lower())
    key = re.sub(r"[^a-z]", "", key)
    if key in _STATION_KEYS:
        return _STATION_KEYS[key]
    match = difflib.get_close_matches(key, list(_STATION_KEYS), n=1, cutoff=0.8)
    return _STATION_KEYS[match[0]] if match else raw.strip().title()


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Parse ISO (YYYY-MM-DD) first, then Indian DD/MM/YYYY.

    pandas' format='mixed' with dayfirst=True silently swaps day and month in
    ISO strings such as 2026-08-12, so each format is parsed explicitly.
    """
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    indian = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(indian)


def drop_protected_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove caste/religion/community style columns if any are present."""
    cols = [c for c in df.columns if c.lower() in PROTECTED_ATTRIBUTES]
    return df.drop(columns=cols)


def clean_cases(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean the FIR case register. Returns (clean_df, log of what was done).

    Only rule-based fixes that learn nothing from the data happen here, so
    they are safe before the train/test split. Learned steps (imputation,
    scaling, encoding) live inside the sklearn Pipeline and are fitted on the
    training split only.
    """
    log: list[str] = []
    df = drop_protected_attributes(raw.copy())

    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Dropped {n0 - len(df)} exact duplicate rows (same FIR keyed twice).")

    before = df["police_station"].nunique()
    df["police_station"] = df["police_station"].map(normalize_station)
    log.append(f"Standardised police_station spellings: {before} -> {df['police_station'].nunique()} unique values.")

    for col in ["date_registered", "date_occurred", "chargesheet_date"]:
        df[col] = parse_mixed_dates(df[col].astype("string"))
    log.append("Parsed date_registered (mixed ISO and DD/MM/YYYY), date_occurred, chargesheet_date.")

    bad_acc = df["num_accused"] > 15
    df.loc[bad_acc, "num_accused"] = np.nan
    log.append(f"num_accused: {int(bad_acc.sum())} impossible values (99) set to NaN -> imputed later (data-entry error).")
    bad_delay = df["registration_delay_days"] < 0
    df.loc[bad_delay, "registration_delay_days"] = np.nan
    log.append(f"registration_delay_days: {int(bad_delay.sum())} negative values set to NaN (FIR cannot precede occurrence).")

    non_prop = ~df["crime_type"].isin(PROPERTY_CRIMES)
    df.loc[non_prop, "property_value_inr"] = df.loc[non_prop, "property_value_inr"].fillna(0.0)
    log.append("property_value_inr: non-property offences set to constant 0 (not applicable, not unknown).")

    for c in BINARY_YES_NO:
        df[c] = df[c].map({"Yes": 1, "No": 0}).astype("Int64")
    log.append(f"Binary Yes/No -> 1/0 for {BINARY_YES_NO}.")

    df["ipc_sections"] = df["ipc_sections"].fillna("")
    df["bns_sections"] = df["ipc_sections"].map(ipc_to_bns)
    log.append("Added bns_sections (IPC -> BNS 2023 mapping).")

    for c in ["victim_name", "court_status"]:
        df[c] = df[c].fillna("")
    return df, log


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features + domain-defined bins (season, time of day)."""
    out = df.copy()
    d = out["date_occurred"]
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["week"] = d.dt.to_period("W-SUN").dt.start_time
    out["weekday"] = d.dt.day_name()
    out["is_weekend"] = (d.dt.day_of_week >= 5).astype(int)
    out["season"] = out["month"].map(SEASON_OF_MONTH)
    out["time_of_day"] = pd.cut(out["hour_of_occurrence"], bins=TIME_OF_DAY_BINS,
                                labels=TIME_OF_DAY_LABELS, right=False).astype("object")
    out["offence_gravity"] = out["crime_type"].map(OFFENCE_GRAVITY)
    return out


def split_accused(s: str) -> list[str]:
    if not isinstance(s, str) or s.strip() in {"", "Unknown"}:
        return []
    return [a.strip() for a in s.split(";") if a.strip()]


def add_repeat_offender_flag(cases: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """1 if any accused already had a documented case *before* this FIR.

    Uses only existing records: earlier FIRs in the register, or prior cases
    listed in criminal_history.csv. Never a prediction about a person.
    """
    out = cases.sort_values("date_registered").copy()
    with_priors = set(history.loc[history["prior_cases"].fillna("").str.strip() != "", "name"])
    seen: set[str] = set()
    flags = []
    for names in out["accused_names"].map(split_accused):
        flags.append(int(any(n in seen or n in with_priors for n in names)))
        seen.update(names)
    out["repeat_offender_involved"] = flags
    return out.sort_index()


def add_case_outcomes(df: pd.DataFrame, as_of: str = AS_OF_DATE) -> pd.DataFrame:
    """Days to chargesheet, case age, and the supervised target.

    target = 1  chargesheet filed after 90 days, or still pending after 90 days
    target = 0  chargesheet filed within 90 days
    target = NaN pending but younger than 90 days, or closed with a final
                 report (outcome unknown / not applicable): excluded from training
    """
    out = df.copy()
    as_of_ts = pd.Timestamp(as_of)
    out["days_to_chargesheet"] = (out["chargesheet_date"] - out["date_registered"]).dt.days
    out["case_age_days"] = (as_of_ts - out["date_registered"]).dt.days
    y = pd.Series(np.nan, index=out.index)
    filed = out["days_to_chargesheet"].notna()
    y[filed] = (out.loc[filed, "days_to_chargesheet"] > CHARGESHEET_DEADLINE_DAYS).astype(float)
    pending = out["case_status"].eq("Under Investigation")
    y[pending & (out["case_age_days"] > CHARGESHEET_DEADLINE_DAYS)] = 1.0
    out[TARGET] = y
    return out


def prepare_cases(raw: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Full rule-based preparation used by the notebook, train.py and tests."""
    df, log = clean_cases(raw)
    df = add_time_features(df)
    df = add_repeat_offender_flag(df, history)
    df = add_case_outcomes(df)
    log.append(f"Target '{TARGET}': {int(df[TARGET].notna().sum())} labelled rows, "
               f"{int(df[TARGET].isna().sum())} unlabelled (pending < 90 days or closed) excluded from training.")
    return df, log


# ---------------------------------------------------------------------------
# Entity resolution for the 5 source files (names, aliases, phones, vehicles)
# ---------------------------------------------------------------------------
PHONE_RE = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")
VEHICLE_RE = re.compile(r"\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}\b")
ABBREVIATIONS = {r"\bkr\.?(?=\s|$)": "kumar", r"\bmd\.?(?=\s|$)": "mohammad"}


def _norm_name(name: str) -> str:
    s = name.lower().strip()
    for pat, rep in ABBREVIATIONS.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(sorted(s.split()))  # token-sorted: 'Rathore Vikram' == 'Vikram Rathore'


class PersonRegistry:
    """Resolve name variants and aliases to one canonical person."""

    def __init__(self, history: pd.DataFrame, extra_names: list[str] | None = None):
        self.canonical: dict[str, str] = {}
        self.alias_of: dict[str, str] = {}
        self.phone_of: dict[str, str] = {}
        self.vehicle_of: dict[str, str] = {}
        for _, r in history.iterrows():
            self._add(r["name"])
            if isinstance(r.get("aliases"), str) and r["aliases"].strip():
                for a in r["aliases"].split(","):
                    self.alias_of[a.strip().lower()] = r["name"]
            if isinstance(r.get("phone"), str):
                self.phone_of[r["phone"]] = r["name"]
            if isinstance(r.get("vehicle"), str) and r["vehicle"].strip():
                self.vehicle_of[r["vehicle"].strip()] = r["name"]
        for n in extra_names or []:
            self._add(n)

    def _add(self, name: str) -> None:
        self.canonical.setdefault(_norm_name(name), name)

    def resolve(self, raw: str, cutoff: float = 0.85) -> str | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
        if raw.strip().lower() in self.alias_of:
            return self.alias_of[raw.strip().lower()]
        key = _norm_name(raw)
        if key in self.canonical:
            return self.canonical[key]
        m = difflib.get_close_matches(key, list(self.canonical), n=1, cutoff=cutoff)
        return self.canonical[m[0]] if m else None

    def resolution_table(self, raw_names: list[str]) -> pd.DataFrame:
        rows = [{"raw_name": r, "resolved_to": self.resolve(r)} for r in sorted(set(raw_names))]
        return pd.DataFrame(rows)

    def find_in_text(self, text: str) -> list[str]:
        """Find known people in free text, including aliases and 'Ramesh Kr.' forms."""
        found = []
        low = text.lower()
        for key, name in self.canonical.items():
            first, last = name.split()[0], name.split()[-1]
            pats = [re.escape(name.lower()), re.escape(f"{last} {first}".lower())]
            if last.lower() == "kumar":
                pats.append(re.escape(first.lower()) + r"\s+kr\.?")
            if any(re.search(r"\b" + p, low) for p in pats):
                found.append(name)
        for alias, name in self.alias_of.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", low) and name not in found:
                found.append(name)
        return found


def extract_entities(text: str, registry: PersonRegistry) -> dict[str, list[str]]:
    """Rule-based entity extraction for FIR / intelligence / social-media text."""
    phones = PHONE_RE.findall(text)
    vehicles = VEHICLE_RE.findall(text)
    people = registry.find_in_text(text)
    people += [registry.phone_of[p] for p in phones if p in registry.phone_of and registry.phone_of[p] not in people]
    people += [registry.vehicle_of[v] for v in vehicles if v in registry.vehicle_of and registry.vehicle_of[v] not in people]
    places = [s for s in STATION_NAMES if s.lower() in text.lower()]
    orgs = re.findall(r"\b([A-Z][a-z]+ (?:Traders|Enterprises|Pvt\.? Ltd\.?|Firm))\b", text)
    return {"persons": people, "phones": phones, "vehicles": vehicles,
            "places": places, "organisations": sorted(set(orgs))}


# ---------------------------------------------------------------------------
# Anonymisation
# ---------------------------------------------------------------------------
def pseudonym(value: str, salt: str, prefix: str = "P") -> str:
    """Stable salted hash -> 'P-1a2b3c4d'. Same person -> same ID across files."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return f"{prefix}-{hashlib.sha256((salt + value.strip().lower()).encode()).hexdigest()[:8]}"


def mask_phone(p) -> str:
    s = re.sub(r"\D", "", str(p)) if p is not None else ""
    return ("X" * (len(s) - 2) + s[-2:]) if len(s) >= 4 else ""


def mask_address(addr: str) -> str:
    """Keep only area + district (drop house number and ward)."""
    if not isinstance(addr, str):
        return ""
    parts = [p.strip() for p in addr.split(",")]
    return ", ".join(parts[-2:])


def redact_text(text: str, names: list[str], salt: str) -> str:
    if not isinstance(text, str):
        return text
    out = PHONE_RE.sub(lambda m: mask_phone(m.group()), text)
    for n in sorted({n for n in names if n}, key=len, reverse=True):
        out = re.sub(re.escape(n), pseudonym(n, salt), out, flags=re.IGNORECASE)
    return out


def anonymize_cases(df: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (anonymised cases, restricted lookup table).

    * accused names -> salted pseudonyms (analysis still links the same person)
    * complainant name/phone/address -> pseudonym / masked / area only
    * sexual offences: victim and complainant identity removed completely,
      address reduced to district, text redacted (BNS 72 / IPC 228A)
    The lookup table must stay in a restricted, access-logged location and is
    git-ignored (models/restricted/).
    """
    out = df.copy()
    lookup = []
    for names in out["accused_names"].map(split_accused):
        for n in names:
            lookup.append({"pseudonym": pseudonym(n, salt), "name": n, "role": "accused"})
    sexual = out["crime_type"].isin(SEXUAL_OFFENCES)

    out["accused_ids"] = out["accused_names"].map(
        lambda s: "; ".join(pseudonym(n, salt) for n in split_accused(s)) or "Unknown")
    out["complainant_id"] = out["complainant_name"].map(lambda n: pseudonym(n, salt, "C"))
    out.loc[sexual, "complainant_id"] = "[PROTECTED]"
    out["complainant_phone"] = out["complainant_phone"].map(mask_phone)
    out.loc[sexual, "complainant_phone"] = "[PROTECTED]"
    out["complainant_area"] = out["complainant_address"].map(mask_address)
    out.loc[sexual, "complainant_area"] = out.loc[sexual, "district"]
    out["victim_identity"] = np.where(out["victim_name"].astype(str).str.len() > 0, "[WITHHELD]", "")
    out.loc[sexual, "victim_identity"] = "[PROTECTED - sexual offence]"
    all_names = [n for s in out["accused_names"] for n in split_accused(s)]
    out["modus_operandi"] = [
        redact_text(t, all_names + [c, v], salt)
        for t, c, v in zip(out["modus_operandi"], out["complainant_name"], out["victim_name"])
    ]
    out = out.drop(columns=["accused_names", "complainant_name", "complainant_address", "victim_name"])
    lookup_df = pd.DataFrame(lookup).drop_duplicates().reset_index(drop=True)
    return out, lookup_df
