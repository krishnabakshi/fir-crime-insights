"""Score newly registered FIRs against the trained models and the records.

For each new FIR this returns, in the same Insight -> Evidence -> Next step
-> Priority format as the main report:
* chargesheet-delay risk (trained classifier)
* most similar earlier FIRs by modus operandi (possible same offender/gang)
* accused already named in earlier FIRs (documented records only)
* whether the location falls inside a known hotspot
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.hotspots import to_km
from src.insights_report import InsightBook
from src.model_pipeline import to_model_frame
from src.preprocessing import PHONE_RE, VEHICLE_RE, add_time_features, clean_cases, split_accused


def prepare_new_firs(new_raw: pd.DataFrame, history_cases: pd.DataFrame) -> pd.DataFrame:
    """Clean new FIRs with the same rules and add record-based features."""
    df, _ = clean_cases(new_raw)
    df = add_time_features(df)
    known = {n for s in history_cases["accused_names"] for n in split_accused(s)}
    df["repeat_offender_involved"] = df["accused_names"].map(
        lambda s: int(any(n in known for n in split_accused(s))))
    return df


def analyse_new_firs(new: pd.DataFrame, history_cases: pd.DataFrame, matcher, model,
                     hotspot_summary: pd.DataFrame, label=lambda s: s,
                     sim_threshold: float = 0.8, max_days: int = 365, top_k: int = 3) -> pd.DataFrame:
    book = InsightBook(label)
    risk = model.predict_proba(to_model_frame(new))[:, 1]
    hist_sim = matcher.similarity(new["modus_operandi"], history_cases["modus_operandi"])
    hs_xy = to_km(hotspot_summary["lat"], hotspot_summary["lon"], 22.7) if len(hotspot_summary) else np.empty((0, 2))
    for i, (_, r) in enumerate(new.iterrows()):
        fir = r["fir_no"]
        # 1. delay risk
        p = float(risk[i])
        book.add("New FIR: case progress",
                 f"{fir} ({r['crime_type']}, {r['police_station']}): {p:.0%} risk of missing the 90-day chargesheet deadline.",
                 f"Accused {int(r['num_accused']) if pd.notna(r['num_accused']) else 'n/a'}, IO caseload {r['io_caseload']}, "
                 f"forensic {'yes' if r['forensic_required'] else 'no'}, financial trail {'yes' if r['has_financial_trail'] else 'no'}.",
                 "Assign a supervising officer and set interim deadlines." if p >= 0.6 else "Routine monitoring.",
                 0.35 + 0.5 * p)
        # 2. MO matches among same crime type within the last `max_days`
        age = (r["date_occurred"] - history_cases["date_occurred"]).dt.days.to_numpy()
        ok = (history_cases["crime_type"] == r["crime_type"]).to_numpy() & (age >= 0) & (age <= max_days)
        sims = np.where(ok, hist_sim[i], 0.0)
        top = [j for j in np.argsort(sims)[::-1][:top_k] if sims[j] >= sim_threshold]
        if top:
            firs = history_cases.iloc[top]["fir_no"].tolist()
            ids = set(VEHICLE_RE.findall(r["modus_operandi"])) | set(PHONE_RE.findall(r["modus_operandi"]))
            shared = sorted({x for j in top for x in ids if x in history_cases.iloc[j]["modus_operandi"]})
            best = float(sims[top[0]])
            book.add("New FIR: pattern (MO)",
                     f"{fir} matches the modus operandi of {len(top)} FIR(s) from the last {max_days} days"
                     + (f", including the same vehicle/phone {', '.join(shared)}." if shared else "."),
                     "; ".join(f"{f} (similarity {sims[j]:.2f})" for f, j in zip(firs, top)),
                     f"Link FIR {fir} and FIR {firs[0]} for joint investigation; compare suspects, vehicles and CCTV.",
                     (0.9 if shared else 0.3 + 0.4 * best))
        # 3. accused already on record
        for name in split_accused(r["accused_names"]):
            prior = history_cases[history_cases["accused_names"].map(lambda s: name in split_accused(s))]
            if len(prior):
                book.add("New FIR: repeat offender",
                         f"Accused {label(name)} in {fir} is already named in {len(prior)} earlier FIR(s).",
                         ", ".join(prior.sort_values("date_registered")["fir_no"].tail(5)),
                         "Attach the earlier case records to this investigation (documented cases only).",
                         0.5 + 0.03 * min(len(prior), 10))
        # 4. inside a hotspot?
        if len(hs_xy) and pd.notna(r.get("latitude")):
            d = np.linalg.norm(hs_xy - to_km(pd.Series([r["latitude"]]), pd.Series([r["longitude"]]), 22.7), axis=1)
            j = int(d.argmin())
            if d[j] <= max(float(hotspot_summary.iloc[j]["radius_km"]), 0.3):
                h = hotspot_summary.iloc[j]
                book.add("New FIR: hotspot",
                         f"{fir} occurred inside hotspot H{h['hotspot']} near {h['top_station']}.",
                         f"{h['firs']} earlier FIRs in this hotspot, mostly {h['top_crime']}.",
                         f"Add this location to the {h['top_time']} patrol route.", 0.5)
    return book.to_frame()
