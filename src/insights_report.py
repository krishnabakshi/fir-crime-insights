"""Turn analysis outputs into ranked insights:
Insight -> Evidence (FIR numbers / data) -> Suggested Next Step -> Priority.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Callable

import pandas as pd

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]
PRIORITY_COLOUR = {"Critical": "#c0392b", "High": "#e67e22", "Medium": "#f1c40f", "Low": "#7f8c8d"}


def priority_of(score: float) -> str:
    """Score in [0, 1] -> priority band."""
    if score >= 0.85:
        return "Critical"
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


class InsightBook:
    def __init__(self, label: Callable[[str], str] = lambda s: s):
        self.rows: list[dict] = []
        self.label = label  # maps a real name to what may be shown (pseudonym)

    def add(self, category: str, insight: str, evidence: str, next_step: str, score: float) -> None:
        score = max(0.0, min(1.0, float(score)))
        self.rows.append({"category": category, "insight": insight, "evidence": evidence,
                          "suggested_next_step": next_step, "priority": priority_of(score),
                          "score": round(score, 3)})

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        df["priority"] = pd.Categorical(df["priority"], PRIORITY_ORDER, ordered=True)
        df = df.sort_values(["priority", "score"], ascending=[True, False]).reset_index(drop=True)
        df.insert(0, "insight_id", [f"INS-{i + 1:03d}" for i in range(len(df))])
        df["priority"] = df["priority"].astype(str)
        return df


def _fir_sample(firs, n: int = 4) -> str:
    firs = list(firs)
    more = f" (+{len(firs) - n} more)" if len(firs) > n else ""
    return ", ".join(firs[:n]) + more


# ---------------------------------------------------------------------------
# Rule sets: one function per analysis
# ---------------------------------------------------------------------------
# What to do about a rise, by kind of crime (patrolling does not stop cyber fraud)
ACTION_FOR_CRIME = {
    "Cyber Fraud": "Run cyber-safety awareness camps in {st}, share the fraud numbers with banks and the cyber cell, "
                   "and freeze mule accounts quickly through the 1930 helpline workflow.",
    "Money Laundering": "Ask the economic offences wing to review accounts linked to {st} cases and request bank statements early.",
    "Narcotics": "Increase {tod} checking at entry points and highway dhabas near {st}; work with the narcotics cell.",
    "Extortion": "Visit shopkeepers in {st}, share a direct helpline, and step up {tod} patrolling in market areas.",
    "Sexual Offence": "Deploy women patrol teams and check lighting and CCTV at {st} spots used at {tod}.",
}
DEFAULT_ACTION = "Increase {tod} patrolling and beat checks for {crime} in {st}; brief the station on the latest MO."


def next_step_for(crime: str, st: str, tod: str) -> str:
    band = tod.split(" ")[0].lower() if isinstance(tod, str) else "peak-hour"
    return ACTION_FOR_CRIME.get(crime, DEFAULT_ACTION).format(st=st, tod=band, crime=crime.lower())


def add_trend_insights(book: InsightBook, cases: pd.DataFrame, area_trends: pd.DataFrame,
                       city_trends: pd.DataFrame, peaks: pd.DataFrame, min_recent: int = 10) -> None:
    window = area_trends.attrs.get("window", "")
    pk = peaks.set_index("police_station")
    recent_start = cases["date_occurred"].max().to_period("M") - 5
    in_window = cases["date_occurred"].dt.to_period("M") >= recent_start
    rises = area_trends[(area_trends["direction"] == "Rise") & (area_trends["recent"] >= min_recent)]
    for _, r in rises.head(5).iterrows():
        st, crime = r["police_station"], r["crime_type"]
        recent = cases[in_window & (cases["police_station"] == st) & (cases["crime_type"] == crime)]
        recent = recent.sort_values("date_occurred", ascending=False)
        tod = recent["time_of_day"].mode().iloc[0] if recent["time_of_day"].notna().any() else pk.loc[st, "peak_time_of_day"]
        score = 0.4 + 0.25 * min(r["change_pct"], 200) / 200 + 0.25 * min(r["recent"], 40) / 40
        book.add("Crime trend",
                 f"{crime} is rising in {st}: {r['recent']} FIRs vs {r['year_before']} a year earlier ({r['change_pct']:+.0f}%).",
                 f"Window {window}; mostly {tod}. Recent FIRs: {_fir_sample(recent['fir_no'])}",
                 next_step_for(crime, st, tod), score)
    for _, r in city_trends[city_trends["direction"] == "Rise"].head(3).iterrows():
        book.add("Crime trend",
                 f"City-wide {r['crime_type']} up {r['change_pct']:+.0f}% year on year ({r['recent']} vs {r['year_before']}).",
                 f"Window {window}; all stations combined.",
                 f"Assign a nodal officer for {r['crime_type'].lower()} and review the rise at the monthly crime meeting.",
                 0.45 + 0.35 * min(r["change_pct"], 150) / 150)
    for _, r in city_trends[city_trends["direction"] == "Fall"].head(2).iterrows():
        book.add("Crime trend",
                 f"{r['crime_type']} is falling city-wide ({r['change_pct']:+.0f}%).",
                 f"Window {window}: {r['recent']} vs {r['year_before']} FIRs.",
                 "Keep the current preventive measures; review again next quarter.", 0.2)


def add_hotspot_insights(book: InsightBook, hotspots: pd.DataFrame, incidents: pd.DataFrame) -> None:
    top_recent = max(int(hotspots["last_90d"].max()), 1)
    for _, r in hotspots.sort_values("last_90d", ascending=False).head(5).iterrows():
        firs = incidents[incidents["hotspot"] == r["hotspot"]].sort_values("date_occurred", ascending=False)["fir_no"]
        score = 0.45 + 0.4 * r["last_90d"] / top_recent
        book.add("Hotspot",
                 f"Hotspot near {r['top_station']} (H{r['hotspot']}): {r['firs']} FIRs within ~{r['radius_km']} km, "
                 f"{r['last_90d']} in the last 90 days; mostly {r['top_crime']}.",
                 f"Centre {r['lat']:.4f}, {r['lon']:.4f}; density {r['density_per_km2']}/km2. FIRs: {_fir_sample(firs)}",
                 f"Station a PCR van / picket at this spot during {r['top_time']}; add CCTV coverage and street lighting check.",
                 score)


def add_forecast_insights(book: InsightBook, forecast: pd.DataFrame) -> None:
    for _, r in forecast.head(3).iterrows():
        score = 0.75 if r["risk_rank"] == 1 else 0.6
        book.add("Prediction",
                 f"{r['police_station']} is forecast to have the most FIRs in the next 4 weeks "
                 f"(~{r['forecast_next_weeks']}, rank {r['risk_rank']}).",
                 f"Model {r['best_model']} (backtest MAE {r['backtest_mae']}/week); recent average {r['recent_12wk_avg']}/week; "
                 f"forecast from {r['forecast_start']}.",
                 f"Plan the patrol roster for {r['police_station']}: extra staff on {r['peak_weekday']}s, "
                 f"{r['peak_time_of_day']}.", score)


def add_mo_insights(book: InsightBook, series: pd.DataFrame, eval_info: dict) -> None:
    for _, r in series.head(6).iterrows():
        firs = r["fir_numbers"].split(", ")
        score = 0.55 + 0.1 * min(r["n_stations"], 3) + 0.01 * min(r["series_size"], 15) + (0.1 if r["pending_in_series"] else 0)
        book.add("Pattern (MO)",
                 f"{r['series_size']} {r['crime_type']} FIRs across {r['n_stations']} station(s) share the same modus operandi "
                 f"({r['first_date']} to {r['last_date']}).",
                 f"FIRs: {_fir_sample(firs, 5)}. MO: \"{r['example_mo']}\". "
                 f"Matched pairs naming a common accused: {eval_info['matched_share_common_accused']:.0%} "
                 f"(random pairs: {eval_info['random_share_common_accused']:.0%}).",
                 f"Link FIR {firs[0]} and FIR {firs[1]} (and the rest of the series) for joint investigation by "
                 f"{r['stations']}; compare suspects, vehicles and CCTV.", score)


def add_network_insights(book: InsightBook, cent: pd.DataFrame, hidden: pd.DataFrame, cycles: pd.DataFrame,
                         bursts: pd.DataFrame) -> None:
    lab = book.label
    for rank, (_, r) in enumerate(cent[cent["type"] == "person"].head(2).iterrows(), start=1):
        book.add("Network",
                 f"{lab(r['node'])} is the {'most' if rank == 1 else 'second most'} central broker in the network "
                 f"(betweenness {r['betweenness']}, {r['degree']} direct links).",
                 f"Link types: {r['link_types']}; community {r['community']}.",
                 f"Prioritise CDR, financial and associate analysis of {lab(r['node'])}; seek supervisory review "
                 f"before any action.", 0.8 if rank == 1 else 0.66)
    for _, r in cycles.iterrows():
        book.add("Network",
                 f"Money moved in a closed loop: {' -> '.join(lab(x) for x in r['holders'].split(' -> '))} "
                 f"({r['retained_%']}% of the amount came back).",
                 f"Accounts {r['accounts']}; amounts {r['amounts']}; {r['first_leg']} to {r['last_leg']}.",
                 "Refer the accounts for financial-intelligence analysis and request bank statements; "
                 "consider a freeze request under the applicable law.", 0.9)
    for _, r in bursts.iterrows():
        book.add("Network",
                 f"{lab(r['caller'])} made {r['short_calls']} very short calls (~{r['avg_seconds']} s) to "
                 f"{lab(r['callee'])} within hours.",
                 f"CDR {r['from']} to {r['to']}, tower {r['tower']}.",
                 "Check incidents reported around that time and place; obtain the tower dump for the window.", 0.55)
    for _, r in hidden.head(3).iterrows():
        book.add("Network",
                 f"Possible unrecorded link between {lab(r['person_a'])} and {lab(r['person_b'])}.",
                 f"Adamic-Adar {r['adamic_adar']}; common contacts: {', '.join(lab(x) for x in r['common_contacts'].split(', '))}.",
                 "Verify through field enquiry before recording any association.", 0.42)


def add_repeat_offender_insights(book: InsightBook, rep: pd.DataFrame, top: int = 5) -> None:
    lab = book.label
    for _, r in rep.head(top).iterrows():
        cross = r["districts"] > 1 or r["stations"] >= 3
        score = 0.42 + 0.025 * min(r["fir_count"], 10) + (0.1 if cross else 0) + (0.05 if r["pending_cases"] else 0)
        book.add("Repeat offender",
                 f"{lab(r['name'])} is named in {r['fir_count']} FIRs across {r['stations']} station(s) "
                 f"and {r['districts']} district(s) ({r['crime_types']}).",
                 f"Latest FIRs: {r['fir_numbers']}. Pending: {r['pending_cases']}. "
                 f"Prior record: {r['prior_record_in_history'] or 'none in history file'}.",
                 "Consolidate these cases with one supervising officer; update the history sheet from these records "
                 "(documented cases only)." + (" Share with the other station(s)/district." if cross else ""), score)


def add_case_progress_insights(book: InsightBook, overdue: pd.DataFrame, pend_st: pd.DataFrame,
                               conv: pd.DataFrame, io: pd.DataFrame, risk: pd.DataFrame | None) -> None:
    if len(overdue):
        very_old = overdue[overdue["case_age_days"] > 180]
        book.add("Case progress",
                 f"{len(overdue)} investigations are pending beyond 90 days without a chargesheet "
                 f"({len(very_old)} beyond 180 days).",
                 f"Oldest: {_fir_sample(overdue['fir_no'], 6)}",
                 "Prioritise chargesheets pending over 90 days: weekly review by SHO/ACP, starting with the oldest.",
                 0.92 if len(very_old) else 0.75)
    for _, r in pend_st.head(2).iterrows():
        if r["overdue_90d"] == 0:
            continue
        book.add("Case progress",
                 f"{r['police_station']} has {r['overdue_90d']} overdue and {r['pending']} pending investigations.",
                 f"{r['pending_%']}% of its {r['total_firs']} FIRs are still under investigation.",
                 f"Hold a case-review meeting at {r['police_station']} and set chargesheet target dates.", 0.62)
    overall = conv.attrs.get("overall_rate", None)
    for _, r in conv.head(3).iterrows():
        book.add("Case progress",
                 f"Low conviction rate for {r['crime_type']}: {r['conviction_rate_%']}% "
                 f"({r['convicted']}/{r['decided']} decided cases; overall {overall}%).",
                 "Court outcomes in the case register.",
                 f"Review evidence collection for {r['crime_type'].lower()} cases (forensics, witness protection, "
                 "digital evidence) with the prosecution.", 0.5 if r["conviction_rate_%"] > 25 else 0.66)
    top_io = io.head(1)
    for _, r in top_io.iterrows():
        if r["overdue_90d"] >= 2:
            book.add("Case progress",
                     f"Investigating officer {r['io_id']} holds {r['overdue_90d']} overdue cases.",
                     f"{r['pending']} pending; median caseload {r['median_caseload']}.",
                     f"Redistribute part of {r['io_id']}'s caseload within {r['police_station']}.", 0.55)
    if risk is not None and len(risk):
        hi = risk[risk["delay_risk"] >= 0.7]
        if len(hi):
            book.add("Prediction",
                     f"{len(hi)} recent cases (under 90 days old) are likely to miss the 90-day chargesheet deadline.",
                     f"Model risk >= 0.70: {_fir_sample(hi.sort_values('delay_risk', ascending=False)['fir_no'], 6)}",
                     "Supervise these cases now: fix IO timelines, fast-track forensic and CDR requests.", 0.8)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def to_html_report(ins: pd.DataFrame, title: str = "FIR Crime Insights Report", extra_html: str = "") -> str:
    rows = []
    for _, r in ins.iterrows():
        c = PRIORITY_COLOUR[r["priority"]]
        rows.append(
            f"<tr><td>{r['insight_id']}</td><td><span class='p' style='background:{c}'>{r['priority']}</span></td>"
            f"<td>{html.escape(r['category'])}</td><td>{html.escape(r['insight'])}</td>"
            f"<td class='ev'>{html.escape(r['evidence'])}</td><td>{html.escape(r['suggested_next_step'])}</td></tr>")
    counts = ins["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    badges = " ".join(f"<span class='p' style='background:{PRIORITY_COLOUR[p]}'>{p}: {n}</span>" for p, n in counts.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}
th{{background:#2c3e50;color:#fff;position:sticky;top:0}}tr:nth-child(even){{background:#f7f7f7}}
.p{{color:#fff;padding:2px 8px;border-radius:10px;font-weight:600;white-space:nowrap}}.ev{{color:#555;font-size:12px}}
.note{{background:#fff8e1;border-left:4px solid #f1c40f;padding:8px 12px;margin:12px 0}}</style></head><body>
<h1>{title}</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M}. {badges}</p>
<div class="note">Synthetic data for academic use. Names are pseudonymised; victim identity in sexual-offence cases is never shown.
Insights about individuals are based only on documented records, never on predictions about who will offend.</div>
{extra_html}
<table><tr><th>ID</th><th>Priority</th><th>Category</th><th>Insight</th><th>Evidence</th><th>Suggested next step</th></tr>
{''.join(rows)}</table></body></html>"""


def build_dashboard(figures: dict, insights: pd.DataFrame, map_file: str | None = None,
                    title: str = "FIR Crime Insights Dashboard") -> str:
    """Single HTML page: KPI tiles, plotly charts, map, top insights."""
    import plotly.io as pio
    blocks = []
    for i, (name, fig) in enumerate(figures.items()):
        blocks.append(f"<div class='card'><h3>{html.escape(name)}</h3>"
                      f"{pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)}</div>")
    mp = (f"<div class='card wide'><h3>Hotspot map</h3><iframe src='{map_file}' width='100%' height='520' "
          f"style='border:0'></iframe></div>") if map_file else ""
    top = insights.head(12)
    trs = "".join(f"<tr><td><span class='p' style='background:{PRIORITY_COLOUR[r.priority]}'>{r.priority}</span></td>"
                  f"<td>{html.escape(r.insight)}</td><td>{html.escape(r.suggested_next_step)}</td></tr>"
                  for r in top.itertuples())
    kpis = insights["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    tiles = "".join(f"<div class='kpi' style='border-top:4px solid {PRIORITY_COLOUR[p]}'><b>{n}</b><br>{p}</div>"
                    for p, n in kpis.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#222}}
header{{background:#2c3e50;color:#fff;padding:14px 24px}}main{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}}
.card{{background:#fff;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}.wide{{grid-column:1/3}}
.kpis{{display:flex;gap:12px;padding:16px 16px 0}}.kpi{{background:#fff;padding:10px 18px;border-radius:8px;text-align:center}}
.kpi b{{font-size:26px}}table{{border-collapse:collapse;font-size:13px;width:100%}}td{{border-bottom:1px solid #eee;padding:6px}}
.p{{color:#fff;padding:2px 8px;border-radius:10px;font-weight:600}}</style></head><body>
<header><h2 style="margin:0">{title}</h2><small>Synthetic data | pseudonymised | generated {datetime.now():%Y-%m-%d %H:%M}</small></header>
<div class="kpis">{tiles}</div><main>
<div class="card wide"><h3>Top insights and next steps</h3><table>{trs}</table></div>
{mp}{''.join(blocks)}</main></body></html>"""
