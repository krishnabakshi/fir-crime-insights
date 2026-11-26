"""Criminal network / link analysis with networkx.

Two sources are combined into one person graph:
* the 5 intelligence files (calls, money trail, FIR/intel co-mentions, tips)
* the FIR case register (people named together as co-accused)
Then: centrality ("kingpin" ranking), communities, hidden-link suggestions,
money round-trips, burst calling, repeat and cross-jurisdiction offenders.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from src.contract import RANDOM_STATE
from src.preprocessing import PersonRegistry, extract_entities, split_accused


def _add_edge(g: nx.Graph, a: str, b: str, kind: str, weight: float = 1.0, evidence: str = "") -> None:
    if not a or not b or a == b:
        return
    if g.has_edge(a, b):
        d = g[a][b]
        d["weight"] += weight
        d["kinds"].add(kind)
        if evidence:
            d["evidence"].add(evidence)
    else:
        g.add_edge(a, b, weight=weight, kinds={kind}, evidence={evidence} if evidence else set())


def build_intel_graph(src: dict[str, pd.DataFrame], registry: PersonRegistry) -> tuple[nx.Graph, pd.DataFrame]:
    """Person/organisation graph from the 5 files. Returns (graph, extracted-entity table)."""
    g = nx.Graph()
    resolve = lambda n: registry.resolve(n) or (n.strip() if isinstance(n, str) else n)  # noqa: E731

    cdr = src["cdr"]
    for (a, b), grp in cdr.groupby([cdr["caller_name"].map(resolve), cdr["callee_name"].map(resolve)]):
        _add_edge(g, a, b, "call", weight=len(grp), evidence=f"{len(grp)} calls (CDR)")

    fin = src["financial"]
    for _, r in fin.iterrows():
        _add_edge(g, resolve(r["from_name"]), resolve(r["to_name"]), "money",
                  weight=1 + np.log10(r["amount"]) / 2, evidence=f"Rs {r['amount']:,} {r['mode']} {r['from_account']}->{r['to_account']}")

    ent_rows = []
    for _, r in src["fir_reports"].iterrows():
        ents = extract_entities(r["text"], registry)
        people = ents["persons"] + ents["organisations"]
        ent_rows.append({"source": r["document_id"], **{k: ", ".join(v) for k, v in ents.items()}})
        for a, b in combinations(sorted(set(people)), 2):
            _add_edge(g, a, b, "co-mention", 1.0, r["document_id"])
    for _, r in src["social_media"].iterrows():
        ents = extract_entities(r["text"], registry)
        ent_rows.append({"source": f"{r['platform']}:{r['author']}", **{k: ", ".join(v) for k, v in ents.items()}})
        people = ents["persons"] + ents["organisations"]
        for a, b in combinations(sorted(set(people)), 2):
            _add_edge(g, a, b, "tip", 0.5, f"{r['platform']} tip")
        # vehicle seen with a person's vehicle -> link owners
    hist = src["criminal_history"]
    for _, r in hist.iterrows():
        if r["name"] in g:
            g.nodes[r["name"]]["prior_cases"] = r["prior_cases"] if isinstance(r["prior_cases"], str) else ""
            g.nodes[r["name"]]["home_area"] = r["address"]
    for n in g.nodes:
        g.nodes[n].setdefault("type", "organisation" if n.endswith("Traders") else "person")
    return g, pd.DataFrame(ent_rows)


def add_case_edges(g: nx.Graph, cases: pd.DataFrame, registry: PersonRegistry, only_known: bool = False) -> nx.Graph:
    """Add co-accused edges from the FIR register (optionally only people already in g)."""
    for _, r in cases.iterrows():
        names = [registry.resolve(n) or n for n in split_accused(r["accused_names"])]
        if only_known:
            names = [n for n in names if n in g]
        for a, b in combinations(sorted(set(names)), 2):
            _add_edge(g, a, b, "co-accused", 1.0, r["fir_no"])
    return g


def centrality_table(g: nx.Graph) -> pd.DataFrame:
    """Degree, betweenness (broker / kingpin), PageRank and community per node."""
    deg = dict(g.degree())
    wdeg = dict(g.degree(weight="weight"))
    btw = nx.betweenness_centrality(g, normalized=True)
    pr = nx.pagerank(g, weight="weight")
    comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    df = pd.DataFrame({"node": list(g.nodes)})
    df["type"] = df["node"].map(lambda n: g.nodes[n].get("type", "person"))
    df["degree"] = df["node"].map(deg)
    df["weighted_degree"] = df["node"].map(wdeg).round(2)
    df["betweenness"] = df["node"].map(btw).round(3)
    df["pagerank"] = df["node"].map(pr).round(3)
    df["community"] = df["node"].map(comm_of)
    df["link_types"] = df["node"].map(lambda n: ", ".join(sorted({k for _, _, d in g.edges(n, data=True) for k in d["kinds"]})))
    df["kingpin_score"] = ((df["betweenness"].rank(pct=True) + df["pagerank"].rank(pct=True)) / 2).round(3)
    return df.sort_values(["kingpin_score", "betweenness"], ascending=False).reset_index(drop=True)


def suggest_hidden_links(g: nx.Graph, top: int = 10) -> pd.DataFrame:
    """Link prediction: Adamic-Adar + common neighbours for unconnected pairs."""
    people = [n for n in g if g.nodes[n].get("type") == "person"]
    pairs = [(a, b) for a, b in combinations(people, 2) if not g.has_edge(a, b)]
    rows = []
    for a, b, score in nx.adamic_adar_index(g, pairs):
        common = sorted(nx.common_neighbors(g, a, b))
        if common:
            rows.append({"person_a": a, "person_b": b, "adamic_adar": round(score, 3),
                         "common_contacts": ", ".join(common)})
    out = pd.DataFrame(rows, columns=["person_a", "person_b", "adamic_adar", "common_contacts"])
    return out.sort_values("adamic_adar", ascending=False).head(top).reset_index(drop=True)


def money_cycles(fin: pd.DataFrame) -> pd.DataFrame:
    """Round-trip money flows (A -> B -> C -> A): a classic layering pattern."""
    dg = nx.DiGraph()
    for _, r in fin.iterrows():
        dg.add_edge(r["from_account"], r["to_account"], amount=r["amount"],
                    names=(r["from_name"], r["to_name"]), ts=r["timestamp"])
    rows = []
    for cyc in nx.simple_cycles(dg):
        if len(cyc) < 2:
            continue
        legs = list(zip(cyc, cyc[1:] + cyc[:1]))
        amounts = [dg[a][b]["amount"] for a, b in legs]
        names = [dg[a][b]["names"][0] for a, b in legs]
        rows.append({"accounts": " -> ".join(cyc + [cyc[0]]), "holders": " -> ".join(names + [names[0]]),
                     "amounts": ", ".join(f"{x:,}" for x in amounts),
                     "retained_%": round(min(amounts) / max(amounts) * 100, 1),
                     "first_leg": min(dg[a][b]["ts"] for a, b in legs), "last_leg": max(dg[a][b]["ts"] for a, b in legs)})
    return pd.DataFrame(rows)


def burst_calls(cdr: pd.DataFrame, max_seconds: int = 15, min_calls: int = 3, window_hours: int = 12) -> pd.DataFrame:
    """Repeated very short calls between the same pair (signal / alert calls)."""
    c = cdr.copy()
    c["timestamp"] = pd.to_datetime(c["timestamp"])
    short = c[c["duration_seconds"] <= max_seconds].sort_values("timestamp")
    rows = []
    for (a, b), g in short.groupby(["caller_name", "callee_name"]):
        if len(g) >= min_calls and (g["timestamp"].max() - g["timestamp"].min()) <= pd.Timedelta(hours=window_hours):
            rows.append({"caller": a, "callee": b, "short_calls": len(g),
                         "avg_seconds": round(g["duration_seconds"].mean(), 1),
                         "from": g["timestamp"].min(), "to": g["timestamp"].max(),
                         "tower": g["tower_location"].mode().iloc[0]})
    return pd.DataFrame(rows)


def repeat_offenders(cases: pd.DataFrame, history: pd.DataFrame, registry: PersonRegistry | None = None,
                     min_firs: int = 2) -> pd.DataFrame:
    """People named in >= min_firs FIRs (documented records only)."""
    rec = defaultdict(list)
    for _, r in cases.iterrows():
        for n in split_accused(r["accused_names"]):
            n = registry.resolve(n) if registry and registry.resolve(n) else n
            rec[n].append(r)
    prior = history.set_index("name")["prior_cases"].fillna("").to_dict()
    rows = []
    for name, rs in rec.items():
        if len(rs) < min_firs:
            continue
        sub = pd.DataFrame(rs)
        rows.append({"name": name, "fir_count": len(sub),
                     "crime_types": ", ".join(sub["crime_type"].value_counts().index[:3]),
                     "stations": sub["police_station"].nunique(), "districts": sub["district"].nunique(),
                     "station_list": ", ".join(sorted(sub["police_station"].unique())),
                     "pending_cases": int((sub["case_status"] == "Under Investigation").sum()),
                     "convictions": int((sub["court_status"] == "Convicted").sum()),
                     "first_fir": sub["date_registered"].min().date(), "last_fir": sub["date_registered"].max().date(),
                     "prior_record_in_history": prior.get(name, ""),
                     "fir_numbers": ", ".join(sub.sort_values("date_registered")["fir_no"].tolist()[-8:])})
    out = pd.DataFrame(rows)
    return out.sort_values(["fir_count", "stations"], ascending=False).reset_index(drop=True) if len(out) else out


def plot_network(g: nx.Graph, cent: pd.DataFrame, label_map: dict | None = None, out_html: str | None = None,
                 title: str = "Criminal network"):
    """Interactive plotly network: node size = kingpin score, colour = community."""
    import plotly.graph_objects as go
    pos = nx.spring_layout(g, seed=RANDOM_STATE, weight="weight", k=0.9)
    label = (lambda n: label_map.get(n, n)) if label_map else (lambda n: n)
    ex, ey = [], []
    for a, b, d in g.edges(data=True):
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    edge_tr = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.8, color="#aaaaaa"), hoverinfo="none")
    c = cent.set_index("node")
    nodes = list(g.nodes)
    node_tr = go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=[label(n) for n in nodes], textposition="top center",
        marker=dict(size=[12 + 30 * c.loc[n, "kingpin_score"] for n in nodes],
                    color=[c.loc[n, "community"] for n in nodes], colorscale="Turbo", line=dict(width=1, color="#333")),
        hovertext=[f"{label(n)}<br>degree {c.loc[n, 'degree']} | betweenness {c.loc[n, 'betweenness']}"
                   f"<br>links: {c.loc[n, 'link_types']}" for n in nodes], hoverinfo="text")
    fig = go.Figure([edge_tr, node_tr])
    fig.update_layout(title=title, showlegend=False, height=620, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white")
    if out_html:
        fig.write_html(out_html, include_plotlyjs="cdn")
    return fig
