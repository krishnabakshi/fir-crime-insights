"""Modus-operandi (MO) similarity matching for FIR text in English, Hindi and Hinglish.

Approach (works offline on the free Colab tier):
1. Detect the script of each text (Devanagari / Latin).
2. Normalise: Hindi (Devanagari) and Hinglish (romanised Hindi) crime words
   are mapped to English with a small domain glossary, so
   "चेन छीनी", "chain chheen li" and "snatched chain" share tokens.
3. Represent each MO with TF-IDF (word 1-2 grams + character 3-5 grams; the
   character n-grams tolerate spelling variants such as vasooli/vasuli).
4. Optional: multilingual sentence embeddings (sentence-transformers,
   paraphrase-multilingual-MiniLM-L12-v2) which understand Hindi directly.
   If the model cannot be downloaded the TF-IDF result is used alone.
5. Cosine similarity between FIRs of the same crime type; pairs above a
   threshold are linked, and connected groups form "possible series".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.contract import RANDOM_STATE

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Devanagari -> English (domain glossary; extend as new FIR vocabulary appears)
HINDI_GLOSSARY = {
    "चेन": "chain", "छीन": "snatch", "छीनकर": "snatch", "छीनी": "snatch", "छीन ली": "snatch",
    "मोटरसाइकिल": "motorcycle", "बाइक": "motorcycle", "स्कूटी": "scooter", "काली": "black",
    "लाल": "red", "नीली": "blue", "सफेद": "white", "स्लेटी": "grey",
    "बिना नंबर": "without number plate", "नंबर प्लेट": "number plate", "युवक": "youth",
    "महिला": "woman", "गले": "neck", "सोने": "gold", "चोरी": "theft stole", "मोबाइल": "mobile",
    "फोन": "phone", "पर्स": "purse", "नकदी": "cash", "जेवर": "jewellery", "ताला": "lock",
    "तोड़कर": "broken", "मकान": "house", "अलमारी": "almirah", "रात": "night", "बाजार": "market",
    "दुकानदार": "shopkeeper", "धमकी": "threat threatened", "हफ्ता": "protection money",
    "वसूली": "extortion", "मांग": "demand", "पैसे": "money", "पैसों": "money", "खाते": "account",
    "ओटीपी": "otp", "बैंक": "bank", "अधिकारी": "officer", "देसी कट्टा": "country-made pistol",
    "कट्टा": "pistol", "कारतूस": "cartridges", "बरामद": "recovered", "मारपीट": "assault beaten",
    "विवाद": "quarrel dispute", "पीड़िता": "victim", "पीछा": "followed", "छेड़छाड़": "harassed",
    "शाम": "evening", "पास": "near", "लैपटॉप": "laptop",
}
# Hinglish (romanised Hindi) -> English
HINGLISH_GLOSSARY = {
    "chori": "theft", "chheen": "snatch", "kheench": "snatch", "ladka": "youth", "ladke": "youths",
    "aurat": "woman", "mahila": "woman", "gaadi": "vehicle", "kaali": "black", "bina": "without",
    "dukaandar": "shopkeeper", "hafta": "protection money", "vasooli": "extortion",
    "vasuli": "extortion", "dhamki": "threat", "dhamkaya": "threatened", "paise": "money",
    "paisa": "money", "khaate": "account", "nikaal": "withdrawn", "loot": "robbed",
    "chaaku": "knife", "pakda": "caught", "maal": "contraband", "pudiya": "packet",
    "katta": "pistol", "desi": "country-made", "nakad": "cash", "benami": "benami",
    "ghuma": "routed", "wapas": "returned", "shaam": "evening", "raste": "road",
    "zariye": "through", "bike": "motorcycle", "aaye": "came", "aaya": "came", "do": "two",
    "paas": "near", "chhen": "snatch", "ke": "", "ki": "", "ka": "", "se": "", "mein": "", "par": "",
    "pe": "", "ko": "", "aur": "and", "kar": "", "gaya": "", "gaye": "", "diya": "",
}
STOPWORDS_HI = {"ने", "के", "की", "का", "में", "से", "को", "और", "पर", "कर", "ली", "गई", "गए", "हुए", "किए", "आए", "दी"}


def detect_script(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "empty"
    deva = len(DEVANAGARI.findall(text))
    letters = sum(ch.isalpha() for ch in text)
    if deva / max(letters, 1) > 0.5:
        return "Hindi (Devanagari)"
    low = text.lower()
    if any(re.search(rf"\b{w}\b", low) for w in ["ke", "ki", "mein", "se", "par", "gaya", "kar"]):
        return "Hinglish (romanised)"
    return "English"


def normalize_text(text: str) -> str:
    """Lower-case, translate Hindi/Hinglish crime words, drop punctuation."""
    if not isinstance(text, str):
        return ""
    out = text
    for hi, en in sorted(HINDI_GLOSSARY.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(hi, f" {en} ")
    out = out.lower()
    out = re.sub(r"[^\w\s\-]", " ", out)
    tokens = []
    for tok in out.split():
        if tok in STOPWORDS_HI:
            continue
        if DEVANAGARI.search(tok):
            continue  # untranslated Hindi word: dropped from TF-IDF (embeddings still see it)
        tokens.append(HINGLISH_GLOSSARY.get(tok, tok))
    return " ".join(t for t in tokens if t)


class MOMatcher:
    """Fit on historical FIR texts; score new FIRs against them."""

    def __init__(self, use_embeddings: bool = False,
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self._encoder = None
        self.embedding_status = "not requested"

    def _encode(self, raw_texts: list[str]) -> np.ndarray | None:
        if not self.use_embeddings:
            return None
        try:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model, device="cpu")
            self.embedding_status = f"using {self.embedding_model}"
            return self._encoder.encode(raw_texts, batch_size=64, show_progress_bar=False,
                                        normalize_embeddings=True)
        except Exception as exc:  # offline / no package -> TF-IDF only
            self.embedding_status = f"unavailable ({type(exc).__name__}); TF-IDF only"
            self.use_embeddings = False
            return None

    def fit(self, texts: pd.Series) -> "MOMatcher":
        norm = texts.fillna("").map(normalize_text)
        self.word.fit(norm)
        self.char.fit(norm)
        return self

    def transform(self, texts: pd.Series):
        norm = texts.fillna("").map(normalize_text)
        return hstack([self.word.transform(norm), self.char.transform(norm)]).tocsr()

    def similarity(self, texts_a: pd.Series, texts_b: pd.Series | None = None) -> np.ndarray:
        A = self.transform(texts_a)
        B = A if texts_b is None else self.transform(texts_b)
        sim = cosine_similarity(A, B)
        ea = self._encode(texts_a.fillna("").tolist())
        if ea is not None:
            eb = ea if texts_b is None else self._encode(texts_b.fillna("").tolist())
            sim = 0.5 * sim + 0.5 * (ea @ eb.T)
        return sim


def find_similar_pairs(df: pd.DataFrame, matcher: MOMatcher, threshold: float = 0.7,
                       max_days: int = 45, text_col: str = "modus_operandi") -> pd.DataFrame:
    """FIR pairs of the same crime type with MO similarity >= threshold that
    occurred within `max_days` of each other.

    The time window matters: offenders work in sprees, while a common MO
    ("mobile stolen in market") recurs all year. Requiring both similar text
    and closeness in time keeps generic descriptions from being chained into
    one huge false "series".
    """
    rows = []
    for crime, g in df.groupby("crime_type"):
        if len(g) < 2:
            continue
        sim = matcher.similarity(g[text_col])
        dates = g["date_occurred"].to_numpy()
        iu = np.triu_indices(len(g), k=1)
        gap = np.abs((dates[iu[0]] - dates[iu[1]]).astype("timedelta64[D]").astype(int))
        keep = (sim[iu] >= threshold) & (gap <= max_days)
        ga = g.reset_index(drop=True)
        for i, j, s_, d in zip(iu[0][keep], iu[1][keep], sim[iu][keep], gap[keep]):
            rows.append({"fir_a": ga.at[i, "fir_no"], "fir_b": ga.at[j, "fir_no"], "crime_type": crime,
                         "similarity": round(float(s_), 3),
                         "station_a": ga.at[i, "police_station"], "station_b": ga.at[j, "police_station"],
                         "days_apart": int(d)})
    cols = ["fir_a", "fir_b", "crime_type", "similarity", "station_a", "station_b", "days_apart"]
    return pd.DataFrame(rows, columns=cols).sort_values("similarity", ascending=False).reset_index(drop=True)


def group_series(pairs: pd.DataFrame, df: pd.DataFrame, min_size: int = 4,
                 min_density: float = 0.65) -> pd.DataFrame:
    """Groups of linked FIRs = possible series by one offender or gang.

    A genuine series is a tight group: almost every FIR in it matches every
    other FIR (graph density close to 1). Loose chains, where A matches B and
    B matches C but A and C differ, are usually different offenders using a
    common method, so groups below `min_density` are not reported.
    """
    import networkx as nx
    g = nx.Graph()
    for r in pairs.itertuples():
        g.add_edge(r.fir_a, r.fir_b, similarity=r.similarity)
    info = df.set_index("fir_no")
    rows = []
    for comp in nx.connected_components(g):
        sub_g = g.subgraph(comp)
        density = nx.density(sub_g)
        if len(comp) < min_size or density < min_density:
            continue
        sub = info.loc[sorted(comp)]
        rows.append({
            "series_size": len(comp),
            "density": round(density, 2),
            "mean_similarity": round(float(np.mean([d["similarity"] for *_, d in sub_g.edges(data=True)])), 3),
            "crime_type": sub["crime_type"].mode().iloc[0],
            "stations": ", ".join(sorted(sub["police_station"].unique())),
            "n_stations": sub["police_station"].nunique(),
            "first_date": sub["date_occurred"].min().date(), "last_date": sub["date_occurred"].max().date(),
            "pending_in_series": int((sub["case_status"] == "Under Investigation").sum()),
            "example_mo": sub["modus_operandi"].iloc[0][:160],
            "fir_numbers": ", ".join(sub.sort_values("date_occurred").index),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["cohesion"] = (out["series_size"] * out["density"]).round(2)
    return out.sort_values(["cohesion", "n_stations"], ascending=False).reset_index(drop=True)


def series_pairs(series: pd.DataFrame) -> pd.DataFrame:
    """All FIR pairs inside reported series (for evaluation)."""
    rows = []
    for firs in series.get("fir_numbers", pd.Series(dtype=str)):
        f = firs.split(", ")
        rows += [(a, b) for i, a in enumerate(f) for b in f[i + 1:]]
    return pd.DataFrame(rows, columns=["fir_a", "fir_b"])


def evaluate_against_records(pairs: pd.DataFrame, df: pd.DataFrame, id_col: str = "accused_ids") -> dict:
    """Proxy check using documented records: share of matched pairs whose FIRs
    name at least one common accused, compared with random same-type pairs."""
    acc = df.set_index("fir_no")[id_col].map(
        lambda s: set(x.strip() for x in str(s).split(";")) - {"Unknown", ""})

    def share(p):
        if len(p) == 0:
            return 0.0
        return float(np.mean([len(acc[a] & acc[b]) > 0 for a, b in zip(p["fir_a"], p["fir_b"])]))

    rng = np.random.default_rng(RANDOM_STATE)
    rand_rows = []
    for crime, g in df.groupby("crime_type"):
        firs = g["fir_no"].to_numpy()
        if len(firs) < 2:
            continue
        for _ in range(200):
            a, b = rng.choice(firs, 2, replace=False)
            rand_rows.append((a, b))
    rand = pd.DataFrame(rand_rows, columns=["fir_a", "fir_b"])
    return {"matched_pairs": int(len(pairs)),
            "matched_share_common_accused": round(share(pairs), 3),
            "random_share_common_accused": round(share(rand), 3)}


def cluster_mo(texts: pd.Series, matcher: MOMatcher, k_range=range(4, 21, 2), k: int | None = None):
    """LSA (TruncatedSVD = PCA for sparse text) + K-Means with elbow curve."""
    X = matcher.transform(texts)
    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    Z = svd.fit_transform(X)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-9)
    curve = pd.DataFrame([{"k": kk, "inertia": KMeans(n_clusters=kk, n_init=10, random_state=RANDOM_STATE).fit(Z).inertia_}
                          for kk in k_range])
    if k is None:
        from src.hotspots import knee_point
        k = knee_point(curve)
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
    return labels, curve, float(svd.explained_variance_ratio_.sum())
