"""Generate the synthetic FIR case register (data/fir_cases.csv).

WHY THIS SCRIPT EXISTS
----------------------
The five provided files (data/synthetic/) describe one criminal network in
detail (FIR/intelligence text, call records, money trail, social-media tips,
criminal history) but contain only a handful of rows and no case-register
fields (crime type, police station, chargesheet date, court outcome). The
CSL236 end-term guidelines require >= 1,000 rows and >= 5 features and allow
"a synthetic one you generate with a documented script". This script is that
documented generator.

HOW IT LINKS TO THE 5 FILES
---------------------------
* Accused persons are drawn from criminal_history.csv (plus Anita Deshmukh,
  who appears in the CDR, money-trail and intelligence files) and from a pool
  of clearly synthetic offenders.
* Police stations are the places named in the files (Bhawarkuan, Rajwada,
  Chandan Nagar, Rau, Palasia, Vijay Nagar, Sector 12, Dewas Naka, Karol Bagh).
* Gang cases reuse the modus operandi described in fir_reports.json
  (extortion at Bhawarkuan, highway-dhaba hand-overs at Dewas Naka, funds
  routed through a shell trading firm, illegal arms).

Everything is seeded, so running it twice produces the identical file.
All names, phone numbers and addresses are fictitious.

Usage:  python data/generate_fir_cases.py [--n 2400] [--seed 42]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contract import (  # noqa: E402
    AS_OF_DATE, CRIME_TYPES, OFFENCE_GRAVITY, PROPERTY_CRIMES, STATIONS,
)

SYN_DIR = ROOT / "data" / "synthetic"
OUT_FILE = ROOT / "data" / "fir_cases.csv"

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-08-31")

# --------------------------------------------------------------------------
# Distributions (documented assumptions)
# --------------------------------------------------------------------------
CRIME_BASE = {  # share of all FIRs at the start of the period
    "Theft": 0.20, "Vehicle Theft": 0.12, "Chain Snatching": 0.10,
    "Burglary": 0.09, "Robbery": 0.05, "Assault": 0.12, "Extortion": 0.05,
    "Cyber Fraud": 0.10, "Narcotics": 0.06, "Money Laundering": 0.03,
    "Illegal Arms": 0.04, "Sexual Offence": 0.04,
}
# Yearly growth factor of each crime type (1.0 = flat). Creates rising and
# falling trends for the trend-analysis section to find.
CRIME_GROWTH = {"Cyber Fraud": 1.8, "Narcotics": 1.35, "Chain Snatching": 0.6,
                "Vehicle Theft": 0.90, "Extortion": 1.15}
# Month multipliers (festival season theft, winter-evening snatching, summer assault)
CRIME_SEASON = {
    "Theft":           {10: 1.9, 11: 2.0, 12: 1.3},
    "Burglary":        {10: 1.5, 11: 1.6, 5: 1.3},
    "Chain Snatching": {11: 1.5, 12: 1.7, 1: 1.6, 2: 1.3},
    "Assault":         {4: 1.3, 5: 1.5, 6: 1.3},
    "Vehicle Theft":   {7: 1.2, 8: 1.3},
}
STATION_WEIGHT = {"Bhawarkuan": 0.17, "Rajwada": 0.16, "Chandan Nagar": 0.10,
                  "Rau": 0.08, "Palasia": 0.11, "Vijay Nagar": 0.12,
                  "Sector 12": 0.09, "Dewas Naka": 0.11, "Karol Bagh": 0.06}
# Station x crime affinity multipliers (local crime character)
AFFINITY = {
    "Rajwada": {"Theft": 2.0, "Chain Snatching": 1.8, "Money Laundering": 1.5},
    "Bhawarkuan": {"Extortion": 2.5, "Assault": 1.5},
    "Dewas Naka": {"Narcotics": 3.0, "Vehicle Theft": 1.8, "Robbery": 1.5},
    "Vijay Nagar": {"Cyber Fraud": 2.5, "Chain Snatching": 1.4},
    "Rau": {"Money Laundering": 2.5, "Burglary": 1.4},
    "Palasia": {"Illegal Arms": 2.0, "Chain Snatching": 1.3},
    "Sector 12": {"Extortion": 1.6, "Burglary": 1.4},
    "Karol Bagh": {"Narcotics": 2.5, "Cyber Fraud": 1.3},
    "Chandan Nagar": {"Assault": 1.4, "Illegal Arms": 1.3},
}
# Hour-of-day: list of (mean hour, sd, weight) mixtures per crime type
HOUR_MIX = {
    "Theft": [(14, 3, 0.6), (20, 2, 0.4)],
    "Vehicle Theft": [(2, 2, 0.6), (21, 2, 0.4)],
    "Chain Snatching": [(7.5, 1.2, 0.45), (19, 1.5, 0.55)],
    "Burglary": [(2, 1.5, 0.8), (14, 2, 0.2)],
    "Robbery": [(22, 2, 0.7), (13, 3, 0.3)],
    "Assault": [(21, 2.5, 0.7), (15, 3, 0.3)],
    "Extortion": [(11, 2, 0.6), (18, 2, 0.4)],
    "Cyber Fraud": [(13, 3, 1.0)],
    "Narcotics": [(23, 2, 0.8), (16, 2, 0.2)],
    "Money Laundering": [(12, 2, 1.0)],
    "Illegal Arms": [(22, 2.5, 0.7), (12, 3, 0.3)],
    "Sexual Offence": [(21, 3, 0.7), (14, 3, 0.3)],
}
IPC_OF_CRIME = {
    "Theft": "IPC 379", "Vehicle Theft": "IPC 379", "Chain Snatching": "IPC 356; IPC 379",
    "Burglary": "IPC 457; IPC 380", "Robbery": "IPC 392", "Assault": "IPC 323; IPC 506",
    "Extortion": "IPC 384; IPC 506; IPC 34", "Cyber Fraud": "IPC 420; IT Act 66D",
    "Narcotics": "NDPS 8/20", "Money Laundering": "PMLA 3; IPC 120B",
    "Illegal Arms": "Arms Act 25", "Sexual Offence": "IPC 354",
}
# Effects on the log-odds of the chargesheet missing the 90-day deadline
DELAY_CRIME_EFFECT = {
    "Theft": -0.5, "Vehicle Theft": -0.3, "Chain Snatching": -0.3, "Burglary": 0.1,
    "Robbery": 0.2, "Assault": -0.7, "Extortion": 0.4, "Cyber Fraud": 1.0,
    "Narcotics": 0.9, "Money Laundering": 1.5, "Illegal Arms": 0.5,
    "Sexual Offence": 0.4,
}
CONVICTION_RATE = {
    "Theft": 0.45, "Vehicle Theft": 0.40, "Chain Snatching": 0.50, "Burglary": 0.38,
    "Robbery": 0.55, "Assault": 0.35, "Extortion": 0.48, "Cyber Fraud": 0.22,
    "Narcotics": 0.62, "Money Laundering": 0.18, "Illegal Arms": 0.70,
    "Sexual Offence": 0.30,
}

FIRST = ["Rahul", "Amit", "Pooja", "Neha", "Rohit", "Sunil", "Kavita", "Manoj", "Priya", "Arjun",
         "Nikhil", "Sneha", "Vivek", "Rakesh", "Meena", "Ajay", "Imran", "Salim", "Gurpreet", "Joseph",
         "Lakshmi", "Harish", "Tarun", "Kiran", "Dinesh", "Mohit", "Asha", "Naveen", "Ritu", "Sachin",
         "Pankaj", "Alok", "Bhavna", "Chetan", "Devendra", "Faisal", "Gaurav", "Hemant", "Irfan", "Jitendra",
         "Kamal", "Lokesh", "Mukesh", "Nitin", "Om", "Prakash", "Rajesh", "Sameer", "Uday", "Yogesh",
         "Anil", "Bharat", "Deepika", "Girish", "Jagdish", "Kunal", "Mahesh", "Narendra", "Raju", "Shivam"]
LAST = ["Sharma", "Verma", "Patel", "Joshi", "Gupta", "Malviya", "Chouhan", "Tiwari", "Saxena", "Jain",
        "Pandey", "Mishra", "Rawat", "Solanki", "Bhatia", "Kushwaha", "Dubey", "Thakur", "Rao", "Agarwal",
        "Yadav", "Singh", "Parmar", "Rathod", "Sen", "Kale", "Pawar", "Bairagi", "Sisodia", "Goyal",
        "Choudhary", "Nagar", "Bhargava", "Sahu", "Prajapati", "Mandloi", "Dhakad", "Gehlot", "Kumawat", "Vyas"]

# Modus-operandi templates: English, Hindi (Devanagari) and Hinglish.
# {slots} are filled at random so ordinary cases differ from each other,
# while gang cases (below) repeat a distinctive signature.
SLOTS = {
    "item": ["mobile phone", "wallet", "purse", "laptop bag", "gold ring", "cash", "tablet", "silver anklet"],
    "item_hi": ["मोबाइल फोन", "पर्स", "लैपटॉप", "नकदी", "अंगूठी", "बैग"],
    "when": ["in the afternoon rush", "during the evening", "on a holiday", "at closing time", "in the morning"],
    "when_hi": ["सुबह", "दोपहर", "शाम", "रात"],
    "colour": ["red", "blue", "white", "grey", "silver", "green"],
    "colour_hi": ["लाल", "नीली", "सफेद", "स्लेटी", "हरी"],
    "ride": ["scooter", "moped", "motorcycle", "bicycle"],
    "landmark": ["park", "bus stop", "vegetable market", "school gate", "hospital road", "petrol pump", "railway crossing"],
    "landmark_hi": ["मंदिर", "पार्क", "बस स्टॉप", "सब्जी मंडी", "स्कूल", "अस्पताल"],
    "vehicle": ["two-wheeler", "auto-rickshaw", "car", "e-rickshaw", "pickup van"],
    "lock": ["handle lock", "ignition wiring", "side window", "steering lock", "duplicate key"],
    "weapon": ["knife", "iron rod", "toy pistol", "blade", "chopper"],
    "cause": ["parking", "a land dispute", "loud music", "a road accident", "money owed", "a cricket match"],
    "hit": ["sticks", "fists", "a brick", "an iron rod", "a belt"],
    "bank_ruse": ["bank officer", "electricity board official", "courier company agent", "army officer buying goods", "loan app agent"],
    "channel": ["phone call", "WhatsApp message", "SMS link", "fake website"],
    "drug": ["brown sugar", "ganja", "MD powder", "opium", "codeine syrup"],
    "qty": ["12 grams", "50 grams", "1.2 kg", "200 grams", "35 bottles"],
    "amount": ["Rs 18,000", "Rs 45,000", "Rs 1.2 lakh", "Rs 3 lakh", "Rs 75,000"],
    "n": ["two", "three", "four"],
}
MO_TEMPLATES = {
    "Theft": [
        "Unknown person stole {item} from complainant's bag in crowded {place} market {when}.",
        "{place} {landmark_hi} के पास {when_hi} में अज्ञात व्यक्ति ने शिकायतकर्ता का {item_hi} चोरी कर लिया।",
        "{place} mein shop ke counter se {item} chori, {when} ek ladka dikh raha tha.",
    ],
    "Vehicle Theft": [
        "{colour} {vehicle} parked outside {place} {landmark} was stolen {when} by breaking the {lock}.",
        "{place} में {landmark_hi} के बाहर खड़ी {colour_hi} मोटरसाइकिल का लॉक तोड़कर {when_hi} में चोरी कर ली गई।",
    ],
    "Chain Snatching": [
        "A youth on a {colour} {ride} snatched gold chain from a woman near {place} {landmark} {when}.",
        "{colour_hi} स्कूटी पर आए युवक ने {place} {landmark_hi} के पास {when_hi} में महिला की चेन छीन ली।",
        "{colour} {ride} pe ek ladka aaya, {place} {landmark} ke paas aurat ki chain kheench ke bhaag gaya.",
    ],
    "Burglary": [
        "House lock broken while family was away; {item} and jewellery stolen from almirah near {place} {landmark}.",
        "{place} में {landmark_hi} के पास बंद मकान का ताला तोड़कर {when_hi} में {item_hi} और जेवर चोरी किए गए।",
    ],
    "Robbery": [
        "Complainant was stopped near {place} {landmark} by {n} men who showed a {weapon} and took {item}.",
        "{weapon} dikha kar {n} logon ne {place} {landmark} ke raste mein {item} loot liya.",
    ],
    "Assault": [
        "Quarrel over {cause} near {place} {landmark} turned violent; complainant beaten with {hit}.",
        "{place} {landmark_hi} के पास {when_hi} में विवाद के बाद आरोपियों ने शिकायतकर्ता के साथ मारपीट की।",
    ],
    "Extortion": [
        "Shopkeeper near {place} {landmark} threatened over {channel} and asked to pay {amount}.",
        "{place} में दुकानदार को {when_hi} फोन पर धमकी देकर पैसों की मांग की गई।",
        "{place} {landmark} ke dukaandar ko {channel} par dhamki, {amount} na dene par nuksaan ki baat kahi.",
    ],
    "Cyber Fraud": [
        "Complainant received a {channel} from a person posing as {bank_ruse}, shared OTP and lost {amount}.",
        "{bank_ruse} ban kar {channel} bheja, OTP lekar khaate se {amount} nikaal liye.",
        "{when_hi} में बैंक अधिकारी बनकर फोन किया और ओटीपी लेकर खाते से {amount} निकाल लिए।",
    ],
    "Narcotics": [
        "Accused found carrying {qty} {drug} near {place} {landmark} for sale to local users.",
        "{qty} {drug} ki pudiya bechte hue {place} {landmark} par pakda gaya.",
    ],
    "Money Laundering": [
        "Unexplained cash deposits of {amount} in multiple accounts at {place}; source of funds not disclosed.",
        "Benami khaaton mein {place} se {amount} nakad jama karaya gaya.",
    ],
    "Illegal Arms": [
        "{weapon} and live cartridges recovered from accused during vehicle checking near {place} {landmark}.",
        "{place} {landmark_hi} के पास चेकिंग में आरोपी के पास से देसी कट्टा और जिंदा कारतूस बरामद हुए।",
    ],
    "Sexual Offence": [
        "Victim was harassed and followed by the accused near {place} {landmark}; complaint filed by family.",
        "{place} {landmark_hi} के पास पीड़िता का पीछा कर छेड़छाड़ की गई, परिवार ने शिकायत दर्ज कराई।",
    ],
}


# Case-specific detail sentences. Every FIR gets 1-2 of these, so ordinary
# FIRs that use the same template still read differently (as real FIRs do).
DETAILS = {
    "en": ["Complainant noticed it after about {minutes} minutes.", "The accused fled towards {direction}.",
           "No CCTV camera was found in the lane.", "Nearby shopkeepers were questioned by the beat constable.",
           "Complainant lives in ward {ward} and was returning from work.", "A neighbour raised an alarm but no one was caught.",
           "The area had a power cut at the time.", "Complainant could describe only the clothes of the accused.",
           "Loss estimated at Rs {money}.", "Information was given on Dial 112 at {clock}.",
           "A rickshaw driver saw the incident from a distance.", "Footage from a private camera has been requested."],
    "hi": ["घटना की सूचना डायल 112 पर दी गई।", "आरोपी {direction_hi} की ओर भाग गया।", "गली में कोई सीसीटीवी कैमरा नहीं है।",
           "पड़ोसियों से पूछताछ की गई।", "शिकायतकर्ता वार्ड {ward} में रहता है।", "नुकसान लगभग {money} रुपये का है।",
           "उस समय बिजली गुल थी।", "एक दुकानदार ने घटना देखी।"],
    "hinglish": ["Complainant ne {minutes} minute baad dekha.", "Aaropi {direction} ki taraf bhaag gaya.",
                 "Gali mein koi CCTV nahi hai.", "Padosiyon se poochtaach ki gayi.", "Nuksaan lagbhag {money} rupaye.",
                 "Dial 112 par {clock} baje soochna di gayi.", "Ek auto wale ne ghatna dekhi."],
}
DETAIL_SLOTS = {
    "minutes": [str(m) for m in range(5, 95, 5)], "direction": ["the highway", "the railway station", "the old city", "the bypass", "the market", "the river bank"],
    "direction_hi": ["हाईवे", "रेलवे स्टेशन", "पुराने शहर", "बायपास", "बाजार"], "ward": [str(w) for w in range(1, 86)],
    "money": [f"{v:,}" for v in range(2000, 200000, 3500)], "clock": [f"{h}:{m:02d}" for h in range(1, 13) for m in (0, 15, 30, 45)],
}
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_HINGLISH = re.compile(r"\b(ke|ki|mein|se|par|kar|ko|liya|gaya|diya)\b")


def _lang(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "hi"
    return "hinglish" if _HINGLISH.search(text.lower()) else "en"


def fill_slots(rng, template: str, place: str, n_details: int = 1) -> str:
    out = template.replace("{place}", place)
    for key, options in SLOTS.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, str(rng.choice(options)))
    out = out[0].upper() + out[1:]
    pool = DETAILS[_lang(out)]
    for d in rng.choice(pool, size=n_details, replace=False):
        for key, options in DETAIL_SLOTS.items():
            d = d.replace("{" + key + "}", str(rng.choice(options)))
        out += " " + d
    return out


# Gang modus operandi (same offender groups repeated across FIRs/stations)
GANGS = {
    "RATHORE_EXTORTION": {
        "members": ["Vikram Rathore", "Ramesh Kumar", "Zubair Khan"],
        "crime": "Extortion", "stations": ["Bhawarkuan", "Sector 12", "Rajwada"],
        "mo": ["Shopkeeper near {place} threatened by men in vehicle MP09AB1234; they demanded weekly protection money and warned of consequences.",
               "Dukaandar ko {place} par MP09AB1234 gaadi mein aaye logon ne hafta vasooli ke liye dhamkaya.",
               "Protection money demanded from shop owner near {place} by two men arriving in vehicle MP09AB1234; repeated threatening calls received."],
        "n": 34,
    },
    "HIGHWAY_NARCOTICS": {
        "members": ["Farhan Ali", "Ramesh Kumar", "Om Prakash"],
        "crime": "Narcotics", "stations": ["Dewas Naka", "Karol Bagh"],
        "mo": ["Package handed over at highway dhaba near {place}; contraband arrived in Delhi registered car DL1CAB1234.",
               "Highway dhaba par Delhi number ki gaadi DL1CAB1234 se maal ka packet diya gaya.",
               "Contraband package exchanged at a dhaba on the highway near {place}; Delhi-registered car DL1CAB1234 involved."],
        "n": 26,
    },
    "SHELL_FIRM_LAUNDERING": {
        "members": ["Suresh Yadav", "Anita Deshmukh", "Vikram Rathore"],
        "crime": "Money Laundering", "stations": ["Rau", "Rajwada"],
        "mo": ["Proceeds routed through Chatterjee Traders shell firm via RTGS and returned to the original account in a loop.",
               "Chatterjee Traders ke khaate se RTGS dwara paisa ghuma kar wapas mool khaate mein bheja gaya.",
               "Funds layered through shell trading firm Chatterjee Traders and forwarded onward through UPI transfers."],
        "n": 16,
    },
    "ARMS_SUPPLY": {
        "members": ["Sanjay Mehta", "Deepak Verma"],
        "crime": "Illegal Arms", "stations": ["Palasia", "Sector 12", "Vijay Nagar"],
        "mo": ["Country-made pistols supplied from {place}; buyer contacted through a taxi driver acting as courier.",
               "Taxi driver ke zariye desi katta supply kiya gaya, deal {place} mein hui.",
               "Illegal pistols delivered at {place} using a taxi as courier vehicle."],
        "n": 18,
    },
    "BLACK_BIKE_SNATCHERS": {
        "members": ["Tarun Solanki", "Mohit Rawat"],
        "crime": "Chain Snatching", "stations": ["Rajwada", "Palasia", "Vijay Nagar", "Chandan Nagar"],
        "mo": ["Two youths on a black motorcycle without number plate snatched gold chain from a woman walking near {place} in the evening.",
               "Kaali bike par bina number plate ke do ladke aaye aur shaam ko {place} ke paas mahila ki chain chheen li.",
               "बिना नंबर प्लेट की काली बाइक पर आए दो युवकों ने शाम को {place} के पास महिला की चेन छीनी।"],
        "n": 30,
    },
}


def _known_people() -> list[str]:
    hist = pd.read_csv(SYN_DIR / "criminal_history.csv")
    return list(hist["name"]) + ["Anita Deshmukh"]


def _day_weights(crime: str, days: pd.DatetimeIndex) -> np.ndarray:
    years = np.asarray((days - START).days, dtype=float) / 365.25
    growth = CRIME_GROWTH.get(crime, 1.0) ** years
    season = np.array([CRIME_SEASON.get(crime, {}).get(m, 1.0) for m in days.month])
    weekend = np.where(np.asarray(days.day_of_week) >= 5, 1.15, 1.0)
    w = growth * season * weekend
    return w / w.sum()


def _sample_hour(rng, crime: str) -> int:
    mix = HOUR_MIX[crime]
    probs = np.array([m[2] for m in mix])
    mu, sd, _ = mix[rng.choice(len(mix), p=probs / probs.sum())]
    return int(round(rng.normal(mu, sd))) % 24


def _phone(rng) -> str:
    return str(rng.choice([6, 7, 8, 9])) + "".join(str(d) for d in rng.integers(0, 10, 9))


def generate(n: int = 2400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")
    day_w = {c: _day_weights(c, days) for c in CRIME_TYPES}
    stations = list(STATIONS)
    st_w = np.array([STATION_WEIGHT[s] for s in stations])

    # Offender pool: the documented people + synthetic repeat offenders
    known = _known_people()
    # Synthetic offenders, each with a home station. Most appear once; a
    # small group of habitual offenders appears repeatedly (repeat offenders).
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(combos)
    offenders = combos[:2000]
    home = {o: stations[i % len(stations)] for i, o in enumerate(offenders)}
    habitual = set(offenders[:30])
    by_home = {s: [o for o in offenders if home[o] == s] for s in stations}

    def draw_offenders(station: str, k: int) -> list[str]:
        out = []
        for _ in range(k):
            src_st = station if rng.random() < 0.85 else rng.choice(stations)
            cand = by_home[src_st]
            w = np.array([6.0 if o in habitual else 1.0 for o in cand])
            out.append(str(rng.choice(cand, p=w / w.sum())))
        return list(dict.fromkeys(out))

    # Investigating officers: 4 per station with different case loads
    io_load = {}
    for s in stations:
        for k in range(1, 5):
            io_load[f"{STATIONS[s]['code']}-IO{k}"] = int(rng.integers(8, 52))

    # ---- pick crime type + station for each non-gang case ----
    gang_rows = sum(g["n"] for g in GANGS.values())
    rows = []
    plan = []
    # Gangs work in sprees: 3 windows of ~40 days each in which their cases occur
    spree = {}
    for gname, g in GANGS.items():
        starts = sorted(rng.choice(len(days) - 45, size=3, replace=False))
        spree[gname] = [days[st:st + 40] for st in starts]
        for _ in range(g["n"]):
            plan.append(("gang", gname))
    for _ in range(n - gang_rows):
        plan.append(("solo", None))

    serial = {s: {} for s in stations}
    for kind, gname in plan:
        if kind == "gang":
            g = GANGS[gname]
            crime = g["crime"]
            station = rng.choice(g["stations"])
            k = int(rng.integers(2, len(g["members"]) + 1))
            accused = list(rng.choice(g["members"], size=k, replace=False))
            if rng.random() < 0.3:
                accused += draw_offenders(station, 1)
            mo = rng.choice(g["mo"])
        else:
            # station first, then crime type with station affinity
            station = rng.choice(stations, p=st_w)
            cw = np.array([CRIME_BASE[c] * AFFINITY.get(station, {}).get(c, 1.0) for c in CRIME_TYPES])
            crime = rng.choice(CRIME_TYPES, p=cw / cw.sum())
            if crime in {"Theft", "Vehicle Theft", "Burglary"} and rng.random() < 0.35:
                accused = []  # accused not yet identified
            else:
                k = 1 + rng.poisson(1.2 if crime in {"Narcotics", "Money Laundering", "Robbery", "Assault"} else 0.4)
                accused = draw_offenders(station, k)
                if rng.random() < 0.04:  # occasional documented person in an unrelated case
                    accused.append(str(rng.choice(known)))
            mo = rng.choice(MO_TEMPLATES[crime])

        if kind == "gang":
            window = spree[gname][int(rng.integers(0, 3))]
            date_occ = window[int(rng.integers(0, len(window)))]
        else:
            date_occ = days[rng.choice(len(days), p=day_w[crime])]
        reg_delay = int(min(rng.exponential(1.5 if crime not in {"Sexual Offence", "Cyber Fraud", "Money Laundering"} else 6), 120))
        date_reg = date_occ + pd.Timedelta(days=reg_delay)
        if date_reg > END:
            date_reg, reg_delay = END, int((END - date_occ).days)
        hour = _sample_hour(rng, crime)

        meta = STATIONS[station]
        # each station has 2 micro-hotspots + background scatter
        hs = (stations.index(station) % 7) / 1000.0  # deterministic offset
        centres = [(meta["lat"] + 0.006 + hs, meta["lon"] - 0.004), (meta["lat"] - 0.005, meta["lon"] + 0.006 - hs)]
        if rng.random() < 0.65:
            c = centres[rng.integers(0, 2)]
            lat, lon = rng.normal(c[0], 0.0025), rng.normal(c[1], 0.0025)
        else:
            lat, lon = rng.normal(meta["lat"], 0.012), rng.normal(meta["lon"], 0.012)

        is_prop = crime in PROPERTY_CRIMES
        if is_prop:
            scale = {"Money Laundering": 13.0, "Extortion": 11.0, "Cyber Fraud": 10.8,
                     "Robbery": 10.5, "Burglary": 11.0, "Vehicle Theft": 10.8}.get(crime, 9.8)
            prop_val = float(np.round(rng.lognormal(scale, 0.9), -2))
        else:
            prop_val = 0.0

        io_id = f"{meta['code']}-IO{int(rng.integers(1, 5))}"
        caseload = io_load[io_id] + int(rng.integers(-3, 4))
        n_acc = max(len(accused), 1)
        has_cctv = rng.random() < (0.55 if station in {"Rajwada", "Vijay Nagar", "Palasia"} else 0.3)
        has_cdr = rng.random() < (0.75 if crime in {"Extortion", "Narcotics", "Money Laundering", "Cyber Fraud", "Illegal Arms"} else 0.2)
        has_fin = rng.random() < (0.9 if crime in {"Money Laundering", "Cyber Fraud"} else 0.45 if crime == "Extortion" else 0.05)
        forensic = rng.random() < (0.8 if crime in {"Burglary", "Sexual Offence", "Illegal Arms", "Narcotics"} else 0.35 if crime in {"Assault", "Robbery"} else 0.08)
        witnesses = int(rng.poisson(2.5 if crime not in {"Cyber Fraud", "Money Laundering"} else 0.8))

        # ---- outcome: days to chargesheet ----
        logit = (-1.5 + DELAY_CRIME_EFFECT[crime] + 0.32 * (n_acc - 1)
                 + 0.055 * (caseload - 25) + 0.9 * forensic + 0.6 * has_fin
                 - 0.7 * has_cctv + 0.015 * reg_delay - 0.1 * min(witnesses, 5)
                 + (0.6 if station == "Karol Bagh" else 0.35 if station == "Rau" else 0.0)
                 + rng.normal(0, 0.6))
        delayed = rng.random() < 1 / (1 + np.exp(-logit))
        days_to_cs = int(91 + rng.exponential(75)) if delayed else int(rng.integers(15, 91))

        as_of = pd.Timestamp(AS_OF_DATE)
        age = (as_of - date_reg).days
        cs_date, status, court = pd.NaT, "Under Investigation", ""
        if not accused and rng.random() < 0.5 and age > 60:
            status = "Closed - Final Report"
        elif days_to_cs <= age:
            status, cs_date = "Chargesheet Filed", date_reg + pd.Timedelta(days=days_to_cs)
            if (as_of - cs_date).days > 240:
                court = "Convicted" if rng.random() < CONVICTION_RATE[crime] else "Acquitted"
            else:
                court = "Pending Trial"

        code = meta["code"]
        yr = date_reg.year
        serial[station][yr] = serial[station].get(yr, 0) + 1
        place = station
        rows.append({
            "_sort": date_reg, "police_station": station, "district": meta["district"],
            "date_occurred": date_occ, "date_registered": date_reg,
            "hour_of_occurrence": hour, "crime_type": crime,
            "ipc_sections": IPC_OF_CRIME[crime], "offence_gravity": OFFENCE_GRAVITY[crime],
            "modus_operandi": fill_slots(rng, mo, place, n_details=1 if kind == "gang" else int(rng.integers(1, 3))),
            "latitude": round(lat, 5), "longitude": round(lon, 5),
            "accused_names": "; ".join(accused) if accused else "Unknown",
            "num_accused": n_acc if accused else 0, "num_witnesses": witnesses,
            "complainant_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "complainant_phone": _phone(rng),
            "complainant_address": f"{int(rng.integers(1, 400))}, Ward {int(rng.integers(1, 60))}, {station}, {meta['district']}",
            "victim_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}" if crime in {"Sexual Offence", "Assault", "Robbery"} else "",
            "property_value_inr": prop_val,
            "io_id": io_id, "io_caseload": caseload,
            "registration_delay_days": reg_delay,
            "has_cctv": "Yes" if has_cctv else "No",
            "has_cdr_evidence": "Yes" if has_cdr else "No",
            "has_financial_trail": "Yes" if has_fin else "No",
            "forensic_required": "Yes" if forensic else "No",
            "case_status": status, "chargesheet_date": cs_date, "court_status": court,
            "_code": code,
        })

    df = pd.DataFrame(rows).sort_values("_sort").reset_index(drop=True)
    # FIR numbers in registration order per station and year, e.g. RJW/0042/2025
    df["_yr"] = df["date_registered"].dt.year
    df["_n"] = df.groupby(["_code", "_yr"]).cumcount() + 1
    df.insert(0, "fir_no", df["_code"] + "/" + df["_n"].astype(str).str.zfill(4) + "/" + df["_yr"].astype(str))
    df = df.drop(columns=["_sort", "_code", "_yr", "_n"])

    # redundant column (perfectly correlated) for the correlation-filter step
    df["property_value_lakh"] = (df["property_value_inr"] / 1e5).round(3)

    # ---------------- realistic data-quality problems ----------------
    idx = df.index.to_numpy()
    def pick(frac):
        return rng.choice(idx, size=int(len(idx) * frac), replace=False)
    df.loc[pick(0.06), "num_witnesses"] = np.nan                     # not recorded
    prop_rows = df.index[df["crime_type"].isin(PROPERTY_CRIMES)].to_numpy()
    miss = rng.choice(prop_rows, size=int(len(prop_rows) * 0.05), replace=False)
    df.loc[miss, ["property_value_inr", "property_value_lakh"]] = np.nan  # value not assessed
    df.loc[pick(0.04), "hour_of_occurrence"] = np.nan                # time unknown
    df.loc[pick(0.03), "io_caseload"] = np.nan                       # IO roster missing
    df.loc[rng.choice(idx, 4, replace=False), "num_accused"] = 99    # data-entry error
    df.loc[rng.choice(idx, 5, replace=False), "registration_delay_days"] = -1  # entry error
    # inconsistent station spellings
    variants = {"Bhawarkuan": ["BHAWARKUAN P.S.", "Bhanwarkuan", " bhawarkuan "],
                "Rajwada": ["RAJWADA", "Rajwada PS"], "Vijay Nagar": ["Vijaynagar", "VIJAY NAGAR"],
                "Chandan Nagar": ["Chandannagar"], "Dewas Naka": ["Dewas naka"]}
    for st, vs in variants.items():
        rows_st = df.index[df["police_station"] == st].to_numpy()
        chg = rng.choice(rows_st, size=int(len(rows_st) * 0.08), replace=False)
        df.loc[chg, "police_station"] = [rng.choice(vs) for _ in chg]
    # mixed date formats in date_registered
    fmt_rows = pick(0.15)
    df["date_registered"] = df["date_registered"].dt.strftime("%Y-%m-%d")
    df.loc[fmt_rows, "date_registered"] = pd.to_datetime(df.loc[fmt_rows, "date_registered"]).dt.strftime("%d/%m/%Y")
    df["date_occurred"] = df["date_occurred"].dt.strftime("%Y-%m-%d")
    df["chargesheet_date"] = pd.to_datetime(df["chargesheet_date"]).dt.strftime("%Y-%m-%d")
    # exact duplicate entries (same FIR keyed twice)
    dups = df.loc[rng.choice(idx, 12, replace=False)]
    df = pd.concat([df, dups], ignore_index=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()
    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns to {args.out}")


if __name__ == "__main__":
    main()
