"""
local_classifier.py
-------------------
A small keyword-based fallback classifier for CampusSense AI.

It is used ONLY when the Hugging Face model cannot answer (no credits, rate limit, network
error, bad JSON). Instead of a useless "Neutral / Other / Low" placeholder, every row still
gets a sensible sentiment, category, severity, keywords and suggested action, so the
filters and dashboard always have real data to show.

Rows analyzed this way carry an `analysis_error` message, so the app can tell you about it.
"""

import re

_NEG = ["broken", "break down", "breaks down", "not working", "does not work", "doesn't work", "slow", "dirty",
        "filthy", "smell", "unsafe", "hazard", "damaged", "crash", "failed", "fail", "down for", "dead",
        "disconnect", "drops", "unreliable", "not enough", "too few", "too long", "long line", "sold out",
        "cold food", "spoiled", "sick", "injur", "stranded", "congested", "chaotic", "confusing", "unusable",
        "out of order", "leak", "dangerous", "uncomfortable", "unbearable", "noisy", "worn", "closed for weeks",
        "keeps malfunctioning", "needs repair", "not cleaned", "dusty", "sticky", "overflow", 
        " late ", "delays", "unattended", "poorly", "not well", "lost their", "erased", "logging users out", "freez",
        "errors", "cracked", "does not", "doesn't", "cannot", "can't", "keeps ", "loose", "wobbly", "crack", "blocked", "disorganized", "ruining", "miss ", "miss required", "stains", "last minute", "still needs", "same problem", "restart", "unfinished", "still a problem", "still dusty", "dirty quickly", "gets dirty", "gets uncomfortable", "slows down", "too small", "always full", "poor shape", "bad in others", "worn out", "very slow", "poor", "limited", "too ", "worse", "missing", "not save", "changing", "shorter hours" , "flood", "no replacement", "burned out", "insects", "pests", "risky", "nothing was done"]
_POS = ["great", "excellent", "fast and", "improved", "improve a lot", "spotless", "delicious", "friendly",
        "thank", "helpful", "quick and helpful", "comfortable", "well organized", "well-maintained",
        "well maintained", "well managed", "much better", "perfectly", "perfect", "renovated", "looks great",
        "love", "pleasant", "clean thanks", "always clean", "safer", "secure", "easy to use", "stable now",
        "saves us", "smoothly", "well cleaned", "much cleaner", "fixed quickly", "repaired quickly", "appreciate",
        "good news", "replaced with a new", "solved", "looks better", "better than before", "clean in the morning", "looks great", "fixed right after", "was fixed", "was repaired", "reliable", "faster", "helps", "helped", "hope it stays", "easy to", "works well", "good meals", "tasty", "cheap", "fast", "convenient", "upgrade", "stable"]
_NEUTRAL_CUES = ["scheduled", "will be", "was moved", "moved to", "notice", "announced", "announcement", "fyi",
                 "for your information", "open from", "checked by", "was checked", "inspection", "inspected",
                 "reviewed", "measured", "included in", "posted on", "will start", "will have", "changed its menu",
                 "was rearranged", "was adjusted", "logged for review", "was updated", "was reported to"]
_CONTRAST = [" but ", " though", " however", " although", " while ", "mixed feelings", "both good and bad"]

_HIGH = ["dangerous", "serious", "injur", "stranded", "sick", "poisoning", "stomach", "collapsed", "erased",
         "lost their work", "lost the data", "fainted", "faint", "dizzy", "urgent", "immediate", "unusable",
         "hazard", "burned out", "insects", "pests", "health", "nearly caused", "near-accident", "badly hurt",
         "dead for most", "down for days", "unbearable", "closed for weeks", "exam", "shock", "unsanitary",
         "suspended without notice", "overlap", "miss required"]
_LOW = ["minor", "a little", "slightly", "small", "not a big", "manageable", "bearable", "could be", "could use",
        "not urgent", "a bit", "nothing serious", "does not bother", "still usable", "still works", "still okay",
        "nothing has happened", "no one has been hurt"]

# category -> (keywords, weight). Higher total score wins; ties resolved by CATEGORY_PRIORITY.
_CATEGORY_KEYWORDS = {
    "Internet": ["wi-fi", "wifi", "internet", "wireless", "connection", "network speed", "signal", "online exam",
                 "online class", "network"],
    "Software": ["portal", "software", "website", "login", "log in", "app ", " app", "platform", "system",
                 "license", "page", "logs us out", "logging users out", "registration", "enrollment", "e-book"],
    "Hardware": ["printer", "projector", "computer", "scanner", "dryer", "faucet", "dispenser", "vending",
                 "monitor", "microphone", "keyboard", "mouse", "terminal", "fan", "router", "sensor", "turnstile",
                 "speaker", "refrigerator", "warmer", "tv", "photocopier", "barrier", "intercom", "antenna",
                 "flush valve", "machine", "access point", "detector", "charging station", "network cabinet"],
    "Facilities": ["door", "ceiling", "elevator", "roof", "seating", "sofa", "bench", "gate", "shed", "walkway",
                   "pavement", "fence", "renovat", "table", "counter", "cubicle", "room", "blackboard", "window frame",
                   "cabinet", "guard house", "stall door", "sink", "tile floor", "structural", "lounge"],
    "Cleanliness": ["dirty", "clean", "smell", "trash", "garbage", "dust", "sticky", "stain", "filthy", "spotless",
                    "sanitation", "janitor", "scraps", "unsanitary", "pests", "wipe", "tray return"],
    "Food Service": ["food", "meal", "snack", "drinks", "coffee", "price", "portion", "menu", "rice", "noodle",
                     "bakery", "sandwich", "cook", "vendor", "stall", "fruit stand", "fishball", "served"],
    "Transportation": ["shuttle", "jeepney", "tricycle", "drop-off", "pickup", "traffic", "motorcycle", "bus ",
                       "parking slot", "commute", "stranded", "entry arrangement", "car entry", "terminal near the gate"],
    "Safety": ["safety", "unsafe", "hazard", "injur", "accident", "fire", "emergency", "exit", "cctv", "security",
               "wiring", "electric", "lock", "slippery", "danger", "guard", "latch", "stairs", "staircase", "lighting",
               "crossing", "streetlight", "risky", "hurt"],
    "Environment": ["temperature", "heat", "hot", "noise", "humid", "air quality", "ventilation", "dust level",
                    "lighting level", "shade", "odor", "greenery", "exhaust", "air circulation", "sun exposure",
                    "sustainab", "weather"],
    "Academic": ["exam schedule", "schedule", "lecture", "tutoring", "assignment", "deadline", "course", "subjects",
                 "study group", "lab session", "consultation", "review session", "practical exam", "grouping",
                 "activity plan", "room assignment", "midterm", "peer"],
    "Other": ["help desk", "desk", "lost and found", "id validation", "id replacement", "suggestion", "announcement",
              "bulletin", "signage", "information", "office window", "support ticket", "assistance", "sign-in",
              "complaints", "request process", "reporting process", "sticker claiming", "student council"],
}
_CATEGORY_PRIORITY = ["Internet", "Food Service", "Transportation", "Academic", "Safety", "Cleanliness",
                      "Environment", "Software", "Hardware", "Other", "Facilities"]
_ACTIONS = {
    "Internet": "Ask IT to inspect the network and add or upgrade access points.",
    "Software": "Ask the system administrator to fix the errors and test the system before peak periods.",
    "Hardware": "Have maintenance repair or replace the equipment.",
    "Facilities": "Schedule a facilities inspection and repair.",
    "Cleanliness": "Increase the cleaning frequency and empty bins more often.",
    "Food Service": "Review food quality, prices and queue handling with the service provider.",
    "Transportation": "Review the traffic or transport arrangement and add capacity during rush hours.",
    "Safety": "Have the safety officer inspect the hazard and fix it as a priority.",
    "Environment": "Improve ventilation, lighting or noise control in the affected area.",
    "Academic": "Communicate schedules earlier and review session capacity.",
    "Other": "Assign a clear owner to respond to this request.",
}
_STOP = set("the a an and or but in on at of to is are was were be been for with that this it its as by from near "
            "very more much have has had not no so than then there their they them we our us i my me you your".split())


def _hits(text: str, words) -> int:
    return sum(1 for w in words if w in text)


def local_classify(feedback: str) -> dict:
    """Return the same structured dict the AI produces, using simple keyword rules."""
    text = " " + re.sub(r"\s+", " ", str(feedback or "").lower()) + " "
    neg, pos, neu = _hits(text, _NEG), _hits(text, _POS), _hits(text, _NEUTRAL_CUES)
    contrast = any(c in text for c in _CONTRAST)

    if "some areas and bad" in text or "good and bad" in text or (pos and neg and contrast):
        sentiment = "Mixed"
    elif neu and pos == 0 and neg <= neu:
        sentiment = "Neutral"
    elif pos > neg:
        sentiment = "Positive"
    elif neg > 0:
        sentiment = "Negative"
    elif neu:
        sentiment = "Neutral"
    else:
        # no explicit cue either way: mild wording ("a bit", "needs", "not urgent") still describes a problem
        mild = _hits(text, _LOW) + _hits(text, _HIGH) + _hits(text, ["needs", "without enough", "extra clicks", "clearer"])
        sentiment = "Negative" if mild else "Neutral"

    scores = {c: _hits(text, kws) for c, kws in _CATEGORY_KEYWORDS.items()}
    best = max(scores.values())
    category = "Other"
    if best > 0:
        category = next(c for c in _CATEGORY_PRIORITY if scores[c] == best)

    if sentiment == "Negative":
        severity = "High" if _hits(text, _HIGH) else ("Low" if _hits(text, _LOW) else "Medium")
    elif sentiment == "Mixed":
        severity = "Medium"
    else:
        severity = "Low"

    words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", text) if w not in _STOP]
    keywords = list(dict.fromkeys(words))[:4]
    main_issue = "None" if sentiment in ("Positive", "Neutral") else f"{category} issue"
    summary = str(feedback).strip()
    summary = summary if len(summary) <= 140 else summary[:137].rstrip() + "..."
    return {
        "sentiment": sentiment,
        "category": category,
        "severity": severity,
        "keywords": keywords,
        "main_issue": main_issue,
        "summary": summary,
        "suggested_action": "No action needed" if sentiment in ("Positive", "Neutral") else _ACTIONS[category],
    }
