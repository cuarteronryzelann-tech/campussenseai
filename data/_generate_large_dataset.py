"""
_generate_large_dataset.py
--------------------------
Builds the large, balanced sample dataset for CampusSense AI (500+ fictional records).

Outputs (in this folder):
  campus_feedback.csv   feedback_id, date, location, feedback   <- what the app loads
  expected_labels.csv   feedback_id, location, category, sentiment, severity
                        <- the label each record was WRITTEN to have. Used only for
                           verification (check_coverage.py); the app never reads it.

Why two files: category / sentiment / severity are assigned by the AI when you click
"Analyze Feedback", so they must NOT be columns in campus_feedback.csv.

Design
  - 9 locations x 11 categories. Only combinations that make sense are generated
    (e.g. no "Food Service" reports for the Restroom), so every record is realistic.
  - Every generated combination gets 6-20 records; every location and category is covered.
  - Text = optional opener + neutral subject + valence clause (+ optional location phrase / tail).
    Subjects are deliberately neutral ("the printer near the help desk"); the clause carries
    the problem or the praise, and clauses are grouped by severity/sentiment.
  - No two records share the same (location, subject, clause), and all texts are unique.

Run from this folder:  python3 _generate_large_dataset.py
"""

import csv
import random
from collections import Counter, defaultdict
from datetime import date, timedelta

random.seed(2026)

LOCATIONS = ["Library", "Computer Laboratory", "Classroom", "Cafeteria", "Restroom",
             "Parking Area", "Student Lounge", "Campus Entrance", "Wi-Fi Areas"]
CATEGORIES = ["Internet", "Hardware", "Software", "Facilities", "Cleanliness", "Food Service",
              "Transportation", "Safety", "Environment", "Academic", "Other"]

# Slot types: H/M/L = Negative (High/Medium/Low), P = Positive, N = Neutral, X = Mixed
LABEL = {"H": ("Negative", "High"), "M": ("Negative", "Medium"), "L": ("Negative", "Low"),
         "P": ("Positive", "Low"), "N": ("Neutral", "Low"), "X": ("Mixed", "Medium")}
CYCLE = ["M", "P", "H", "X", "L", "N", "M", "P", "M", "X", "H", "N", "L"]

# ---------------------------------------------------------------------------
# Clauses per category:  H, M, L, P, N, X
# ---------------------------------------------------------------------------
CLAUSES = {
    "Internet": {
        "H": ["went down completely during the online exam, and many students lost their work",
              "has been unusable for days, so nobody can submit requirements online",
              "dropped without warning during a graded activity and would not reconnect",
              "is dead for most of the day, which is stopping our online classes"],
        "M": ["is very slow, and pages take more than a minute to load",
              "keeps disconnecting every few minutes",
              "becomes unreliable whenever many students are online",
              "often shows no internet even when the device is connected"],
        "L": ["is a little slow sometimes, but still usable",
              "is slightly weaker in the corners, only a minor issue",
              "occasionally lags for a moment, not a big problem"],
        "P": ["is fast and stable now, great for research",
              "has improved a lot this semester",
              "worked perfectly during my online class today",
              "is much more reliable after the recent upgrade"],
        "N": ["was checked by the IT staff during their regular inspection",
              "is scheduled for a routine check this weekend",
              "was reported to the IT office and logged for review"],
        "X": ["is fast in the morning but slows down badly in the afternoon",
              "works well near the entrance but drops further inside",
              "is stable most of the time, though it sometimes fails during peak hours",
              "connects quickly, but the speed for video calls is poor"],
    },
    "Software": {
        "H": ["crashed during enrollment and nobody could register for classes for hours",
              "lost the data we entered right before the deadline",
              "has been down for days, so required tasks cannot be completed",
              "failed during the exam and erased our answers"],
        "M": ["often shows errors when I try to log in",
              "is slow and freezes when several students use it",
              "keeps logging users out in the middle of a task",
              "does not save changes properly"],
        "L": ["has a confusing layout, but it still works",
              "shows an annoying pop-up every time I open it",
              "takes a few extra clicks to do simple things"],
        "P": ["is now easy to use and quick",
              "was updated and works much better than before",
              "made my request so much faster this week"],
        "N": ["will be down for scheduled maintenance this weekend",
              "was updated to a new version according to the announcement",
              "now has a new login page with the same features"],
        "X": ["is easy to navigate, but it is very slow at peak times",
              "has good features, but it logs me out too often",
              "works fine on my laptop but keeps failing on my phone"],
    },
    "Hardware": {
        "H": ["is completely broken and nobody has fixed it since the start of the month",
              "broke down in the middle of exam week and there is no replacement",
              "burned out and is now unusable"],
        "M": ["keeps malfunctioning and needs repair",
              "works only some of the time",
              "is old and breaks down often",
              "has a broken part that makes it hard to use"],
        "L": ["makes a strange noise but still works",
              "is a bit loose and wobbly, but usable",
              "has a small crack that does not affect use yet"],
        "P": ["was replaced with a new one that works perfectly",
              "was repaired quickly after I reported it",
              "is in great condition thanks to the maintenance team"],
        "N": ["was checked by the maintenance team this week",
              "is scheduled for replacement next month",
              "was moved to a different spot this week"],
        "X": ["works well when it works, but it often needs a restart",
              "is fast, but the buttons are worn out",
              "was repaired last week, though the same problem appeared again"],
    },
    "Facilities": {
        "H": ["has been closed for weeks because of serious damage",
              "is badly damaged and unusable, so students have nowhere else to go",
              "has a serious structural problem that needs urgent repair"],
        "M": ["is damaged and needs repair",
              "has been out of order for weeks",
              "is in bad condition and uncomfortable to use",
              "is worn out and needs to be replaced"],
        "L": ["has a small crack, but it is still okay to use",
              "needs repainting, but it is not urgent",
              "is a little worn, but still usable"],
        "P": ["was renovated and looks great",
              "is now comfortable and well-maintained",
              "was fixed quickly, thanks to the facilities office"],
        "N": ["will be closed for two hours on Friday for inspection",
              "was rearranged as announced last week",
              "is included in the maintenance schedule for next month"],
        "X": ["looks great after the renovation, but some parts are still unfinished",
              "is comfortable, but it is in poor shape in some corners",
              "was fixed recently, though it still needs some work"],
    },
    "Cleanliness": {
        "H": ["is filthy and unsanitary, which is a health risk",
              "has not been cleaned for days and now smells terrible",
              "is covered in garbage and pests are starting to appear",
              "is so dirty that students are avoiding the area"],
        "M": ["is often dirty and sticky",
              "is not cleaned regularly",
              "has a bad smell by the afternoon",
              "is dusty and untidy most days"],
        "L": ["has a few stains, but nothing serious",
              "could use a quick wipe now and then",
              "has some small scraps of paper, a minor issue"],
        "P": ["is spotless every morning",
              "is always clean thanks to the cleaning staff",
              "is much cleaner than last semester"],
        "N": ["is cleaned every day at 5 pm according to the schedule",
              "will be deep-cleaned next weekend as announced",
              "was inspected by the sanitation team this week"],
        "X": ["is clean in the morning, but it gets dirty quickly by afternoon",
              "is well cleaned, though the bins are not emptied often enough",
              "looks better than before, but the corners are still dusty"],
    },
    "Food Service": {
        "H": ["served spoiled food and several students had stomach aches afterwards",
              "was found with insects in the food, which is a serious hygiene problem",
              "sold food that made a number of students sick this week"],
        "M": ["is often sold out by early afternoon",
              "serves cold food and small portions for the price",
              "has a very long line that takes more than 20 minutes",
              "has raised prices again without any improvement"],
        "L": ["could offer a few more choices",
              "is a bit pricey but fine",
              "sometimes runs out of change, a minor inconvenience"],
        "P": ["serves delicious hot food at a fair price",
              "has friendly staff and quick service",
              "added healthy options that students love"],
        "N": ["is open from 7 am to 4 pm on weekdays",
              "will be closed for one afternoon for a routine inspection",
              "changed its menu this week"],
        "X": ["has tasty food, but the line is very slow",
              "is cheap and friendly, though it often runs out of drinks",
              "serves good meals, but the seating is limited"],
    },
    "Transportation": {
        "H": ["left hundreds of students stranded this morning",
              "is dangerously chaotic, and near-accidents happen every week",
              "was suspended without notice, so many students could not get to class on time",
              "is completely blocked in the morning and students miss their first classes"],
        "M": ["is very congested during the morning rush",
              "does not have enough capacity for the number of students",
              "is disorganized and wastes a lot of time",
              "often causes delays, making students late for class"],
        "L": ["could use clearer signs, but it is manageable",
              "is a bit far from the buildings, but okay",
              "is slightly crowded at times, not a big issue"],
        "P": ["has become much more organized",
              "saves us a lot of time now",
              "works smoothly even during rush hour"],
        "N": ["was adjusted this week according to the posted notice",
              "has a new schedule posted on the notice board",
              "will be adjusted during the school holiday"],
        "X": ["is a great help on rainy days, but it is always full",
              "is well managed, but it is too small for the number of students",
              "is better organized, but still crowded at 7 am"],
    },
    "Safety": {
        "H": ["is a serious hazard and someone could get badly hurt",
              "already caused an injury this week",
              "poses an immediate danger and should be closed until fixed",
              "nearly caused an accident yesterday and needs urgent attention"],
        "M": ["is unsafe and needs to be fixed soon",
              "has been unsafe for a while and needs attention",
              "is risky when it is crowded or rainy",
              "has been reported before, but nothing was done"],
        "L": ["could be safer, but nothing has happened so far",
              "has a minor issue that should be checked",
              "is a little worn, but no one has been hurt"],
        "P": ["was recently improved and I feel much safer",
              "is well maintained and secure",
              "was fixed right after we reported it"],
        "N": ["will be inspected during the fire drill on Thursday",
              "is included in the safety checklist for this month",
              "was reviewed by the safety officer this week"],
        "X": ["is better after the recent upgrade, but some parts are still risky",
              "has good security, but the response is slow",
              "was fixed, though a small hazard remains"],
    },
    "Environment": {
        "H": ["reaches dangerous levels, and some students felt dizzy or faint",
              "has become unbearable and is affecting students' health",
              "is so bad that classes and studying are being disrupted",
              "got worse this week and students are asking to be moved elsewhere"],
        "M": ["is uncomfortable for most of the afternoon",
              "is too high and makes it hard to concentrate",
              "has been bothering students for weeks",
              "gets worse every time the weather changes"],
        "L": ["is a little uncomfortable at times, but bearable",
              "is slightly off on some days, a minor thing",
              "could be better, though it does not bother me much"],
        "P": ["has improved a lot and feels comfortable now",
              "is pleasant and perfect for studying",
              "is much better after the recent changes"],
        "N": ["was measured by the facilities office this week",
              "is scheduled for review by the campus committee",
              "is being monitored as part of the sustainability program"],
        "X": ["is fine in the morning but gets uncomfortable in the afternoon",
              "improved after the changes, but it is still a problem at peak times",
              "is good in some areas and bad in others"],
    },
    "Academic": {
        "H": ["was changed without notice, and several activities now overlap",
              "is so disorganized that many students may miss required exams",
              "keeps changing at the last minute, which is ruining our preparation",
              "was announced too late, so many students missed it completely"],
        "M": ["is confusing and not well communicated",
              "does not fit the number of students who need it",
              "is often changed without enough notice",
              "is not enough for our subjects"],
        "L": ["could be a bit clearer, but it is manageable",
              "has a small conflict that can be worked around",
              "is posted late sometimes, a minor problem"],
        "P": ["is well organized and easy to follow",
              "helped me prepare for my exams much better",
              "is now clear and very convenient"],
        "N": ["is posted on the bulletin board for next week",
              "was updated according to the latest announcement",
              "will start after the midterm exams"],
        "X": ["is helpful, but there are too few sessions",
              "is well organized, but it is announced too late",
              "is good for my course, but it conflicts with my other class"],
    },
    "Other": {   # no High-severity clauses: admin / information issues are never critical
        "M": ["is often closed or unattended when we need help",
              "takes far too long to respond to requests",
              "gives confusing and inconsistent instructions",
              "is poorly organized and nobody knows who is in charge"],
        "L": ["could be a bit faster, but it works",
              "has unclear signage, a minor issue",
              "sometimes has a short line, nothing serious"],
        "P": ["was quick and helpful when I needed assistance",
              "has friendly staff who solved my problem right away",
              "improved a lot this semester"],
        "N": ["is open from 8 am to 5 pm on weekdays",
              "moved to a new location as announced",
              "will have shorter hours during the holiday break"],
        "X": ["is friendly, but the waiting time is long",
              "solved my problem, but I had to visit three times",
              "is helpful but often closed during lunch"],
    },
}

# ---------------------------------------------------------------------------
# Subjects per (location, category). A combination that is missing here is not generated.
# Subjects are singular and neutral in tone; they always start with "the".
# ---------------------------------------------------------------------------
SUBJECTS = {
    "Library": {
        "Internet": ["the Wi-Fi signal on the second floor", "the internet connection at the reading tables",
                     "the wireless network in the periodicals section", "the online journal access"],
        "Software": ["the library catalog website", "the book reservation system", "the e-book portal",
                     "the online renewal page"],
        "Hardware": ["the photocopier near the entrance", "the barcode scanner at the circulation desk",
                     "the printer beside the help desk", "the public desktop computer near the window"],
        "Facilities": ["the second-floor reading room", "the elevator to the upper floors",
                       "the group study room door", "the ceiling above the periodicals section"],
        "Cleanliness": ["the reading area floor", "the study table surface", "the reference section shelf area",
                        "the carpet near the entrance"],
        "Safety": ["the emergency exit on the ground floor", "the stairwell", "the fire extinguisher near the entrance",
                   "the electrical wiring behind the charging tables"],
        "Environment": ["the temperature on the upper floor", "the noise level near the entrance",
                        "the air quality in the reading room", "the humidity around the book stacks"],
        "Academic": ["the tutoring schedule at the library", "the consultation schedule with librarians",
                     "the reservation arrangement for discussion rooms"],
        "Other": ["the library help desk", "the lost and found counter", "the ID validation window",
                  "the suggestion box process"],
    },
    "Computer Laboratory": {
        "Internet": ["the internet connection at the workstations", "the wired network in the back row",
                     "the Wi-Fi signal inside the lab", "the network speed during programming sessions"],
        "Software": ["the exam software installed on the lab computers", "the lab login system",
                     "the programming software license server", "the lab reservation system"],
        "Hardware": ["the computer in the third row", "the projector at the front of the lab",
                     "the mouse and keyboard set at the corner station", "the lab printer"],
        "Facilities": ["the laboratory door", "the ceiling panel above the third row", "the lab table in the back row",
                       "the storage cabinet"],
        "Cleanliness": ["the lab floor", "the desk surface at the workstations", "the keyboard area at the stations",
                        "the trash can area"],
        "Safety": ["the electrical wiring near the workstations", "the emergency exit path",
                   "the power strip arrangement under the tables", "the fire extinguisher inside the lab"],
        "Environment": ["the room temperature", "the noise level from the hallway", "the air quality",
                        "the lighting level over the workstations"],
        "Academic": ["the lab session schedule", "the practical exam arrangement", "the lab grouping arrangement",
                     "the lab activity plan"],
        "Other": ["the lab assistant desk", "the equipment borrowing counter", "the sign-in process",
                  "the lab announcement board process"],
    },
    "Classroom": {
        "Internet": ["the Wi-Fi signal in Room 204", "the internet connection during online quizzes",
                     "the wireless network in the lecture hall", "the connection for online class sessions"],
        "Software": ["the online learning platform used in class", "the attendance system", "the classroom quiz app",
                     "the grade viewing portal"],
        "Hardware": ["the ceiling projector", "the classroom microphone", "the electric fan at the back",
                     "the wall-mounted speaker"],
        "Facilities": ["the classroom door", "the ceiling in Room 301", "the blackboard", "the window frame"],
        "Cleanliness": ["the classroom floor", "the whiteboard tray", "the desk area at the back",
                        "the window sill"],
        "Safety": ["the ceiling fan mounting", "the staircase outside the room", "the fire exit",
                   "the electrical outlet near the front"],
        "Environment": ["the classroom temperature", "the noise level from the hallway", "the air circulation",
                        "the humidity"],
        "Academic": ["the lecture schedule", "the exam schedule", "the room assignment for classes",
                     "the assignment deadline arrangement"],
        "Other": ["the department office window", "the lost and found counter of the building",
                  "the announcement process", "the room request process"],
    },
    "Cafeteria": {
        "Internet": ["the Wi-Fi signal in the dining area", "the internet connection near the counter",
                     "the wireless network at the far tables"],
        "Software": ["the cashless payment system", "the online meal ordering app", "the meal card balance checker",
                     "the queue number system"],
        "Hardware": ["the refrigerator display", "the cashier terminal", "the rice warmer", "the water dispenser"],
        "Facilities": ["the seating area", "the serving counter", "the cafeteria roof", "the dining table section"],
        "Cleanliness": ["the table area", "the tray return station", "the floor near the counter",
                        "the trash bin area", "the food serving counter"],
        "Food Service": ["the main meal counter", "the snack stall", "the drinks counter", "the noodle station",
                         "the rice bowl station", "the bakery stand"],
        "Safety": ["the floor near the counter", "the gas stove area", "the exit path during the lunch rush",
                   "the electrical wiring near the refrigerator"],
        "Environment": ["the temperature during lunch", "the noise level at lunch", "the air quality near the kitchen",
                        "the humidity"],
        "Other": ["the complaints counter", "the lost and found station", "the announcement board process",
                  "the suggestion box process"],
    },
    "Restroom": {
        "Hardware": ["the hand dryer", "the automatic faucet", "the flush valve", "the soap dispenser"],
        "Facilities": ["the stall door", "the sink area", "the restroom ceiling", "the tile floor by the entrance"],
        "Cleanliness": ["the restroom floor", "the toilet area", "the sink area", "the trash bin corner"],
        "Safety": ["the floor near the sinks", "the stall lock", "the lighting inside the restroom", "the door latch"],
        "Environment": ["the air quality", "the humidity", "the lighting level", "the odor level"],
        "Other": ["the reporting process for issues", "the maintenance request form", "the notice posting process"],
    },
    "Parking Area": {
        "Software": ["the vehicle sticker registration system", "the online parking permit page",
                     "the parking slot tracking app"],
        "Hardware": ["the entrance barrier gate", "the ticket machine", "the parking sensor", "the lamp post controller"],
        "Facilities": ["the covered parking shed", "the parking lot pavement", "the perimeter fence",
                       "the motorcycle shelter"],
        "Cleanliness": ["the parking shed floor", "the drainage area", "the garbage collection point",
                        "the wall by the entrance ramp"],
        "Transportation": ["the motorcycle parking section", "the shuttle pickup point", "the car entry arrangement",
                           "the drop-off zone", "the parking slot allocation"],
        "Safety": ["the lighting at night", "the pedestrian crossing lane", "the CCTV coverage", "the pavement"],
        "Environment": ["the heat from the pavement", "the dust level", "the noise level from the road",
                        "the temperature under the shed"],
        "Other": ["the guard station help desk", "the lost and found request process",
                  "the sticker claiming window"],
    },
    "Student Lounge": {
        "Internet": ["the Wi-Fi signal near the sofas", "the internet connection at the lounge tables",
                     "the wireless network in the lounge corner"],
        "Software": ["the lounge reservation system", "the charging locker app", "the event sign-up page"],
        "Hardware": ["the vending machine", "the wall-mounted TV", "the charging station", "the water dispenser",
                     "the electric fan"],
        "Facilities": ["the lounge sofa area", "the lounge ceiling", "the lounge counter", "the lounge door"],
        "Cleanliness": ["the sofa area", "the floor", "the trash bin area", "the microwave corner"],
        "Food Service": ["the coffee cart", "the microwave meal station", "the snack vending corner",
                         "the drinks vending machine", "the sandwich stand"],
        "Safety": ["the tile flooring", "the extension cord arrangement", "the emergency exit", "the glass door"],
        "Environment": ["the temperature", "the noise level", "the air quality", "the lighting level"],
        "Academic": ["the peer tutoring schedule", "the study group arrangement",
                     "the exam review session schedule"],
        "Other": ["the student council desk", "the lost and found box process", "the suggestion box process"],
    },
    "Campus Entrance": {
        "Software": ["the visitor registration system", "the ID scanning software", "the gate log system"],
        "Hardware": ["the turnstile", "the ID scanner", "the metal detector", "the gate intercom"],
        "Facilities": ["the main gate", "the guard house", "the entrance waiting shed", "the pedestrian walkway"],
        "Cleanliness": ["the area around the guard house", "the sidewalk at the gate", "the garbage spot near the gate",
                        "the waiting shed"],
        "Food Service": ["the snack stall near the gate", "the street food cart by the entrance",
                         "the fruit stand outside the gate", "the coffee stall beside the guard house",
                         "the fishball vendor at the gate"],
        "Transportation": ["the jeepney drop-off area", "the shuttle service", "the tricycle waiting arrangement at the gate",
                           "the tricycle terminal near the gate", "the morning entry arrangement at the gate"],
        "Safety": ["the security check", "the sidewalk", "the streetlight near the gate",
                   "the road crossing at the gate"],
        "Environment": ["the dust level", "the noise level from traffic", "the heat at the gate",
                        "the air quality from vehicle exhaust"],
        "Other": ["the information desk", "the ID replacement window", "the lost and found counter",
                  "the visitor assistance process"],
    },
    "Wi-Fi Areas": {
        "Internet": ["the Wi-Fi signal at the quadrangle", "the connection near the gymnasium",
                     "the wireless network in the dormitory area", "the internet speed at the outdoor benches"],
        "Software": ["the Wi-Fi login portal", "the campus network sign-in page",
                     "the account verification page for the network"],
        "Hardware": ["the access point on the quadrangle", "the router near the gym", "the outdoor antenna",
                     "the network cabinet"],
        "Facilities": ["the outdoor bench area", "the covered walkway", "the shaded seating near the quadrangle",
                       "the charging bench"],
        "Safety": ["the loose-prone cable run along the walkway", "the outdoor power box",
                   "the antenna mount above the bench", "the floor of the covered walkway"],
        "Environment": ["the heat in the open quadrangle", "the noise level on the walkway",
                        "the humidity at the outdoor benches", "the dust level"],
        "Other": ["the IT help desk booth", "the account assistance window", "the support ticket process"],
    },
}
# neutral wording fix for one subject
SUBJECTS["Wi-Fi Areas"]["Safety"][0] = "the cable run along the walkway"

# Records per (location, category): base per category, small boost for the two smallest locations
BASE_N = {"Internet": 8, "Software": 6, "Hardware": 6, "Facilities": 6, "Cleanliness": 6, "Food Service": 14,
          "Transportation": 20, "Safety": 6, "Environment": 6, "Academic": 11, "Other": 6}
BOOST = {"Restroom": 2, "Wi-Fi Areas": 2}

OPENERS = {
    "Negative": ["", "", "", "I would like to report that", "Just a heads up that",
                 "Many of us have noticed that", "Please look into this:", "Honestly,"],
    "Positive": ["", "", "", "I want to thank the staff because", "Good news:", "Just wanted to share that",
                 "I appreciate that", "Honestly,"],
    "Neutral": ["", "", "FYI,", "Just noting that", "For your information,", "I saw a notice that"],
    "Mixed": ["", "", "Mixed feelings:", "To be fair,", "I have both good and bad to say:", "Honestly,"],
}
LOC_PHRASE = {"Library": "in the library", "Computer Laboratory": "in the computer lab",
              "Classroom": "in the classroom", "Cafeteria": "at the cafeteria", "Restroom": "in the restroom",
              "Parking Area": "in the parking area", "Student Lounge": "in the student lounge",
              "Campus Entrance": "at the campus entrance", "Wi-Fi Areas": "in the Wi-Fi area"}
LOC_WORDS = ["library", "lab", "room", "cafeteria", "restroom", "parking", "lounge", "entrance", "gate",
             "wi-fi area", "quadrangle", "dining", "campus"]
NEG_TAIL = {
    "Library": ["which makes studying here really hard", "and it disrupts everyone working in the library",
                "so many of us have to look for another place to study"],
    "Computer Laboratory": ["and it disrupts our lab activities", "so we cannot finish our exercises on time",
                            "which slows down the whole class"],
    "Classroom": ["and it disrupts our lectures", "so it is hard to focus in class",
                  "which affects the whole class"],
    "Cafeteria": ["and it affects everyone during the lunch rush", "so many students end up going off campus",
                  "which is frustrating during our short breaks"],
    "Restroom": ["and it makes using the restroom uncomfortable", "so students avoid using it",
                 "which really needs attention"],
    "Parking Area": ["and it makes coming to campus stressful", "so many of us arrive late",
                     "which is a daily headache"],
    "Student Lounge": ["and it makes the lounge hard to enjoy between classes",
                       "so students go elsewhere to rest", "which defeats the purpose of the lounge"],
    "Campus Entrance": ["and it affects everyone who comes in through the gate", "especially during the morning rush",
                        "which is a poor first impression for visitors"],
    "Wi-Fi Areas": ["and it affects students who study outdoors", "so we cannot rely on it between classes",
                    "which is frustrating when we have online tasks"],
}
POS_TAIL = ["and I hope it stays this way", "which really helps students", "thank you for the effort"]


def with_location(subject, loc):
    s = subject.lower()
    if any(w in s for w in LOC_WORDS) or random.random() > 0.6:
        return subject
    return f"{subject} {LOC_PHRASE[loc]}"


def compose(loc, cat, subject, clause, slot):
    sentiment = LABEL[slot][0]
    subj = with_location(subject, loc)
    body = f"{subj} {clause}"
    simple = "," not in clause and " and " not in clause and " but " not in clause
    if slot in ("M", "H") and simple and cat != "Other" and random.random() < 0.5:
        body += ", " + random.choice(NEG_TAIL[loc])
    elif slot == "P" and simple and random.random() < 0.35:
        body += ", " + random.choice(POS_TAIL)
    opener = random.choice(OPENERS[sentiment])
    text = f"{opener} {body}".strip() if opener else body[0].upper() + body[1:]
    if opener and opener[0].islower():
        text = text[0].upper() + text[1:]
    return text.rstrip(".") + "."


def random_weekday(start=date(2026, 6, 1), end=date(2026, 9, 18)):
    while True:
        d = start + timedelta(days=random.randint(0, (end - start).days))
        if d.weekday() < 5:
            return d


def build():
    records, seen_text, seen_pairs = [], set(), set()
    for loc in LOCATIONS:
        for cat in CATEGORIES:
            subjects = SUBJECTS[loc].get(cat)
            if not subjects:
                continue
            n = BASE_N[cat] + BOOST.get(loc, 0)
            slots = [CYCLE[i % len(CYCLE)] for i in range(n)]
            if cat == "Other":                       # admin/info issues are never High severity
                slots = ["M" if x == "H" else x for x in slots]
            use_count = Counter()
            for slot in slots:
                key = slot
                pairs = [(s, c) for s in subjects for c in CLAUSES[cat][key] if (loc, s, c) not in seen_pairs]
                random.shuffle(pairs)
                pairs.sort(key=lambda p: use_count[p[0]])       # spread over subjects
                for s, c in pairs:
                    text = compose(loc, cat, s, c, slot)
                    if text in seen_text:
                        continue
                    seen_pairs.add((loc, s, c)); seen_text.add(text); use_count[s] += 1
                    sentiment, severity = LABEL[slot]
                    records.append({"date": random_weekday().isoformat(), "location": loc, "feedback": text,
                                    "category": cat, "sentiment": sentiment, "severity": severity})
                    break
                else:
                    raise RuntimeError(f"Ran out of unique texts for {loc}/{cat}/{slot}")
    records.sort(key=lambda r: (r["date"], r["location"], r["feedback"]))
    for i, r in enumerate(records, start=1):
        r["feedback_id"] = f"FB{i:04d}"
    return records


def verify(rows):
    n = len(rows)
    combos = Counter((r["location"], r["category"]) for r in rows)
    by_loc, by_cat = Counter(r["location"] for r in rows), Counter(r["category"] for r in rows)
    checks = {
        "1. 500+ records": n >= 500,
        "2. every location has reports": all(by_loc[l] > 0 for l in LOCATIONS),
        "3. every category has reports": all(by_cat[c] > 0 for c in CATEGORIES),
        "4. filtering by any single location returns results": all(by_loc[l] >= 5 for l in LOCATIONS),
        "5. filtering by any single category returns results": all(by_cat[c] >= 5 for c in CATEGORIES),
        "6. every intended location+category combination returns results":
            all(combos[(l, c)] >= 5 for l in LOCATIONS for c in CATEGORIES if SUBJECTS[l].get(c)),
        "7. no empty locations/categories": len(by_loc) == len(LOCATIONS) and len(by_cat) == len(CATEGORIES),
        "   all texts unique": len({r["feedback"] for r in rows}) == n,
        "   no empty/null required fields": all(str(r[k]).strip() for r in rows
                                                 for k in ("feedback_id", "date", "location", "feedback")),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "verification failed"
    return by_loc, by_cat, combos


def main():
    rows = build()
    with open("campus_feedback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feedback_id", "date", "location", "feedback"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open("expected_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feedback_id", "location", "category", "sentiment", "severity"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} records.\n")
    by_loc, by_cat, combos = verify(rows)
    print("\nRecords per location:", dict(by_loc))
    print("Records per category:", dict(by_cat))
    print("Sentiment:", dict(Counter(r["sentiment"] for r in rows)))
    print("Severity:", dict(Counter(r["severity"] for r in rows)))
    print(f"Location x category combinations generated: {len(combos)} (min {min(combos.values())} records each)")


if __name__ == "__main__":
    main()
