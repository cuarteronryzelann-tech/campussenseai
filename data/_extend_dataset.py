"""
_extend_dataset.py
------------------
One-off helper that APPENDS extra, fictional feedback rows to campus_feedback.csv
so that every value the sidebar can filter on has plenty of matching records:

  - every location            (9 locations)
  - every category            (11 categories)
  - every sentiment           (Positive / Neutral / Negative / Mixed)
  - every severity            (Low / Medium / High)

Category, sentiment and severity are assigned by the AI at analysis time, so
each supplemental row below is written to be an unambiguous example of the
label shown next to it. Each category gets 6 rows (Negative-High, Negative-Medium,
Negative-Low, Positive, Neutral, Mixed), on top of the original 60 rows.

Safe to re-run: rows whose text is already in the CSV are skipped.
Run from the data/ folder:  python3 _extend_dataset.py
"""

import csv
import random
from datetime import datetime, timedelta

CSV_PATH = "campus_feedback.csv"

# (location, feedback, intended_category, intended_sentiment, intended_severity)
SUPPLEMENTAL = [
    # ---- Internet ----
    ("Wi-Fi Areas", "Wi-Fi went down across the whole engineering building during the online midterm exam, and many students lost their answers.", "Internet", "Negative", "High"),
    ("Library", "The library internet is very slow in the afternoons, pages take a full minute to load.", "Internet", "Negative", "Medium"),
    ("Cafeteria", "The Wi-Fi signal in the far corner of the cafeteria is a bit weak, but it is only a minor inconvenience.", "Internet", "Negative", "Low"),
    ("Computer Laboratory", "Internet speed in the computer lab is now excellent, large downloads finish in seconds.", "Internet", "Positive", "Low"),
    ("Wi-Fi Areas", "The campus Wi-Fi network was renamed this month, and the connection speed seems about the same as before.", "Internet", "Neutral", "Low"),
    ("Student Lounge", "The lounge Wi-Fi is fast in the morning, but it becomes very slow and keeps dropping after lunch.", "Internet", "Mixed", "Medium"),
    # ---- Hardware ----
    ("Computer Laboratory", "Several lab computers will not turn on at all, so half of the class could not take the practical exam.", "Hardware", "Negative", "High"),
    ("Classroom", "The microphone in the lecture room is broken, so students at the back cannot hear the teacher.", "Hardware", "Negative", "Medium"),
    ("Student Lounge", "One of the phone charging ports in the lounge is loose, though the other ports work fine.", "Hardware", "Negative", "Low"),
    ("Library", "The library replaced its old photocopiers with new machines that work fast and print clearly.", "Hardware", "Positive", "Low"),
    ("Computer Laboratory", "The lab received a batch of new keyboards this week, and they are the same model as the old ones.", "Hardware", "Neutral", "Low"),
    ("Classroom", "The new projector gives a bright picture, but the remote control is missing and the HDMI cable keeps failing.", "Hardware", "Mixed", "Medium"),
    # ---- Software ----
    ("Computer Laboratory", "The exam software in the lab crashed halfway through the test and all our unsaved answers were lost.", "Software", "Negative", "High"),
    ("Library", "The library catalog search website often shows errors and does not find books that are on the shelves.", "Software", "Negative", "Medium"),
    ("Computer Laboratory", "The lab computers show a software update reminder pop-up every day, which is slightly annoying.", "Software", "Negative", "Low"),
    ("Computer Laboratory", "The updated coding software installed in the lab is faster and easier to use than the old version.", "Software", "Positive", "Low"),
    ("Library", "The library website now has an updated login page, and the features look the same as before.", "Software", "Neutral", "Low"),
    ("Classroom", "The online learning platform is easy to navigate, but it logs us out every ten minutes during quizzes.", "Software", "Mixed", "Medium"),
    # ---- Facilities ----
    ("Classroom", "The elevator in the main building has been out of service for weeks, so students with disabilities cannot reach the upper floors.", "Facilities", "Negative", "High"),
    ("Library", "The library roof leaks badly during rain, and water drips over the computers and the bookshelves.", "Facilities", "Negative", "High"),
    ("Student Lounge", "The lounge tables are wobbly and there are not enough chairs for the number of students.", "Facilities", "Negative", "Medium"),
    ("Restroom", "The hooks on the restroom stall doors are missing, which is a small inconvenience.", "Facilities", "Negative", "Low"),
    ("Library", "The newly renovated library study cubicles are spacious, bright and very comfortable.", "Facilities", "Positive", "Low"),
    ("Cafeteria", "The cafeteria will be closed for two hours on Friday for a scheduled facility inspection.", "Facilities", "Neutral", "Low"),
    ("Classroom", "The new lecture hall seats are comfortable, but there are far too few power sockets and the door lock is broken.", "Facilities", "Mixed", "Medium"),
    # ---- Cleanliness ----
    ("Restroom", "The restroom near the gym is filthy, with overflowing toilets and a sewage smell that makes it unusable and unhealthy.", "Cleanliness", "Negative", "High"),
    ("Cafeteria", "Cafeteria tables and trays are often left dirty, and the floor is greasy after lunch.", "Cleanliness", "Negative", "Medium"),
    ("Campus Entrance", "There are a few candy wrappers scattered near the guard house, but nothing serious.", "Cleanliness", "Negative", "Low"),
    ("Library", "The library is spotless every morning, the cleaning staff do a great job.", "Cleanliness", "Positive", "Low"),
    ("Student Lounge", "Janitors clean the student lounge every day at 5 pm after classes end.", "Cleanliness", "Neutral", "Low"),
    ("Restroom", "The restrooms are cleaned regularly in the morning, but they are dirty and smelly again by the afternoon.", "Cleanliness", "Mixed", "Medium"),
    # ---- Food Service ----
    ("Cafeteria", "Several students got stomach aches and food poisoning after eating the chicken meal at the cafeteria.", "Food Service", "Negative", "High"),
    ("Cafeteria", "The cafeteria food is often served cold and the portions are very small for the price.", "Food Service", "Negative", "Medium"),
    ("Campus Entrance", "The snack stall near the entrance sometimes has no change for large bills, a small annoyance.", "Food Service", "Negative", "Low"),
    ("Cafeteria", "The new rice bowl meals at the cafeteria are delicious, cheap and served hot.", "Food Service", "Positive", "Low"),
    ("Student Lounge", "A coffee cart now opens in the student lounge from 7 am to 10 am on weekdays.", "Food Service", "Neutral", "Low"),
    ("Cafeteria", "The cafeteria has tasty food and friendly cooks, but the queue is very slow and they often run out of drinks.", "Food Service", "Mixed", "Medium"),
    # ---- Transportation ----
    ("Campus Entrance", "The university shuttle service was canceled without notice, leaving hundreds of students stranded with no way to get to campus.", "Transportation", "Negative", "High"),
    ("Parking Area", "There are not enough parking slots for motorcycles, and many students circle for 20 minutes to find a spot.", "Transportation", "Negative", "Medium"),
    ("Parking Area", "The bike rack in the parking area is a bit far from the buildings, but it is usable.", "Transportation", "Negative", "Low"),
    ("Parking Area", "The new shuttle route from the parking area to the main building saves a lot of walking time.", "Transportation", "Positive", "Low"),
    ("Campus Entrance", "The jeepney drop-off point was moved to the left side of the gate this week.", "Transportation", "Neutral", "Low"),
    ("Campus Entrance", "The shuttle is a big help on rainy days, but it only runs twice in the morning and is always full.", "Transportation", "Mixed", "Medium"),
    # ---- Safety ----
    ("Computer Laboratory", "There are exposed electrical wires hanging near the computers in Lab 2, which could give someone an electric shock.", "Safety", "Negative", "High"),
    ("Restroom", "The restroom floor near the gym is flooded and slippery, and a student already fell and got injured.", "Safety", "Negative", "High"),
    ("Library", "The emergency exit in the library is partly blocked by stacked boxes.", "Safety", "Negative", "Medium"),
    ("Student Lounge", "A loose floor tile in the lounge could make someone trip, but so far nobody has been hurt.", "Safety", "Negative", "Low"),
    ("Campus Entrance", "Security guards check bags politely and I feel safe entering the campus every day.", "Safety", "Positive", "Low"),
    ("Classroom", "A fire drill will be held on Thursday at 10 am, and all classes are expected to join.", "Safety", "Neutral", "Low"),
    ("Parking Area", "The new CCTV cameras make the parking area feel safer, but the lot is still very dark at night and some lamps are out.", "Safety", "Mixed", "Medium"),
    # ---- Environment ----
    ("Classroom", "The heat inside the classrooms reaches dangerous levels in the afternoon, and several students have felt dizzy and fainted.", "Environment", "Negative", "High"),
    ("Student Lounge", "Loud construction noise next to the lounge makes it impossible to study or rest.", "Environment", "Negative", "Medium"),
    ("Cafeteria", "The cafeteria is a bit warm around noon, though the fans help a little.", "Environment", "Negative", "Low"),
    ("Campus Entrance", "The new trees and garden near the campus entrance make the area cool, green and pleasant.", "Environment", "Positive", "Low"),
    ("Library", "The library has placed recycling bins for paper beside the main door.", "Environment", "Neutral", "Low"),
    ("Parking Area", "The trees around the parking area give nice shade, but the lot floods with muddy water every time it rains.", "Environment", "Mixed", "Medium"),
    # ---- Academic ----
    ("Classroom", "Our exam schedule was changed without notice and several exams now overlap, so many students may miss one of them.", "Academic", "Negative", "High"),
    ("Classroom", "The professor gives very heavy assignments with unclear instructions, and it is hard to catch up.", "Academic", "Negative", "Medium"),
    ("Library", "Some reference books for my subject are not up to date, but I can find alternatives online.", "Academic", "Negative", "Low"),
    ("Library", "The tutoring sessions organized at the library have helped me understand calculus much better.", "Academic", "Positive", "Low"),
    ("Classroom", "Midterm exams will be held next week, and the schedule is posted on the bulletin board.", "Academic", "Neutral", "Low"),
    ("Computer Laboratory", "The programming lectures are interesting, but there are not enough lab sessions to practice what we learn.", "Academic", "Mixed", "Medium"),
    # ---- Other ----
    ("Campus Entrance", "It takes more than two weeks to get a replacement student ID and the office staff are not helpful.", "Other", "Negative", "Medium"),
    ("Student Lounge", "The lost and found box is always locked and nobody knows who has the key.", "Other", "Negative", "Medium"),
    ("Library", "The bulletin board near the library has expired posters that nobody removes.", "Other", "Negative", "Low"),
    ("Student Lounge", "The student organization fair was a fun event and I met many new friends.", "Other", "Positive", "Low"),
    ("Campus Entrance", "The bulletin board near the entrance shows announcements for upcoming student events.", "Other", "Neutral", "Low"),
    ("Student Lounge", "The student council events are fun, but the announcements are always posted at the last minute.", "Other", "Mixed", "Medium"),
]


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    existing_text = {r["feedback"].strip() for r in rows}
    next_id = max(int(r["feedback_id"][2:]) for r in rows) + 1

    random.seed(7)
    start, end = datetime(2026, 6, 1), datetime(2026, 9, 20)
    added = []
    for loc, text, *_ in SUPPLEMENTAL:
        if text in existing_text:
            continue
        date = (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")
        added.append({"feedback_id": f"FB{next_id:04d}", "date": date, "location": loc, "feedback": text})
        next_id += 1

    rows += added
    rows.sort(key=lambda r: (r["date"], r["feedback_id"]))
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feedback_id", "date", "location", "feedback"])
        w.writeheader()
        w.writerows(rows)
    print(f"Added {len(added)} rows -> {len(rows)} total in {CSV_PATH}")


if __name__ == "__main__":
    main()
