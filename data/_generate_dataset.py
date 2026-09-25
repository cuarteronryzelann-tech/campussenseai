"""
_generate_dataset.py
---------------------
One-off helper script used to generate the sample campus_feedback.csv
dataset shipped with the CampusSense AI project.

This file is NOT part of the running application - it is kept only so
the sample dataset can be regenerated or extended if needed. It contains
no personally identifiable information; all feedback text is fictional.

Run with:  python3 _generate_dataset.py
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# Feedback templates grouped by campus location.
# Each location has a mix of positive, neutral, and negative comments
# so the dataset gives the GenAI analysis realistic variety to work with.
FEEDBACK_TEMPLATES = {
    "Library": [
        "The library Wi-Fi keeps disconnecting every few minutes, making it hard to research.",
        "Study areas on the second floor are always packed by 10am, we need more seating.",
        "I really appreciate how quiet and well-organized the library has become this semester.",
        "Several power outlets near the reading tables are broken and haven't been fixed in weeks.",
        "The new self-checkout kiosk for borrowing books is fast and very convenient.",
        "Air conditioning in the library is too weak during afternoon hours, it gets uncomfortably warm.",
        "Staff at the library help desk were very friendly and helped me find resources quickly.",
        "There aren't enough charging stations for laptops in the library, especially near exam week.",
        "The group study rooms need better soundproofing, noise leaks in from the hallway.",
        "Book collection in the CS section is outdated, most titles are from over ten years ago.",
    ],
    "Computer Laboratory": [
        "Half of the computers in Lab 2 are running extremely slow and crash during compiling.",
        "The lab technician replaced our mouse and keyboard right away when I reported the issue.",
        "Internet connection in the computer lab drops constantly, especially during peak hours.",
        "Several monitors in Lab 3 have dead pixels and flickering screens that strain the eyes.",
        "New lab computers installed this semester are fast and run our software smoothly.",
        "Printer in the computer lab has been out of ink for two weeks with no replacement.",
        "The lab is well-maintained and the seats are comfortable for long programming sessions.",
        "Software licenses expired on several machines, we can't run our development tools anymore.",
        "Air conditioning in the computer lab is excellent, keeps the equipment and students cool.",
        "USB ports on most lab computers don't work, we can't plug in our own flash drives.",
    ],
    "Classroom": [
        "Projector in Room 204 keeps flickering and shutting off in the middle of lectures.",
        "The classroom chairs are broken and uncomfortable, some have loose screws.",
        "New whiteboards installed in the building are a huge improvement for lectures.",
        "Classroom is too small for the number of students enrolled, we barely have space.",
        "Lighting in the classroom is dim, making it hard to read slides from the back row.",
        "The air conditioning unit makes a loud rattling noise that distracts during lectures.",
        "Classrooms were cleaned and repainted over the break, they look great now.",
        "Electrical outlets near the front of the room don't work, we can't charge our laptops.",
        "Acoustics in the lecture hall are poor, it's hard to hear the professor from the back.",
        "The renovated classroom with new desks is much more comfortable than before.",
    ],
    "Cafeteria": [
        "Food quality in the cafeteria has really improved this semester, more variety too.",
        "Lines at the cafeteria during lunch hour are extremely long, sometimes over 30 minutes.",
        "Some of the tables in the cafeteria are sticky and not cleaned between meal times.",
        "Prices for meals went up again but the portion sizes stayed the same or got smaller.",
        "The new vegetarian options at the cafeteria are delicious and reasonably priced.",
        "Cafeteria ran out of food options by 1pm, students who arrive later have no choices left.",
        "Staff at the cafeteria counter are always polite and quick with orders.",
        "Trash bins near the cafeteria seating area overflow constantly and attract flies.",
        "Seating space in the cafeteria is not enough during peak hours, many students eat standing.",
        "The cafeteria added self-service water stations which is a nice convenient touch.",
    ],
    "Restroom": [
        "Restrooms near the engineering building are often out of soap and paper towels.",
        "Restroom on the third floor has a broken faucet that has been leaking for days.",
        "Cleaning staff keep the restrooms near the library spotless, really appreciate it.",
        "One of the stalls in the men's restroom near the gym has a broken lock.",
        "Restroom ventilation is poor, there's a persistent unpleasant smell most days.",
        "Hand dryers in the restroom near the cafeteria have not worked in over a month.",
        "The renovated restrooms in the new wing are clean, modern, and well-stocked.",
        "There's no toilet paper refilled regularly in the restrooms near the parking area.",
        "Restroom floors are often wet and slippery, could be a safety hazard.",
        "Sanitary disposal bins in the women's restroom are emptied regularly, very clean.",
    ],
    "Parking Area": [
        "Parking area is full by 8am, students have to park far off campus and walk in.",
        "Potholes in the parking lot near gate 2 have damaged several students' tires.",
        "Security guards at the parking entrance are attentive and check IDs properly.",
        "Lighting in the parking area at night is very dim, feels unsafe walking to the car.",
        "New parking lines and signage make it much easier to find a spot now.",
        "There is no designated parking for motorcycles, they end up blocking car spaces.",
        "CCTV cameras were added to the parking area which makes students feel safer.",
        "Drainage in the parking lot is poor, it floods heavily after any rain.",
        "Parking fees increased without any improvement to the facilities.",
        "The new covered parking shed protects vehicles from sun and rain nicely.",
    ],
    "Student Lounge": [
        "The student lounge sofas are torn and worn out, they need replacement.",
        "Vending machines in the student lounge are frequently out of stock or broken.",
        "New board games and a ping pong table in the lounge are a great addition.",
        "Student lounge gets very noisy in the afternoon, hard to relax or study there.",
        "Wi-Fi signal in the student lounge is weak compared to other parts of campus.",
        "The lounge is a comfortable place to relax between classes, well maintained overall.",
        "Air conditioning in the student lounge barely works during hot afternoons.",
        "Charging ports were added to the lounge tables which students really appreciate.",
        "Trash isn't collected often enough in the lounge, bins overflow by midday.",
        "The redesigned student lounge with bean bags and plants feels very inviting.",
    ],
    "Campus Entrance": [
        "Security checks at the campus entrance are quick and don't cause delays.",
        "The main gate gets extremely congested during morning rush hour with long queues.",
        "New ID scanner at the entrance frequently malfunctions, causing long lines.",
        "Sidewalk near the campus entrance is cracked and uneven, a tripping hazard.",
        "Guards at the entrance are courteous and help visitors find their way.",
        "There is no shade or waiting area near the entrance during hot weather.",
        "The entrance was recently repainted and looks much more welcoming now.",
        "Traffic flow at the entrance is chaotic, tricycles and cars block each other constantly.",
        "New signage at the campus entrance makes navigation much clearer for visitors.",
        "Streetlights near the entrance are broken, it gets very dark early in the evening.",
    ],
    "Wi-Fi Areas": [
        "Campus Wi-Fi disconnects constantly in the engineering building, very frustrating.",
        "Wi-Fi speed near the library is fast and reliable, great for downloading resources.",
        "Wi-Fi coverage doesn't reach the outdoor benches near the quadrangle at all.",
        "The IT department upgraded the Wi-Fi routers and the connection is much more stable now.",
        "Too many students on the same Wi-Fi network causes it to slow down significantly.",
        "Wi-Fi password changes without notice, we often can't reconnect during class.",
        "Connection near the gymnasium is nonexistent, we can't access online resources there.",
        "New mesh Wi-Fi system installed this semester covers the whole campus well.",
        "Wi-Fi in the dormitory areas is unreliable especially in the evening.",
        "IT support responded quickly when I reported the Wi-Fi outage near the cafeteria.",
    ],
}

LOCATIONS = list(FEEDBACK_TEMPLATES.keys())

def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

def generate_dataset(num_records=60):
    start_date = datetime(2026, 6, 1)
    end_date = datetime(2026, 9, 20)

    rows = []
    used_pairs = set()
    feedback_id = 1

    # Ensure every template is used at least once for good coverage
    all_pairs = [(loc, text) for loc, texts in FEEDBACK_TEMPLATES.items() for text in texts]
    random.shuffle(all_pairs)

    while len(rows) < num_records:
        if all_pairs:
            location, feedback = all_pairs.pop()
        else:
            location = random.choice(LOCATIONS)
            feedback = random.choice(FEEDBACK_TEMPLATES[location])

        date = random_date(start_date, end_date).strftime("%Y-%m-%d")
        rows.append({
            "feedback_id": f"FB{feedback_id:04d}",
            "date": date,
            "location": location,
            "feedback": feedback,
        })
        feedback_id += 1

    # Sort by date for readability
    rows.sort(key=lambda r: r["date"])
    return rows

def main():
    rows = generate_dataset(num_records=60)
    output_path = "campus_feedback.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feedback_id", "date", "location", "feedback"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} records -> {output_path}")

if __name__ == "__main__":
    main()
