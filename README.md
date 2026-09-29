# CampusSense AI
### An AI-Powered Campus Feedback and Problem Detection System

**Course:** CS 315 – Application Development and Emerging Technologies
**Activity:** Activity 3 – GenAI Application Project

---

## 1. Project Description

CampusSense AI is a Streamlit web application that uses a Generative AI model (open-source
models served by the Hugging Face Inference API: Qwen2.5-7B-Instruct, with automatic fallback through Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507, Gemma-3 12B/4B, Granite-4.2-8B, Phi-4 and Qwen3-14B) to analyze free-text campus feedback — comments students leave about the library, computer
labs, classrooms, cafeteria, restrooms, parking, and more — and automatically turns that raw
text into structured, actionable insight: sentiment, problem category, severity, keywords,
a short summary, and a suggested action for staff.

## 2. Problem Statement

Campuses collect feedback constantly (suggestion boxes, forms, surveys), but almost none of
it gets systematically reviewed. Staff have no easy way to see *which* problems are most
common, *how serious* they are, or *where* they're happening most, so recurring issues
(a flaky Wi-Fi router, a broken restroom faucet) can go unaddressed for a long time simply
because nobody aggregated the complaints.

## 3. Purpose

To demonstrate, for CS 315 Activity 3, an original and genuinely useful GenAI application
that ingests unstructured feedback and produces structured, prioritized, visual insight that
a real campus administration office could act on.

## 4. Objectives

- Load and clean a real-world-style CSV dataset with Pandas.
- Use a GenAI API to extract structured information from unstructured text.
- Build an interactive, filterable Streamlit dashboard with multiple visualizations.
- Rank recurring problems by a transparent, explainable priority score.
- Provide a grounded chatbot that answers questions using only the loaded dataset.
- Package the whole thing so it can be deployed for free on Streamlit Community Cloud.

## 5. Target Users

- Campus administration / facilities management staff
- Student affairs offices reviewing suggestion-box or survey data
- IT support teams tracking recurring technical complaints
- CS 315 instructors/evaluators reviewing the project

## 6. Features

- Bundled 60-record CSV dataset loaded automatically (no upload needed)
- Automatic Pandas-based data cleaning (duplicates, missing values, date parsing, whitespace)
- GenAI-powered per-feedback analysis: sentiment, category, severity, keywords, main issue,
  summary, suggested action — returned as structured JSON
- Sidebar filters: location, category, sentiment, severity, date range
- Dashboard metric cards: total feedback, positive, negative, high-priority, top category, top location
- Six Plotly visualizations: sentiment distribution, categories, locations, severity,
  problems over time, top keywords
- AI-generated "Campus Problem Summary" derived from the current filtered data (never hardcoded)
- **Priority Problems** ranking (frequency + severity + recency, with the formula shown in-app)
- Interactive, searchable **Feedback Details** table
- **CampusSense Assistant** chatbot that answers questions grounded only in the loaded dataset
- One-click export of the fully analyzed dataset as `campus_feedback_analyzed.csv`
- Friendly error handling throughout (missing file, bad columns, missing API key, API/network
  errors, invalid AI JSON, empty dataset, duplicates, missing feedback text)

## 7. Dataset Description

`data/campus_feedback.csv` contains **60 fictional, realistic** feedback records with columns:

| Column        | Description                                   |
|---------------|------------------------------------------------|
| `feedback_id` | Unique record ID, e.g. `FB0001`                |
| `date`        | Date the feedback was submitted (`YYYY-MM-DD`) |
| `location`    | Campus location the feedback refers to         |
| `feedback`    | The free-text feedback comment                 |

Locations covered: Library, Computer Laboratory, Classroom, Cafeteria, Restroom, Parking
Area, Student Lounge, Campus Entrance, Wi-Fi Areas. Comments include a realistic mix of
positive, neutral, and negative feedback. **No real names, student IDs, or other personal
information are used anywhere in the dataset.**

(`data/_generate_dataset.py` is the helper script used to generate this sample dataset and
is not part of the running application — keep it only if you want to regenerate/extend the
sample data.)

## 8. Technologies Used

- **Python 3.10+**
- **Streamlit** – interactive web dashboard
- **Pandas** – data loading and cleaning
- **Hugging Face Inference API** (`huggingface_hub`; 9-model fallback chain, see `DEFAULT_MODEL_CHAIN` in `utils/ai_analysis.py`) – GenAI text analysis, summary generation, and chatbot
- **Plotly Express** – interactive charts
- **python-dotenv** – loading the API key from a local `.env` file

## 9. System Workflow

1. The bundled CSV (`data/campus_feedback.csv`) is loaded automatically on startup.
2. `utils/data_cleaning.py` validates columns, removes duplicates/missing values, normalizes
   text, and parses dates using Pandas.
3. User clicks **Analyze Feedback**. `utils/ai_analysis.py` sends each feedback row to the
   Hugging Face model chain with a strict JSON-only prompt, validates the response
   against an allowed schema, and appends the results as new DataFrame columns.
4. The dashboard (`app.py`) applies the sidebar filters to the analyzed DataFrame and renders
   metrics, charts (`utils/visualization.py`), the priority-problems ranking, the feedback
   table, the AI summary, and the chatbot — all of which update live as filters change.
5. The user can download the fully analyzed dataset as CSV at any time.

## 10. Project Structure

```
CampusSenseAI/
│
├── app.py                     # Main Streamlit application
├── requirements.txt
├── README.md
├── .env                       # HF_TOKEN (not committed with a real key)
│
├── data/
│   ├── campus_feedback.csv        # Sample dataset (60 records)
│   └── _generate_dataset.py       # Helper script that generated the sample dataset
│
├── utils/
│   ├── data_cleaning.py       # Pandas loading/cleaning/validation
│   ├── ai_analysis.py         # Hugging Face integration, summary, chatbot
│   └── visualization.py       # Plotly chart builders
│
└── assets/
    └── logo.png                # Dashboard logo
```

## 11. Installation Instructions

```bash
# 1. Clone or download the project, then move into it
cd CampusSenseAI

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 12. Environment Variable / API Key Setup

1. Create a free access token at https://huggingface.co/settings/tokens
   (a "Read" token, or a fine-grained token with *Make calls to Inference Providers*).
2. Open the `.env` file in the project root and set:

   ```
   HF_TOKEN=hf_your_token_here
   ```

3. To change the models without editing code, set `HF_MODELS=org/model-a,org/model-b` in `.env`.
   All models draw from the same monthly free credits; a `402 Payment Required` error means
   the credits are used up (analysis then falls back to the local keyword classifier).
   `meta-llama/Llama-3.1-8B-Instruct` is a gated model: accept its license on its
   Hugging Face page if you want the last fallback to work.

4. **Never commit your real `.env` file** with a real key to a public GitHub repository —
   add `.env` to `.gitignore` before pushing.

## 13. How to Run the Application (Locally)

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (usually at `http://localhost:8501`). The dataset
loads automatically; click **Analyze Feedback** in the sidebar.

## 14. Example Questions for the CampusSense Assistant Chatbot

- "What is the most common problem?"
- "Which location has the most complaints?"
- "What are the major problems in the library?"
- "Summarize the negative feedback."
- "What problems have high severity?"
- "What improvements are commonly suggested?"

If the loaded dataset doesn't contain enough information to answer a question, the
assistant will say so directly rather than guessing.

## 15. Testing Procedure

1. **Data loading/cleaning:** Run with the bundled sample; confirm the cleaned preview shows
   60 rows, no duplicates, and a valid `date` column (visible before clicking Analyze).
2. **Error handling:** Try replacing the CSV with an empty file, or one missing the `feedback` column, and
   (temporarily) an invalid Hugging Face token — confirm each shows a clear on-screen message
   instead of crashing.
3. **AI analysis:** Click **Analyze Feedback** with a valid API key; confirm every row gets a
   sentiment/category/severity/keywords/summary/suggested action.
4. **Filters:** Change location/category/sentiment/severity/date filters and confirm the
   metrics, all six charts, the priority table, and the feedback table update together.
5. **Priority Problems:** Confirm categories with more reports and higher severity rank
   higher, matching the displayed formula.
6. **Chatbot:** Ask each example question above and confirm answers are grounded in the
   visible data; ask an unrelated/unanswerable question and confirm the assistant says the
   dataset doesn't contain enough information.
7. **Export:** Download `campus_feedback_analyzed.csv` and confirm it opens correctly with
   all analysis columns populated.

## 16. Deployment Instructions (Streamlit Community Cloud)

1. Push this project to a **public or private GitHub repository** (make sure `.env` with a
   real key is excluded via `.gitignore` — only commit the placeholder version if needed).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select your repository, branch, and set the main file path to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   HF_TOKEN = "your-real-token-here"
   ```
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch the app.
6. (Optional) In `app.py`, `os.getenv("HF_TOKEN")` will automatically pick up Streamlit
   Cloud secrets as long as the secret name matches exactly.

## 17. How This Project Satisfies CS 315 Activity 3 Requirements

| Requirement                              | Where it's satisfied                                                        |
|-------------------------------------------|-------------------------------------------------------------------------------|
| Original application idea                 | CampusSense AI — a novel campus feedback/problem-detection concept           |
| Different dataset                         | Custom-generated `campus_feedback.csv` (feedback text, not reused data)      |
| Load & clean data with Pandas             | `utils/data_cleaning.py`                                                     |
| Integrate a GenAI API                     | `utils/ai_analysis.py` (Hugging Face models, structured JSON extraction)              |
| Build an interactive Streamlit interface  | `app.py` — tabs, sidebar filters, metrics, chat interface                    |
| Visualize analysis results                | `utils/visualization.py` — 6 Plotly charts, all filter-reactive              |
| Test and iterate                          | See Testing Procedure above                                                  |
| Prepare for Streamlit Community Cloud     | `requirements.txt`, `.env`/secrets pattern, deployment steps above           |
| Include filters                           | Location, category, sentiment, severity, date range (sidebar)                |
| Include a chatbot for dataset questions   | CampusSense Assistant tab, grounded strictly in the loaded dataset           |

## 18. Future Improvements

- Persist analyzed results to a database instead of re-analyzing on every session
- Support multiple GenAI providers (e.g., OpenAI, Anthropic, local models) as a fallback
- Add authentication so only authorized staff can view/export feedback
- Batch/async API calls to speed up analysis of very large datasets
- Automatic email/notification alerts when a new High-severity problem is detected

## 19. Limitations

- Analysis quality depends entirely on the quality of the underlying GenAI model's output.
- Large datasets will make one API call per row, which can be slow and cost more — batching
  is listed as a future improvement.
- The chatbot's grounding is based on aggregate statistics plus a handful of sample rows, not
  the full dataset, to keep prompts small — very specific numeric questions across the whole
  dataset may occasionally be approximate.
- Requires an active internet connection and a valid Hugging Face token to run analysis.
