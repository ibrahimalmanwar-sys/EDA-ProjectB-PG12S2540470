# EDA Mini Project B — Time-Series Forecasting Starter

Student: Brahim Al Manwari  
Student ID: PG12S2540470

This repository contains a one-file Streamlit starter app for Mini Project B. The app stops at dataset loading, audit, timestamp/target selection, optional resampling, forecast horizon selection, baseline feature table creation, export files, and the fixed AI grader.

## Included files

```text
app.py
requirements.txt
README.md
data/dataset_sample.csv
```

## Dataset

The included sample was created from the uploaded dataset:

- Original file: `UK_National_Demand_HalfHourly_2015-2019.xlsx`
- Confirmed timestamp column: `TIMESTAMP`
- Confirmed target column: `ND`
- Cleaned rows in `data/dataset_sample.csv`: 83,606
- Rows were parsed, cleaned, sorted by timestamp, and kept as a contiguous time-series slice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository named `EDA-ProjectB-PG12S2540470`.
2. Upload these files exactly:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Commit to the `main` branch.
4. Go to Streamlit Community Cloud.
5. Choose **New app**.
6. Connect the GitHub repository.
7. Set:
   - Branch: `main`
   - Main file path: `app.py`
8. Deploy.

## OpenRouter API key

The app does not hardcode any API key. To use the AI grader, provide `OPENROUTER_API_KEY` through one of these methods:

1. Streamlit Secrets: `st.secrets["OPENROUTER_API_KEY"]`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field inside the app

## What to submit

Submit the following to your instructor:

- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json`
- Exported `project_card.md`

The AI score is out of 80. The peer score out of 20 is handled separately by instructors.
