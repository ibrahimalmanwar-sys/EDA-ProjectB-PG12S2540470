import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "TIMESTAMP"
DEFAULT_TARGET_COL = "ND"
DEFAULT_STUDENT_NAME = "Brahim Al Manwari"
DEFAULT_STUDENT_ID = "PG12S2540470"


st.set_page_config(
    page_title="EDA Mini Project B — Time-Series Forecasting",
    page_icon="📈",
    layout="wide",
)


def read_openrouter_key():
    """Read OpenRouter key from Streamlit secrets, environment, or UI input."""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "")

    if not key:
        key = st.sidebar.text_input(
            "OpenRouter API key",
            type="password",
            help="Used only when you click the AI grader button.",
        )

    return key


def load_dataset(path):
    """Load a local CSV dataset."""
    return pd.read_csv(path)


def audit_dataframe(df):
    """Create simple audit tables."""
    dtype_table = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
        }
    )
    missing_table = (
        df.isna()
        .mean()
        .mul(100)
        .round(3)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_percent"})
        .sort_values("missing_percent", ascending=False)
    )
    return dtype_table, missing_table


def clean_time_series(df, timestamp_col, target_col):
    """Parse timestamp, convert target, drop invalid rows, and sort by time."""
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")
    before_rows = len(cleaned)
    cleaned = cleaned.dropna(subset=[timestamp_col, target_col])
    cleaned = cleaned.sort_values(timestamp_col).reset_index(drop=True)
    dropped_rows = before_rows - len(cleaned)
    return cleaned, dropped_rows


def infer_time_coverage(cleaned, timestamp_col):
    """Return min, max, and inferred median step."""
    if cleaned.empty:
        return None, None, "Unavailable"
    min_time = cleaned[timestamp_col].min()
    max_time = cleaned[timestamp_col].max()
    diffs = cleaned[timestamp_col].sort_values().diff().dropna()
    if diffs.empty:
        inferred_step = "Unavailable"
    else:
        inferred_step = str(diffs.median())
    return min_time, max_time, inferred_step


def apply_optional_resampling(cleaned, timestamp_col, target_col, resample_rule):
    """Optionally resample target to a selected frequency."""
    ts = cleaned[[timestamp_col, target_col]].copy()
    ts = ts.set_index(timestamp_col).sort_index()
    if resample_rule != "None":
        ts = ts.resample(resample_rule)[target_col].mean().to_frame()
    ts = ts.dropna(subset=[target_col]).reset_index()
    return ts


def build_baseline_features(ts, timestamp_col, target_col, horizon):
    """Create baseline time-series features only."""
    feature_df = ts[[timestamp_col, target_col]].copy()
    feature_df["lag_1"] = feature_df[target_col].shift(1)
    feature_df["lag_24"] = feature_df[target_col].shift(24)
    feature_df["rolling_mean_24"] = feature_df[target_col].shift(1).rolling(24).mean()
    feature_df["hour"] = feature_df[timestamp_col].dt.hour
    feature_df["weekend"] = feature_df[timestamp_col].dt.dayofweek >= 5
    feature_df["month"] = feature_df[timestamp_col].dt.month
    feature_df["y_target"] = feature_df[target_col].shift(-horizon)

    feature_columns = [
        "lag_1",
        "lag_24",
        "rolling_mean_24",
        "hour",
        "weekend",
        "month",
    ]
    modeling_df = feature_df.dropna(subset=feature_columns + ["y_target"]).copy()
    X = modeling_df[feature_columns]
    y = modeling_df["y_target"]
    return feature_df, modeling_df, X, y, feature_columns


def dataframe_records_or_empty(value):
    """Return DataFrame records when available; otherwise an empty list."""
    if isinstance(value, pd.DataFrame):
        safe_value = value.replace([np.inf, -np.inf], np.nan)
        return safe_value.where(pd.notna(safe_value), None).to_dict(orient="records")
    return []


def make_submission_json(
    student_name,
    student_id,
    deployed_url,
    repo_url,
    project_title,
    project_goal,
    data_path,
    original_rows,
    cleaned_rows,
    dropped_rows,
    timestamp_col,
    target_col,
    min_time,
    max_time,
    inferred_step,
    resample_rule,
    horizon,
    feature_columns,
    modeling_rows,
    has_feature_table,
    results_df,
    dashboard_notes,
    data_integrity_notes,
    insights,
):
    """Build evidence JSON for export and AI grading."""
    has_metrics_table = isinstance(results_df, pd.DataFrame)

    return {
        "student": {
            "name": student_name,
            "id": student_id,
        },
        "links": {
            "deployed_streamlit_url": deployed_url,
            "github_repo_url": repo_url,
        },
        "project": {
            "title": project_title,
            "goal": project_goal,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "dataset": {
            "path": data_path,
            "original_rows": int(original_rows),
            "cleaned_rows": int(cleaned_rows),
            "dropped_invalid_timestamp_or_target_rows": int(dropped_rows),
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "time_min": str(min_time),
            "time_max": str(max_time),
            "inferred_time_step": inferred_step,
            "resampling_rule": resample_rule,
        },
        "forecasting_setup": {
            "horizon_steps": int(horizon),
            "baseline_feature_columns": feature_columns,
            "feature_table_rows_after_dropna": int(modeling_rows),
            "has_baseline_feature_table": bool(has_feature_table),
        },
        "evidence_flags": {
            "has_metrics_table": has_metrics_table,
            "has_student_modeling_additions": has_metrics_table,
            "has_student_dashboard_notes": bool(dashboard_notes.strip()),
            "has_data_integrity_discussion": bool(data_integrity_notes.strip()),
            "has_insights": bool(insights.strip()),
        },
        "student_notes": {
            "data_integrity_notes": data_integrity_notes,
            "dashboard_notes": dashboard_notes,
            "insights": insights,
        },
        "results_table": dataframe_records_or_empty(results_df),
    }


def make_project_card(submission):
    """Create a markdown project card for download."""
    project = submission["project"]
    dataset = submission["dataset"]
    setup = submission["forecasting_setup"]
    flags = submission["evidence_flags"]

    lines = [
        f"# {project['title']}",
        "",
        f"Student: {submission['student']['name']}",
        f"Student ID: {submission['student']['id']}",
        "",
        "## Goal",
        project["goal"],
        "",
        "## Dataset",
        f"- Path: {dataset['path']}",
        f"- Timestamp column: {dataset['timestamp_column']}",
        f"- Target column: {dataset['target_column']}",
        f"- Time coverage: {dataset['time_min']} to {dataset['time_max']}",
        f"- Inferred step: {dataset['inferred_time_step']}",
        f"- Cleaned rows: {dataset['cleaned_rows']}",
        f"- Dropped invalid rows: {dataset['dropped_invalid_timestamp_or_target_rows']}",
        f"- Resampling rule: {dataset['resampling_rule']}",
        "",
        "## Forecasting setup",
        f"- Horizon steps: {setup['horizon_steps']}",
        f"- Baseline features: {', '.join(setup['baseline_feature_columns'])}",
        f"- Feature table rows: {setup['feature_table_rows_after_dropna']}",
        "",
        "## Evidence flags",
        f"- Metrics table present: {flags['has_metrics_table']}",
        f"- Data integrity discussion present: {flags['has_data_integrity_discussion']}",
        f"- Insights present: {flags['has_insights']}",
        "",
        "## Student notes",
        "### Data integrity",
        submission["student_notes"]["data_integrity_notes"] or "Not provided yet.",
        "",
        "### Dashboard",
        submission["student_notes"]["dashboard_notes"] or "Not provided yet.",
        "",
        "### Insights",
        submission["student_notes"]["insights"] or "Not provided yet.",
    ]
    return "\n".join(lines)


def parse_ai_response(text):
    """Try strict JSON parsing first, then extract the first JSON object."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as exc:
            return None, f"Found JSON-like text, but parsing failed: {exc}"

    return None, "No valid JSON object found in the AI response."


def call_openrouter_grader(api_key, evidence_json):
    """Call OpenRouter AI grader using the fixed model and prompt."""
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
        "<insert submission.json contents here>",
        json.dumps(evidence_json, indent=2),
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "EDA Mini Project B AI Grader",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


st.title("EDA Mini Project B — Time-Series Forecasting Starter")
st.caption("Starter app: audit, column selection, resampling, baseline features, exports, and fixed /80 AI grader.")

with st.sidebar:
    st.header("Student info")
    student_name = st.text_input("Student name", value=DEFAULT_STUDENT_NAME)
    student_id = st.text_input("Student ID", value=DEFAULT_STUDENT_ID)
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    repo_url = st.text_input("GitHub repo URL", value="")
    project_title = st.text_input("Project title", value="UK National Demand Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast future electricity demand using historical half-hourly demand data.",
        height=90,
    )
    openrouter_key = read_openrouter_key()

st.header("1. Load local dataset")
data_path = st.text_input("Dataset path", value=DEFAULT_DATA_PATH)

try:
    df = load_dataset(data_path)
except Exception as exc:
    st.error(f"Could not load dataset from {data_path}: {exc}")
    st.stop()

st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

dtype_table, missing_table = audit_dataframe(df)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Columns and inferred dtypes")
    st.dataframe(dtype_table, use_container_width=True)
with col_b:
    st.subheader("Missing values, top 10")
    st.dataframe(missing_table.head(10), use_container_width=True)

st.header("2. Choose timestamp and target columns")
columns = list(df.columns)

timestamp_index = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
timestamp_col = st.selectbox("Timestamp column", columns, index=timestamp_index)

numeric_candidates = []
for col in columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().mean() > 0.5:
        numeric_candidates.append(col)

if DEFAULT_TARGET_COL in columns:
    target_index = columns.index(DEFAULT_TARGET_COL)
else:
    target_index = columns.index(numeric_candidates[0]) if numeric_candidates else 0

target_col = st.selectbox("Target column", columns, index=target_index)

cleaned, dropped_rows = clean_time_series(df, timestamp_col, target_col)
if cleaned.empty:
    st.error("No valid rows remain after parsing timestamp and target. Choose different columns.")
    st.stop()

min_time, max_time, inferred_step = infer_time_coverage(cleaned, timestamp_col)

st.subheader("Cleaned time-series summary")
summary_cols = st.columns(4)
summary_cols[0].metric("Original rows", f"{len(df):,}")
summary_cols[1].metric("Cleaned rows", f"{len(cleaned):,}")
summary_cols[2].metric("Dropped rows", f"{dropped_rows:,}")
summary_cols[3].metric("Inferred step", inferred_step)

st.write(f"Time coverage: **{min_time}** to **{max_time}**")

st.header("3. Optional resampling and horizon")
resample_rule = st.selectbox(
    "Resampling rule",
    options=["None", "30min", "H", "D"],
    index=0,
    help="Use None to keep the original data frequency.",
)
horizon = st.number_input(
    "Forecast horizon, in future rows/steps",
    min_value=1,
    max_value=336,
    value=1,
    step=1,
)

ts = apply_optional_resampling(cleaned, timestamp_col, target_col, resample_rule)
feature_df, modeling_df, X, y, feature_columns = build_baseline_features(
    ts, timestamp_col, target_col, int(horizon)
)

st.header("4. Baseline feature table")
st.write(
    "The starter app creates baseline features only. Add your own models, metrics, and extra visuals under the placeholders below."
)
st.write(f"Prepared X shape: **{X.shape}**")
st.write(f"Prepared y length: **{len(y):,}**")
st.dataframe(modeling_df.head(20), use_container_width=True)

with st.expander("Show target over time preview"):
    preview = ts[[timestamp_col, target_col]].dropna().copy()
    if not preview.empty:
        chart_data = preview.set_index(timestamp_col)[target_col]
        st.line_chart(chart_data)

st.header("5. STUDENT ADDITIONS — MODELING")
st.info("Paste your modeling, time-based split, evaluation, and metrics table code below this marker in app.py.")
st.code(
    """# STUDENT ADDITIONS — MODELING
# Paste your own time-based split, model training, predictions, and metrics here.
# Create a pandas DataFrame named results_df with your metrics table.
# Example column names: model, split, MAE, RMSE, MAPE
results_df = None
""",
    language="python",
)

# STUDENT ADDITIONS — MODELING
results_df = None

st.header("6. STUDENT ADDITIONS — DASHBOARD")
st.info("Paste extra dashboard plots, KPIs, and interpretation below this marker in app.py.")
st.code(
    """# STUDENT ADDITIONS — DASHBOARD
# Add extra plots, KPIs, residual analysis, or comparison charts here.
# Keep your additions focused and clearly linked to the forecasting question.
""",
    language="python",
)

# STUDENT ADDITIONS — DASHBOARD

st.header("7. Notes for export")
data_integrity_notes = st.text_area(
    "Data integrity notes",
    value="",
    placeholder="Discuss missing timestamps, missing target values, outliers, resampling choices, and any limitations.",
    height=110,
)
dashboard_notes = st.text_area(
    "Dashboard notes",
    value="",
    placeholder="Describe the plots/KPIs you added and how they support the forecasting task.",
    height=100,
)
insights = st.text_area(
    "Insights",
    value="",
    placeholder="Summarize the most important demand patterns and forecasting lessons.",
    height=100,
)

submission = make_submission_json(
    student_name=student_name,
    student_id=student_id,
    deployed_url=deployed_url,
    repo_url=repo_url,
    project_title=project_title,
    project_goal=project_goal,
    data_path=data_path,
    original_rows=len(df),
    cleaned_rows=len(cleaned),
    dropped_rows=dropped_rows,
    timestamp_col=timestamp_col,
    target_col=target_col,
    min_time=min_time,
    max_time=max_time,
    inferred_step=inferred_step,
    resample_rule=resample_rule,
    horizon=int(horizon),
    feature_columns=feature_columns,
    modeling_rows=len(modeling_df),
    has_feature_table=not modeling_df.empty,
    results_df=results_df,
    dashboard_notes=dashboard_notes,
    data_integrity_notes=data_integrity_notes,
    insights=insights,
)

submission_json_text = json.dumps(submission, indent=2)
project_card_text = make_project_card(submission)

st.header("8. Export files")
export_col1, export_col2 = st.columns(2)
with export_col1:
    st.download_button(
        "Download submission.json",
        data=submission_json_text,
        file_name="submission.json",
        mime="application/json",
    )
with export_col2:
    st.download_button(
        "Download project_card.md",
        data=project_card_text,
        file_name="project_card.md",
        mime="text/markdown",
    )

with st.expander("Preview submission.json"):
    st.json(submission)

st.header("9. AI grader (/80)")
st.write(f"Model: `{OPENROUTER_MODEL}`")
st.warning(
    "The starter app has no modeling or metrics by default. The AI grader will score higher only after you add evidence such as a time-based split, metrics table, dashboard additions, and insights."
)

if st.button("Run AI grader"):
    if not openrouter_key:
        st.error("Provide an OpenRouter API key through Streamlit Secrets, environment variable, or the password field.")
    else:
        try:
            with st.spinner("Calling AI grader..."):
                raw_output = call_openrouter_grader(openrouter_key, submission)
            parsed, parse_error = parse_ai_response(raw_output)
            if parsed is not None:
                st.success("AI grader returned valid JSON.")
                st.json(parsed)
            else:
                st.error(parse_error)
                st.text_area("Raw AI output", raw_output, height=300)
        except Exception as exc:
            st.error(f"AI grader failed: {exc}")
