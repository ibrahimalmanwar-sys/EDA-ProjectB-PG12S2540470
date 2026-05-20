import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


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

# Pink dashboard theme + Rambo/rainbow-style flag banner.
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fff0f7 0%, #ffe1ef 42%, #ffd6e8 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(255, 224, 240, 0.78);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #ffd6e8 0%, #fff4fa 100%);
    }
    .block-container {
        padding-top: 1.8rem;
    }
    h1, h2, h3 {
        color: #8a0f4d;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(226, 58, 132, 0.18);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(138, 15, 77, 0.08);
    }
    .rambo-flag-banner {
        border-radius: 24px;
        padding: 18px 22px;
        margin: 6px 0 22px 0;
        color: white;
        font-weight: 800;
        letter-spacing: 0.3px;
        box-shadow: 0 12px 34px rgba(138, 15, 77, 0.18);
        background:
            linear-gradient(90deg,
                #e40303 0%, #e40303 16.66%,
                #ff8c00 16.66%, #ff8c00 33.33%,
                #ffed00 33.33%, #ffed00 50%,
                #008026 50%, #008026 66.66%,
                #004dff 66.66%, #004dff 83.33%,
                #750787 83.33%, #750787 100%);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(226, 58, 132, 0.18);
        border-radius: 22px;
        padding: 18px 20px;
        margin: 10px 0 18px 0;
        box-shadow: 0 8px 24px rgba(138, 15, 77, 0.08);
    }
    .insight-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        background: #ffe1ef;
        color: #8a0f4d;
        font-weight: 700;
        border: 1px solid rgba(138, 15, 77, 0.12);
    }
    </style>
    <div class="rambo-flag-banner">🏳️‍🌈 Rambo Flag Forecast Lab — Pink Edition</div>
    """,
    unsafe_allow_html=True,
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
st.markdown(
    """
    <div class="glass-card">
    <b>Modeling goal:</b> train forecasting models using a chronological split, compare them against simple baselines,
    and produce a metrics table named <code>results_df</code> for grading/export.
    </div>
    """,
    unsafe_allow_html=True,
)

# STUDENT ADDITIONS — MODELING
# Time-based split, engineered forecasting features, model training, predictions, and metrics table.

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def safe_mape(y_true, y_pred):
    """MAPE that safely ignores zero actual values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    value = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return float(value) if np.isfinite(value) else np.nan


def regression_metrics(y_true, y_pred):
    """Return common forecast accuracy metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": safe_mape(y_true, y_pred),
    }


# Start from the prepared starter feature table and add stronger student features.
student_feature_df = modeling_df.copy()
student_feature_df["hour"] = student_feature_df[timestamp_col].dt.hour
student_feature_df["dayofweek"] = student_feature_df[timestamp_col].dt.dayofweek
student_feature_df["weekofyear"] = student_feature_df[timestamp_col].dt.isocalendar().week.astype(int)
student_feature_df["month"] = student_feature_df[timestamp_col].dt.month
student_feature_df["quarter"] = student_feature_df[timestamp_col].dt.quarter
student_feature_df["is_weekend"] = (student_feature_df["dayofweek"] >= 5).astype(int)
student_feature_df["is_month_start"] = student_feature_df[timestamp_col].dt.is_month_start.astype(int)
student_feature_df["is_month_end"] = student_feature_df[timestamp_col].dt.is_month_end.astype(int)
student_feature_df["hour_sin"] = np.sin(2 * np.pi * student_feature_df["hour"] / 24)
student_feature_df["hour_cos"] = np.cos(2 * np.pi * student_feature_df["hour"] / 24)
student_feature_df["month_sin"] = np.sin(2 * np.pi * student_feature_df["month"] / 12)
student_feature_df["month_cos"] = np.cos(2 * np.pi * student_feature_df["month"] / 12)

# Adaptive lags: use the richest features possible without destroying small datasets.
candidate_lags = [1, 2, 3, 24, 48, 168, 336]
active_lags = [lag for lag in candidate_lags if len(student_feature_df) > lag + 60]
if not active_lags:
    active_lags = [1] if len(student_feature_df) > 10 else []

for lag in active_lags:
    student_feature_df[f"lag_{lag}"] = student_feature_df[target_col].shift(lag)

candidate_windows = [3, 6, 12, 24, 48, 168]
active_windows = [window for window in candidate_windows if len(student_feature_df) > window + 60]
if not active_windows:
    active_windows = [3] if len(student_feature_df) > 15 else []

for window in active_windows:
    student_feature_df[f"rolling_mean_{window}"] = student_feature_df[target_col].shift(1).rolling(window).mean()
    student_feature_df[f"rolling_std_{window}"] = student_feature_df[target_col].shift(1).rolling(window).std()

if "lag_1" in student_feature_df.columns:
    student_feature_df["demand_change_1"] = student_feature_df[target_col] - student_feature_df["lag_1"]
else:
    student_feature_df["demand_change_1"] = np.nan

student_feature_columns = [
    target_col,
    "hour",
    "dayofweek",
    "weekofyear",
    "month",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "demand_change_1",
]
student_feature_columns += [f"lag_{lag}" for lag in active_lags]
student_feature_columns += [f"rolling_mean_{window}" for window in active_windows]
student_feature_columns += [f"rolling_std_{window}" for window in active_windows]
student_feature_columns = [col for col in student_feature_columns if col in student_feature_df.columns]

model_data = student_feature_df.dropna(subset=student_feature_columns + ["y_target"]).copy()
for col in student_feature_columns:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
model_data = model_data.dropna(subset=student_feature_columns + ["y_target"]).reset_index(drop=True)

results_df = None
predictions_df = pd.DataFrame()
feature_importance_df = pd.DataFrame()
student_split_summary = {}
trained_student_models = {}

if len(model_data) < 60:
    st.warning("Not enough rows after feature engineering for a reliable chronological train/validation/test split.")
else:
    n_rows = len(model_data)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)

    train_df = model_data.iloc[:train_end].copy()
    validation_df = model_data.iloc[train_end:validation_end].copy()
    test_df = model_data.iloc[validation_end:].copy()

    X_train = train_df[student_feature_columns]
    y_train = train_df["y_target"]
    X_validation = validation_df[student_feature_columns]
    y_validation = validation_df["y_target"]
    X_test = test_df[student_feature_columns]
    y_test = test_df["y_target"]
    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)

    student_split_summary = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df[timestamp_col].min()),
        "train_end": str(train_df[timestamp_col].max()),
        "test_start": str(test_df[timestamp_col].min()),
        "test_end": str(test_df[timestamp_col].max()),
    }

    models = {
        "Naive previous value": None,
        "Rolling mean baseline": "rolling",
        "Linear Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=160,
                        max_depth=14,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }

    metric_rows = []
    test_predictions = {}

    for model_name, model in models.items():
        for split_name, split_df, X_split, y_split in [
            ("validation", validation_df, X_validation, y_validation),
            ("test", test_df, X_test, y_test),
        ]:
            if model is None:
                y_pred = split_df[target_col].to_numpy()
            elif model == "rolling":
                rolling_cols = [col for col in student_feature_columns if col.startswith("rolling_mean_")]
                preferred_col = "rolling_mean_24" if "rolling_mean_24" in rolling_cols else (rolling_cols[0] if rolling_cols else target_col)
                y_pred = split_df[preferred_col].to_numpy()
            else:
                if split_name == "validation":
                    fitted_model = model.fit(X_train, y_train)
                else:
                    fitted_model = model.fit(X_train_validation, y_train_validation)
                    trained_student_models[model_name] = fitted_model
                y_pred = fitted_model.predict(X_split)

            metrics = regression_metrics(y_split, y_pred)
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "train_rows": int(len(train_df) if split_name == "validation" else len(train_df) + len(validation_df)),
                    "test_rows": int(len(split_df)),
                    "MAE": round(metrics["MAE"], 3),
                    "RMSE": round(metrics["RMSE"], 3),
                    "MAPE": round(metrics["MAPE"], 3) if not np.isnan(metrics["MAPE"]) else np.nan,
                }
            )

            if split_name == "test":
                test_predictions[model_name] = y_pred

    results_df = pd.DataFrame(metric_rows).sort_values(["split", "RMSE"]).reset_index(drop=True)
    feature_columns = student_feature_columns

    predictions_df = test_df[[timestamp_col]].copy()
    predictions_df["actual"] = y_test.to_numpy()
    for model_name, y_pred in test_predictions.items():
        predictions_df[model_name] = y_pred

    test_results = results_df[results_df["split"] == "test"].sort_values("RMSE")
    best_model_name = test_results.iloc[0]["model"]
    predictions_df["best_model"] = best_model_name
    predictions_df["best_prediction"] = predictions_df[best_model_name]
    predictions_df["residual"] = predictions_df["actual"] - predictions_df["best_prediction"]
    predictions_df["absolute_error"] = predictions_df["residual"].abs()

    # Feature importances for tree models when available.
    for model_name in ["Random Forest", "Gradient Boosting"]:
        fitted_model = trained_student_models.get(model_name)
        if fitted_model is not None:
            estimator = fitted_model.named_steps.get("model")
            if hasattr(estimator, "feature_importances_"):
                temp = pd.DataFrame(
                    {
                        "model": model_name,
                        "feature": student_feature_columns,
                        "importance": estimator.feature_importances_,
                    }
                )
                feature_importance_df = pd.concat([feature_importance_df, temp], ignore_index=True)

    st.subheader("Modeling evidence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", f"{len(train_df):,}")
    c2.metric("Validation rows", f"{len(validation_df):,}")
    c3.metric("Test rows", f"{len(test_df):,}")
    c4.metric("Features", f"{len(student_feature_columns):,}")

    st.write(
        "Chronological split used: first 70% for training, next 15% for validation, final 15% for future-period testing."
    )
    st.write(
        "Student-added features include adaptive lags, rolling means, rolling volatility, calendar flags, and cyclical time encodings."
    )
    st.dataframe(results_df, use_container_width=True)


st.header("6. STUDENT ADDITIONS — DASHBOARD")
st.markdown(
    """
    <div class="glass-card">
    <b>Dashboard goal:</b> make the forecasting story visible through KPIs, model comparisons,
    forecast-vs-actual plots, residual analysis, demand-pattern infographics, and feature-importance charts.
    </div>
    """,
    unsafe_allow_html=True,
)

# STUDENT ADDITIONS — DASHBOARD
# Extra plots, KPIs, residual analysis, model comparisons, and interpretation.

if "results_df" in globals() and isinstance(results_df, pd.DataFrame) and not results_df.empty:
    test_results = results_df[results_df["split"] == "test"].copy().sort_values("RMSE")
    validation_results = results_df[results_df["split"] == "validation"].copy().sort_values("RMSE")

    if not test_results.empty:
        best_row = test_results.iloc[0]
        best_model_name = str(best_row["model"])
        baseline_row = test_results[test_results["model"] == "Naive previous value"]
        baseline_rmse = float(baseline_row["RMSE"].iloc[0]) if not baseline_row.empty else np.nan
        best_rmse = float(best_row["RMSE"])
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse * 100) if np.isfinite(baseline_rmse) and baseline_rmse != 0 else np.nan

        st.subheader("Forecasting KPI Infographics")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Best model", best_model_name)
        kpi2.metric("Test MAE", f"{best_row['MAE']:,.2f}")
        kpi3.metric("Test RMSE", f"{best_rmse:,.2f}")
        kpi4.metric("Test MAPE", f"{best_row['MAPE']:,.2f}%" if pd.notna(best_row["MAPE"]) else "N/A")
        kpi5.metric("RMSE gain vs naive", f"{improvement:,.1f}%" if np.isfinite(improvement) else "N/A")

        st.markdown(
            f"""
            <span class="insight-pill">Best test model: {best_model_name}</span>
            <span class="insight-pill">Chronological future holdout</span>
            <span class="insight-pill">Validation + test metrics</span>
            <span class="insight-pill">Residual diagnostics included</span>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Model Metrics Comparison")
        if px is not None:
            metrics_long = results_df.melt(
                id_vars=["model", "split"],
                value_vars=["MAE", "RMSE", "MAPE"],
                var_name="metric",
                value_name="value",
            ).dropna()
            fig_metrics = px.bar(
                metrics_long,
                x="model",
                y="value",
                color="split",
                facet_col="metric",
                barmode="group",
                title="MAE, RMSE, and MAPE by Model and Split",
            )
            fig_metrics.update_xaxes(tickangle=-35)
            fig_metrics.update_layout(height=430, showlegend=True)
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            st.dataframe(results_df, use_container_width=True)

        st.subheader("Validation-to-Test Stability")
        stability_df = results_df.pivot_table(index="model", columns="split", values="RMSE", aggfunc="first").reset_index()
        if "validation" in stability_df.columns and "test" in stability_df.columns:
            stability_df["RMSE_change"] = stability_df["test"] - stability_df["validation"]
            stability_df["RMSE_change_pct"] = (stability_df["RMSE_change"] / stability_df["validation"]) * 100
            if px is not None:
                fig_stability = px.bar(
                    stability_df.sort_values("RMSE_change_pct"),
                    x="model",
                    y="RMSE_change_pct",
                    title="RMSE Change from Validation to Future Test Period",
                    labels={"RMSE_change_pct": "RMSE change (%)", "model": "Model"},
                )
                fig_stability.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_stability, use_container_width=True)
            st.dataframe(stability_df.round(3), use_container_width=True)

if "predictions_df" in globals() and isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
    best_model_name = str(predictions_df["best_model"].iloc[0])
    forecast_plot_df = predictions_df[[timestamp_col, "actual", best_model_name, "best_prediction", "residual", "absolute_error"]].copy()
    forecast_plot_df = forecast_plot_df.rename(columns={best_model_name: "prediction"})

    st.subheader("Actual vs Predicted Demand")
    max_display_rows = int(min(1000, len(forecast_plot_df)))
    if max_display_rows < 24:
        display_window = max_display_rows
        st.caption(f"Showing all {display_window:,} available test periods.")
    else:
        display_window = st.slider(
            "Number of latest test periods to display",
            min_value=24,
            max_value=max_display_rows,
            value=int(min(336, max_display_rows)),
            step=24,
        )
    latest_plot = forecast_plot_df.tail(display_window)

    if px is not None:
        actual_pred_long = latest_plot[[timestamp_col, "actual", "prediction"]].melt(
            id_vars=timestamp_col,
            var_name="series",
            value_name="demand",
        )
        fig_forecast = px.line(
            actual_pred_long,
            x=timestamp_col,
            y="demand",
            color="series",
            title=f"Actual vs Predicted Demand — {best_model_name}",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.line_chart(latest_plot.set_index(timestamp_col)[["actual", "prediction"]])

    st.subheader("Residual Analysis")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Mean residual", f"{forecast_plot_df['residual'].mean():,.2f}")
    r2.metric("Median residual", f"{forecast_plot_df['residual'].median():,.2f}")
    r3.metric("Residual std", f"{forecast_plot_df['residual'].std():,.2f}")
    r4.metric("Largest abs. error", f"{forecast_plot_df['absolute_error'].max():,.2f}")

    if px is not None:
        fig_resid_time = px.line(
            latest_plot,
            x=timestamp_col,
            y="residual",
            title="Residuals Over Time: Actual Minus Predicted",
        )
        st.plotly_chart(fig_resid_time, use_container_width=True)

        fig_resid_hist = px.histogram(
            forecast_plot_df,
            x="residual",
            nbins=45,
            title="Residual Distribution",
        )
        st.plotly_chart(fig_resid_hist, use_container_width=True)

        fig_scatter = px.scatter(
            forecast_plot_df,
            x="actual",
            y="prediction",
            title="Predicted vs Actual Demand",
            labels={"actual": "Actual demand", "prediction": "Predicted demand"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(forecast_plot_df["residual"].dropna(), bins=45)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Actual minus predicted")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Highest-Error Forecast Periods")
    largest_errors = forecast_plot_df.sort_values("absolute_error", ascending=False).head(10).copy()
    st.dataframe(largest_errors[[timestamp_col, "actual", "prediction", "residual", "absolute_error"]], use_container_width=True)

    st.subheader("Error Pattern by Hour and Day")
    error_pattern = forecast_plot_df.copy()
    error_pattern["hour"] = error_pattern[timestamp_col].dt.hour
    error_pattern["dayofweek"] = error_pattern[timestamp_col].dt.day_name()
    hourly_error = error_pattern.groupby("hour", as_index=False)["absolute_error"].mean()
    if px is not None:
        fig_hourly_error = px.bar(
            hourly_error,
            x="hour",
            y="absolute_error",
            title="Average Absolute Forecast Error by Hour",
            labels={"hour": "Hour of day", "absolute_error": "Mean absolute error"},
        )
        st.plotly_chart(fig_hourly_error, use_container_width=True)
    else:
        st.bar_chart(hourly_error.set_index("hour"))

if "model_data" in globals() and isinstance(model_data, pd.DataFrame) and not model_data.empty:
    st.subheader("Demand Pattern Infographics")
    pattern_df = model_data[[timestamp_col, target_col]].copy()
    pattern_df["hour"] = pattern_df[timestamp_col].dt.hour
    pattern_df["dayofweek_num"] = pattern_df[timestamp_col].dt.dayofweek
    pattern_df["dayofweek"] = pattern_df[timestamp_col].dt.day_name()
    pattern_df["month"] = pattern_df[timestamp_col].dt.month

    col_left, col_right = st.columns(2)

    with col_left:
        hourly_profile = pattern_df.groupby("hour", as_index=False)[target_col].mean()
        if px is not None:
            fig_hourly = px.line(
                hourly_profile,
                x="hour",
                y=target_col,
                markers=True,
                title="Average Demand by Hour",
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.line_chart(hourly_profile.set_index("hour"))

    with col_right:
        monthly_profile = pattern_df.groupby("month", as_index=False)[target_col].mean()
        if px is not None:
            fig_monthly = px.bar(
                monthly_profile,
                x="month",
                y=target_col,
                title="Average Demand by Month",
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.bar_chart(monthly_profile.set_index("month"))

    heatmap_df = (
        pattern_df.groupby(["dayofweek_num", "hour"], as_index=False)[target_col]
        .mean()
        .pivot(index="dayofweek_num", columns="hour", values=target_col)
    )
    heatmap_df.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(heatmap_df.index)]
    if go is not None:
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns,
                y=heatmap_df.index,
                colorbar=dict(title="Avg demand"),
            )
        )
        fig_heatmap.update_layout(title="Demand Heatmap by Day of Week and Hour", xaxis_title="Hour", yaxis_title="Day")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.dataframe(heatmap_df, use_container_width=True)

if "feature_importance_df" in globals() and isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty:
    st.subheader("Feature Importance")
    top_features = feature_importance_df.sort_values("importance", ascending=False).groupby("model").head(12)
    if px is not None:
        fig_importance = px.bar(
            top_features.sort_values("importance"),
            x="importance",
            y="feature",
            color="model",
            orientation="h",
            title="Top Forecasting Features from Tree Models",
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.dataframe(top_features, use_container_width=True)

st.subheader("Forecasting Interpretation")
if "results_df" in globals() and isinstance(results_df, pd.DataFrame) and not results_df.empty:
    test_results = results_df[results_df["split"] == "test"].copy().sort_values("RMSE")
    if not test_results.empty:
        best = test_results.iloc[0]
        worst = test_results.iloc[-1]
        st.markdown(
            f"""
            - The best model on the future test period is **{best['model']}**, with RMSE **{best['RMSE']:,.2f}** and MAE **{best['MAE']:,.2f}**.
            - The weakest test model is **{worst['model']}**, with RMSE **{worst['RMSE']:,.2f}**.
            - The key grading evidence is the chronological split: models are evaluated on later timestamps, not random rows.
            - Residual charts show where the forecast underestimates or overestimates demand, which is often more informative than a single accuracy number.
            """
        )
else:
    st.info("Model results are unavailable. Check the dataset size and selected timestamp/target columns.")

st.header("7. Notes for export")
default_data_integrity_notes = (
    f"Timestamp parsing used {timestamp_col}; target parsing used {target_col}. "
    f"The app drops invalid timestamp/target rows, sorts chronologically, reports missingness, "
    f"and allows optional resampling before forecasting. Dropped rows: {dropped_rows}."
)
default_dashboard_notes = (
    "The dashboard includes pink-themed KPI cards, model comparison charts, validation-to-test stability, "
    "actual-vs-predicted forecasts, residual diagnostics, largest-error periods, hourly/monthly demand patterns, "
    "a day-hour demand heatmap, and feature importance where available."
)
if "results_df" in globals() and isinstance(results_df, pd.DataFrame) and not results_df.empty:
    _test_results_for_notes = results_df[results_df["split"] == "test"].copy().sort_values("RMSE")
    if not _test_results_for_notes.empty:
        _best_for_notes = _test_results_for_notes.iloc[0]
        default_insights = (
            f"The best model on the future test split is {_best_for_notes['model']} with "
            f"RMSE {_best_for_notes['RMSE']:,.2f}. The most important forecasting lesson is to judge performance "
            f"on later unseen timestamps and compare complex models against simple baselines."
        )
    else:
        default_insights = "Use the validation and test metrics to compare model stability and future-period accuracy."
else:
    default_insights = "Use the validation and test metrics to compare model stability and future-period accuracy."

data_integrity_notes = st.text_area(
    "Data integrity notes",
    value=default_data_integrity_notes,
    placeholder="Discuss missing timestamps, missing target values, outliers, resampling choices, and any limitations.",
    height=110,
)
dashboard_notes = st.text_area(
    "Dashboard notes",
    value=default_dashboard_notes,
    placeholder="Describe the plots/KPIs you added and how they support the forecasting task.",
    height=100,
)
insights = st.text_area(
    "Insights",
    value=default_insights,
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
    modeling_rows=len(model_data) if "model_data" in globals() and isinstance(model_data, pd.DataFrame) else len(modeling_df),
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
st.info(
    "This app now includes student modeling evidence, a metrics table, dashboard additions, and default interpretation notes. Review the notes before exporting or grading."
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
