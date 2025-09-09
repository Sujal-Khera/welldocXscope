# streamlit_app.py
import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np

# -----------------------------
# 1. Load model
# -----------------------------
# If you saved it before:
model = lgb.Booster(model_file='lgb_deterioration_model.txt')

# Or if still in memory (from training above), use `model` directly

# -----------------------------
# 2. Load or provide dataset
# -----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['OBS_DATE'])
    return df

st.title("Patient Deterioration Risk Dashboard")

uploaded_file = st.file_uploader("Upload patient_day_df CSV", type="csv")
if uploaded_file:
    patient_df = load_data(uploaded_file)

    # -----------------------------
    # 3. Prepare features for model
    # -----------------------------
    features = [
        'DALY','QALY','QOLS','308136','310798','314076','59621000',
        'AGE_AT_OBS',
        'DALY_lag_1','DALY_lag_7','DALY_lag_30',
        'QALY_lag_1','QALY_lag_7','QALY_lag_30',
        'QOLS_lag_1','QOLS_lag_7','QOLS_lag_30',
        'AGE_AT_OBS_lag_1','AGE_AT_OBS_lag_7','AGE_AT_OBS_lag_30',
        '308136_lag_1','308136_lag_7','308136_lag_30',
        '310798_lag_1','310798_lag_7','310798_lag_30',
        '314076_lag_1','314076_lag_7','314076_lag_30',
        '59621000_lag_1','59621000_lag_7','59621000_lag_30',
        'DALY_rollmean_7','DALY_rollmean_30',
        'QALY_rollmean_7','QALY_rollmean_30',
        'QOLS_rollmean_7','QOLS_rollmean_30',
        'AGE_AT_OBS_rollmean_7','AGE_AT_OBS_rollmean_30',
        '308136_rollmean_7','308136_rollmean_30',
        '310798_rollmean_7','310798_rollmean_30',
        '314076_rollmean_7','314076_rollmean_30',
        '59621000_rollmean_7','59621000_rollmean_30',
        'day_of_week','month','day','is_weekend'
    ]

    X = patient_df[features]

    # -----------------------------
    # 4. Compute probabilities
    # -----------------------------
    patient_df['risk_score'] = model.predict(X)

    # -----------------------------
    # 5. Display table & filters
    # -----------------------------
    st.subheader("Cohort Risk Scores")

    # Risk threshold filter
    threshold = st.slider("Risk threshold to highlight high-risk patients", 0.0, 1.0, 0.2, 0.01)

    patient_df['High_Risk'] = patient_df['risk_score'] >= threshold

    st.dataframe(patient_df.sort_values('risk_score', ascending=False))

    # Optional: download CSV
    csv = patient_df.to_csv(index=False).encode()
    st.download_button("Download predictions as CSV", csv, "risk_scores.csv", "text/csv")
