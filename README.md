# Streamlit Deterioration Prediction App

This application is designed as an interactive tool to visualize patient risk scores, leveraging a LightGBM model to predict potential deteriorations. It allows clinicians to identify high-risk patients quickly and supports data-driven prioritization of resources.

## Patient Deterioration Prediction: Detailed Summary

### Problem Framing & Clinical Value

**Goal:** Predict the probability that a patient’s condition will deteriorate within 90 days, using prior patient observations, lab results, and medication data.

**Clinical value:**

*   Early warning system for high-risk patients.
*   Enables clinicians to proactively intervene before critical deterioration occurs.
*   Helps prioritize patient monitoring, hospital resources, and personalized care planning.

**Prediction target:**

*   Binary label `deterioration_90d` (1 = deterioration within 90 days, 0 = stable).
*   Deterioration was defined using clinically relevant changes in DALY, QALY, and QOLS.

## Installation

1.  Ensure you have Python installed (version 3.7 or higher is recommended).
2.  Install the required Python packages. It is recommended to create a virtual environment first.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Input Data:**

    *   The app requires a CSV file as input. You can use the provided `patient_day_df_small.csv` file for testing.
    *   Upload the CSV file using the file uploader in the app.

3.  **Threshold Adjustment:**

    *   A slider allows you to adjust the prediction threshold.
    *   The threshold affects the trade-off between precision and recall:

        *   **t=0.1:** High recall (77%), low precision (48%). Catches most deteriorations but with more false alarms.
        *   **t=0.2:** Balanced precision (70%), recall (60%), F1-score (0.65). A good compromise.
        *   **t=0.3:** High precision (84%), lower recall (50%). Safer predictions, but misses some deteriorations.
        *   **t=0.5+:** Excellent precision (96-99%), very low recall (20-36%). Very conservative predictions.

4.  **Model:**

    *   The app uses the `lgb_deterioration_model.txt` LightGBM model. Ensure this file is in the same directory as `streamlit_app.py` or provide the correct path.

### Data Handling & Feature Strategy

**Data sources:**

*   Patient observations (DALY, QALY, QOLS)
*   Medications (daily flags for each drug)
*   Conditions (diagnosis codes)
*   Demographics (age, sex, race, income, etc.)
*   Encounters (hospital visits, start/end dates, encounter class)

**Processing steps:**

*   Daily expansion: Medication and condition records were expanded to a daily timeline for each patient.
*   Wide format: Binary flags per medication/condition were pivoted to a wide format.
*   Merging: Observations, medications, conditions, and demographics merged into a single patient-day dataframe.

**Feature engineering:**

*   Lag features (1, 7, 30 days) for observations and labs
*   Rolling means (7, 30 days) to capture short- and medium-term trends
*   Time features: day-of-week, month, weekend indicator

**Target creation:**

*   Label per patient-day if deterioration occurs within the next 90 days.

**Final dataset:**

*   ~680k rows, 60+ features.
*   Highly imbalanced: most patient-days are stable.

### Model Explanation & Evaluation

**Model:** LightGBM (Gradient Boosted Trees)

*   Chosen for speed, handling large feature sets, and ability to capture non-linear interactions.

**Training:**

*   Train/validation split by patients to avoid leakage.
*   Binary classification: predict probability of deterioration within 90 days.
*   Feature set included: observations, lag features, rolling means, medication flags, demographics, time features.

**Evaluation metrics:**

*   AUROC (Area Under ROC Curve): ~0.99 on imbalanced dataset
*   AUPRC (Area Under Precision-Recall Curve): ~0.73
*   Confusion matrix & thresholds: Precision, recall, F1-score calculated for multiple thresholds (0.1–0.9)
*   Calibration plots: Checked whether predicted probabilities align with actual risks.

**Threshold trade-offs:**

*   Low threshold → high recall, more false positives
*   High threshold → high precision, fewer false positives, but more false negatives

**Key insights:**

*   Model is very strong at distinguishing high-risk vs low-risk patient-days.
*   Deterioration mostly driven by recent changes in DALY/QALY, rolling trends, and some medications.

### Dashboard Demo

**Purpose:** Interactive tool to visualize risk scores across patients.

**Implementation:**

*   Streamlit or Gradio interface
*   Upload patient-day CSV (smaller subset for local testing)
*   Display predicted probabilities per patient or cohort
*   Optional filtering by age, medication, or encounter type
*   Allows clinicians to quickly identify high-risk patients.

**Example features shown:**

*   Patient ID, date, predicted probability
*   Observed values (DALY, QALY, QOLS)
*   Highlight top 5–10 high-risk patients

### Impact, Limitations, Next Steps

**Impact:**

*   Clinicians can intervene early
*   Data-driven prioritization of resources
*   Supports preventive care and personalized treatment

**Limitations:**

*   Imbalanced dataset → careful threshold selection needed
*   Model trained on historical data; real-world drift may affect performance
*   Medication adherence, lifestyle factors, and social determinants may not be fully captured

**Next steps:**

*   Deploy to clinical setting with real-time updates
*   Collect feedback from clinicians and refine thresholds
*   Expand feature set to include labs, vitals, and free-text notes


## Additional Notes

*   The app displays the predictions based on the selected threshold.
*   Experiment with different thresholds to find the optimal balance between precision and