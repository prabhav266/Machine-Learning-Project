
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "models"

MODEL_PATH = MODEL_DIR / "tuned_random_forest.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"

st.set_page_config(
    page_title="Bearing Fault Detection",
    page_icon="⚙️",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


st.title("⚙️ ML-Based Bearing Fault Detection")
st.caption(
    "2-HP Induction Motor | Vibration Feature-Based Classification"
)

st.write(
    "Upload a CSV containing the 26 extracted vibration features. "
    "The trained Random Forest model will predict the bearing "
    "condition for each segment."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files not found. Run "
        "`python src/04_final_evaluation.py` first."
    )
    st.stop()

model, metadata = load_artifacts()

feature_names = metadata["features"]
classes = metadata["classes"]

model_name = "Tuned Random Forest"
st.info(f"Loaded model: **{model_name}**")

with st.expander("About this project"):
    st.markdown(
        """
        **Objective:** Classify bearing conditions using numerical
        vibration features.

        **Input:** 26 time-domain, frequency-domain and wavelet features.

        **Output classes:**
        - Normal
        - Ball Fault
        - Inner Race Fault
        - Outer Race Fault

        **Dataset:** CWRU-based extracted feature dataset.

        **Evaluation:** Stratified segment-level train/test split.
        The reported evaluation does not establish performance on
        entirely unseen recordings or motors.
        """
    )

st.subheader("Upload vibration features")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=(
        "The CSV must contain all 26 required feature columns. "
        "An optional label column will be ignored."
    ),
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        if input_df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        if input_df.columns.duplicated().any():
            st.error("Duplicate column names were found.")
            st.stop()

        # Permit a label column in an evaluation CSV, but never
        # use it as a model input.
        if "label" in input_df.columns:
            input_df = input_df.drop(columns=["label"])

        missing = [
            feature for feature in feature_names
            if feature not in input_df.columns
        ]

        if missing:
            st.error(
                "Missing required features: "
                + ", ".join(missing)
            )
            st.stop()

        # Ignore extra columns and preserve training feature order.
        X = input_df[feature_names].copy()

        for col in feature_names:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            st.error(
                "Missing or non-numeric values were found "
                "in the required features. Correct the CSV "
                "and upload it again."
            )
            st.stop()

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            st.error(
                "Infinite or non-finite feature values were found."
            )
            st.stop()

        st.write("Input preview")
        st.dataframe(X.head(10), use_container_width=True)

        if st.button("Predict Bearing Condition", type="primary"):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            results = input_df.copy()
            results["Predicted_Condition"] = predictions
            results["Confidence_Percent"] = (
                probabilities.max(axis=1) * 100
            ).round(2)

            st.subheader("Prediction Results")

            st.subheader("Predictions by segment")
            selected_class = st.selectbox(
                "Filter by predicted condition",
                ["All classes"] + list(classes),
            )

            if selected_class != "All classes":
                display_results = results[
                    results["Predicted_Condition"] == selected_class
                ]
            else:
                display_results = results

            st.dataframe(
                display_results[
                    ["Predicted_Condition", "Confidence_Percent"]
                ],
                use_container_width=True,
            )

            with st.expander("View all input features and predictions"):
                st.dataframe(results, use_container_width=True)

            counts = (
                pd.Series(predictions, name="Predicted_Condition")
                .value_counts()
                .reindex(classes, fill_value=0)
            )

            
            left, right = st.columns(2)

            with left:
                st.metric("Segments analysed", len(predictions))

            with right:
                st.metric(
                    "Predicted classes",
                    int(pd.Series(predictions).nunique()),
                )

            st.subheader("Predicted condition distribution")

            # Display individual counts for all four classes
            cols = st.columns(4)

            for col, class_name in zip(cols, classes):
                col.metric(
                    class_name,
                    f"{counts[class_name]:,}",
                )

            st.bar_chart(counts)


            st.download_button(
                label="Download Predictions CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="bearing_fault_predictions.csv",
                mime="text/csv",
            )

    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not process the CSV: {exc}")

st.divider()
st.caption(
    "Academic prototype for bearing-condition classification. "
    "Predictions should not be used as the sole basis for "
    "industrial maintenance or safety decisions."
)
