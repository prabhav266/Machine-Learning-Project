
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"

MODEL_PATH = MODEL_DIR / "tuned_random_forest.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"
METRICS_PATH = MODEL_DIR / "final_metrics.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
IMPORTANCE_PATH = MODEL_DIR / "final_feature_importance.csv"

st.set_page_config(
    page_title="Bearing Fault Intelligence",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_pdf(results, model_name):
    """Create a downloadable summary report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Bearing Fault Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph("ML-Based Bearing Fault Detection", styles["Heading2"]),
        Paragraph(f"Model: {model_name}", styles["Normal"]),
        Paragraph(
            f"Segments analysed: {len(results)}", styles["Normal"]
        ),
        Spacer(1, 12),
        Paragraph("Predicted condition summary", styles["Heading2"]),
    ]

    summary = results["Predicted_Condition"].value_counts()
    table_data = [["Predicted condition", "Segments"]]
    table_data.extend(
        [[str(label), int(count)] for label, count in summary.items()]
    )
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 12)])

    avg_conf = results["Confidence_Percent"].mean()
    story.append(Paragraph(
        f"Mean maximum predicted class probability: {avg_conf:.2f}%.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This is an academic prototype. Predicted probabilities are not "
        "guarantees of correctness. Results are not a substitute for "
        "industrial inspection or safety procedures.",
        styles["BodyText"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.title("⚙️ Bearing Fault Intelligence")
st.caption(
    "2-HP induction motor | Vibration-feature-based fault classification"
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model artifacts are missing. Check outputs/models and run "
        "the training/evaluation scripts if necessary."
    )
    st.stop()

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

feature_names = metadata["features"]
classes = list(metadata["classes"])
model_name = "Tuned Random Forest"

metrics = {}
if METRICS_PATH.exists():
    import json
    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

comparison = load_csv(str(COMPARISON_PATH))
importance = load_csv(str(IMPORTANCE_PATH))

with st.sidebar:
    st.title("Navigation")
    st.caption("Bearing Fault Intelligence System")
    st.markdown("---")
    st.write("**Current model**")
    st.write(model_name)
    st.write(f"**Required features:** {len(feature_names)}")
    st.markdown("---")
    st.caption("Academic prototype — not safety-certified.")

overview_tab, predict_tab, insights_tab, report_tab = st.tabs([
    "Overview",
    "Predict",
    "Model Insights",
    "Reports",
])

with overview_tab:
    st.subheader("System overview")
    st.write(
        "Classify extracted vibration features into four bearing-condition "
        "classes and review model evaluation results."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Held-out test accuracy",
        f"{metrics['test_accuracy'] * 100:.2f}%"
        if "test_accuracy" in metrics else "N/A",
    )
    c2.metric(
        "Held-out macro F1",
        f"{metrics['test_macro_f1'] * 100:.2f}%"
        if "test_macro_f1" in metrics else "N/A",
    )
    c3.metric("Input features", len(feature_names))

    st.info(
        "Evaluation uses a stratified segment-level split. It does not "
        "establish generalization to unseen recordings or motors."
    )

    st.markdown("#### Bearing conditions")
    st.write(", ".join(map(str, classes)))

    class_image = FIGURE_DIR / "class_distribution.png"
    if class_image.exists():
        st.image(str(class_image), caption="Dataset class distribution")

    st.markdown("#### How to use")
    st.markdown(
        """
        1. Open **Predict**.
        2. Upload a CSV containing all required numerical features.
        3. Run predictions and review the class distribution.
        4. Download the prediction CSV or a PDF summary from **Reports**.
        """
    )

with predict_tab:
    st.subheader("Predict bearing condition")
    st.write(
        "Upload extracted vibration features. The optional `label` column "
        "is excluded from model inputs."
    )

    uploaded_file = st.file_uploader(
        "Upload feature CSV",
        type=["csv"],
        key="prediction_upload",
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

            if "label" in input_df.columns:
                input_df = input_df.drop(columns=["label"])

            missing = [
                feature for feature in feature_names
                if feature not in input_df.columns
            ]

            if missing:
                st.error("Missing features: " + ", ".join(missing))
                st.stop()

            X = input_df[feature_names].copy()
            for col in feature_names:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            if X.isna().any().any():
                st.error("Missing or non-numeric feature values detected.")
                st.stop()

            if not np.isfinite(X.to_numpy(dtype=float)).all():
                st.error("Infinite or non-finite values detected.")
                st.stop()

            st.markdown("#### Input preview")
            st.dataframe(X.head(10), use_container_width=True)

            if st.button(
                "Predict Bearing Condition",
                type="primary",
                key="run_predictions",
            ):
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                results = input_df.copy()
                results["Predicted_Condition"] = predictions
                results["Confidence_Percent"] = (
                    probabilities.max(axis=1) * 100
                ).round(2)

                # Save results in the session for the Reports tab.
                st.session_state["bearing_results"] = results
                st.session_state["bearing_probabilities"] = probabilities

            results = st.session_state.get("bearing_results")

            if results is not None:
                st.markdown("#### Prediction summary")

                a, b, c = st.columns(3)
                a.metric("Segments analysed", f"{len(results):,}")
                b.metric(
                    "Classes predicted",
                    results["Predicted_Condition"].nunique(),
                )
                c.metric(
                    "Mean maximum class probability",
                    f"{results['Confidence_Percent'].mean():.2f}%",
                )

                selected_class = st.selectbox(
                    "Filter predictions",
                    ["All classes"] + [str(c) for c in classes],
                )

                filtered = results
                if selected_class != "All classes":
                    filtered = results[
                        results["Predicted_Condition"] == selected_class
                    ]

                st.dataframe(filtered, use_container_width=True)

                counts = (
                    results["Predicted_Condition"]
                    .value_counts()
                    .reindex(classes, fill_value=0)
                )
                st.markdown("#### Predicted condition distribution")
                st.bar_chart(counts)

                st.download_button(
                    "Download prediction CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="bearing_fault_predictions.csv",
                    mime="text/csv",
                )

                st.caption(
                    "Confidence is the maximum predicted class probability, "
                    "not a guarantee that the prediction is correct."
                )

        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not process CSV: {exc}")

with insights_tab:
    st.subheader("Model evaluation and feature analytics")

    if not comparison.empty:
        st.markdown("#### Model comparison")
        st.dataframe(comparison, use_container_width=True)

        metric_options = [
            col for col in comparison.columns
            if col.lower() in {
                "accuracy", "macro precision", "macro recall",
                "macro f1", "weighted f1"
            }
        ]
        if "Model" in comparison.columns and metric_options:
            chart_df = comparison.set_index("Model")[metric_options]
            st.bar_chart(chart_df)

    else:
        st.warning("Model comparison CSV was not found.")

    st.markdown("#### Evaluation plots")
    plot_names = [
        ("final_confusion_matrix.png", "Final confusion matrix"),
        ("final_feature_importance.png", "Random Forest feature importance"),
        ("pca_bearing_classes.png", "PCA class visualization"),
        ("feature_correlation_heatmap.png", "Feature correlation heatmap"),
    ]

    available_plots = [
        (name, caption) for name, caption in plot_names
        if (FIGURE_DIR / name).exists()
    ]

    for start in range(0, len(available_plots), 2):
        cols = st.columns(2)
        for col, (name, caption) in zip(
            cols, available_plots[start:start + 2]
        ):
            with col:
                st.image(str(FIGURE_DIR / name), caption=caption)

    if not importance.empty:
        st.markdown("#### Feature importance data")
        st.dataframe(importance, use_container_width=True)

with report_tab:
    st.subheader("Export analysis")
    results = st.session_state.get("bearing_results")

    if results is None:
        st.info(
            "Run a prediction in the Predict tab first. The generated "
            "results will then be available here."
        )
    else:
        st.write(f"Report contains {len(results):,} analysed segments.")

        st.download_button(
            "Download results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="bearing_fault_predictions.csv",
            mime="text/csv",
            key="report_csv",
        )

        try:
            pdf_bytes = build_pdf(results, model_name)
            st.download_button(
                "Download PDF summary",
                data=pdf_bytes,
                file_name="bearing_fault_report.pdf",
                mime="application/pdf",
                key="report_pdf",
            )
        except ImportError:
            st.error(
                "PDF dependency is unavailable. Install reportlab in "
                "the active virtual environment."
            )

st.divider()
st.caption(
    "Bearing Fault Intelligence | Academic prototype. Do not use "
    "predictions as the sole basis for industrial maintenance or safety "
    "decisions."
)
