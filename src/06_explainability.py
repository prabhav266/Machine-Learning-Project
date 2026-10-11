
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "models" / "tuned_random_forest.joblib"
METADATA_PATH = ROOT / "outputs" / "models" / "model_metadata.joblib"


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(METADATA_PATH),
    )


def calculate_shap(model, X):
    # TreeExplainer is appropriate for Random Forest models.
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return explainer, values


def main():
    st.title("SHAP Explainability")
    st.caption("Understanding the features behind bearing-fault predictions")

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("Model artifacts not found.")
        return

    model, metadata = load_artifacts()
    features = metadata["features"]
    classes = list(metadata["classes"])

    uploaded = st.file_uploader(
        "Upload a CSV of extracted vibration features",
        type=["csv"],
        key="shap_upload",
    )

    if uploaded is None:
        st.info("Upload a feature CSV to generate SHAP explanations.")
        return

    try:
        df = pd.read_csv(uploaded)

        if "label" in df.columns:
            df = df.drop(columns=["label"])

        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error("Missing features: " + ", ".join(missing))
            return

        X = df[features].copy()
        for col in features:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any() or not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            st.error("Features contain missing or non-finite values.")
            return

        if len(X) > 1000:
            st.warning(
                "For performance, explanations use a reproducible sample "
                "of up to 1,000 rows."
            )
            X = X.sample(n=1000, random_state=42)

        with st.spinner("Calculating SHAP values..."):
            explainer, shap_values = calculate_shap(model, X)

        st.success(f"Explained {len(X)} feature rows.")

        # SHAP output shape differs between versions and model wrappers.
        # Convert it to a 3D array: rows × features × classes.
        if isinstance(shap_values, list):
            values_3d = np.stack(shap_values, axis=-1)
        else:
            values_3d = np.asarray(shap_values)

        if values_3d.ndim == 2:
            values_3d = values_3d[:, :, np.newaxis]

        if values_3d.shape[1] != len(features):
            st.error(
                "Unexpected SHAP output shape: "
                f"{values_3d.shape}. Cannot safely map values to features."
            )
            return

        # Global importance averages absolute SHAP values across
        # samples and output classes.
        global_importance = np.abs(values_3d).mean(axis=(0, 2))
        importance_df = pd.DataFrame({
            "Feature": features,
            "Mean_Absolute_SHAP": global_importance,
        }).sort_values("Mean_Absolute_SHAP", ascending=False)

        st.subheader("Global feature importance")
        st.caption(
            "Mean absolute SHAP value: larger values indicate greater "
            "influence on model output, not a causal effect."
        )
        st.dataframe(importance_df, use_container_width=True)

        top_n = min(15, len(importance_df))
        chart_df = importance_df.head(top_n).sort_values(
            "Mean_Absolute_SHAP"
        )
        st.bar_chart(chart_df.set_index("Feature"))

        st.subheader("Explain an individual prediction")
        row_index = st.number_input(
            "Segment row index",
            min_value=0,
            max_value=len(X) - 1,
            value=0,
            step=1,
        )

        row = X.iloc[[int(row_index)]]
        prediction = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        st.metric("Predicted condition", str(prediction))
        st.metric(
            "Maximum class probability",
            f"{probabilities.max() * 100:.2f}%",
        )

        predicted_class_index = list(model.classes_).index(prediction)
        row_values = values_3d[int(row_index), :, predicted_class_index]

        local_df = pd.DataFrame({
            "Feature": features,
            "Feature_Value": X.iloc[int(row_index)].to_numpy(),
            "SHAP_Value": row_values,
            "Absolute_SHAP": np.abs(row_values),
        }).sort_values("Absolute_SHAP", ascending=False)

        st.write(
            "Features with the largest absolute SHAP values for the "
            "predicted class:"
        )
        st.dataframe(local_df.head(15), use_container_width=True)

        st.download_button(
            "Download global SHAP importance",
            data=importance_df.to_csv(index=False).encode("utf-8"),
            file_name="shap_global_importance.csv",
            mime="text/csv",
        )

        st.caption(
            "SHAP explains the fitted model's output. It does not prove "
            "that a feature physically caused a bearing fault."
        )

    except Exception as exc:
        st.error(f"SHAP analysis failed: {exc}")
        st.caption(
            "If the error concerns SHAP output dimensions, share the "
            "complete error message so we can adapt to the installed version."
        )


if __name__ == "__main__":
    main()
