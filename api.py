
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "models" / "tuned_random_forest.joblib"
METADATA_PATH = ROOT / "outputs" / "models" / "model_metadata.joblib"

app = FastAPI(
    title="Bearing Fault Intelligence API",
    description="API for vibration-feature-based bearing classification.",
    version="1.0.0",
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError("Model artifacts are missing from outputs/models.")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
FEATURES = list(metadata["features"])


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Mapping of all 26 feature names to numerical values.",
    )


class PredictionResponse(BaseModel):
    predicted_condition: str
    confidence_percent: float
    class_probabilities: dict[str, float]


@app.get("/")
def root():
    return {
        "name": "Bearing Fault Intelligence API",
        "status": "running",
        "model": "Tuned Random Forest",
        "required_features": len(FEATURES),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/features")
def get_features():
    return {"required_features": FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    missing = [name for name in FEATURES if name not in request.features]
    extra = [name for name in request.features if name not in FEATURES]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required features", "features": missing},
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail={"message": "Unexpected features", "features": extra},
        )

    try:
        row = pd.DataFrame(
            [[request.features[name] for name in FEATURES]],
            columns=FEATURES,
            dtype=float,
        )
        values = row.to_numpy()

        if not np.isfinite(values).all():
            raise ValueError("Feature values must be finite.")

        prediction = str(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        return PredictionResponse(
            predicted_condition=prediction,
            confidence_percent=round(float(probabilities.max() * 100), 2),
            class_probabilities={
                str(label): round(float(probability * 100), 2)
                for label, probability in zip(model.classes_, probabilities)
            },
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
