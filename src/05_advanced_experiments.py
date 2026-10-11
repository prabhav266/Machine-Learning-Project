
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"
ADV_DIR = ROOT / "outputs" / "advanced"

for folder in [ADV_DIR, FIGURE_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "label"

# --------------------------------------------------
# 2. Load the existing fixed split
# --------------------------------------------------

train = pd.read_csv(SPLIT_DIR / "train.csv")
test = pd.read_csv(SPLIT_DIR / "test.csv")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test feature columns differ.")

print("=" * 70)
print("ADVANCED BEARING FAULT CLASSIFICATION EXPERIMENTS")
print("=" * 70)

# --------------------------------------------------
# EXPERIMENT A: PCA visualization
# --------------------------------------------------
# Scaling is important because features have different magnitudes.
# PCA is used here for visualization, not as the final classifier.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Bearing Condition": y_train.to_numpy(),
})

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Bearing Condition",
    alpha=0.65,
    s=25,
)
plt.title("PCA Visualization of Bearing Vibration Features")
plt.xlabel(
    f"Principal Component 1 "
    f"({pca.explained_variance_ratio_[0] * 100:.2f}% variance)"
)
plt.ylabel(
    f"Principal Component 2 "
    f"({pca.explained_variance_ratio_[1] * 100:.2f}% variance)"
)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "pca_bearing_classes.png", dpi=200)
plt.close()

pca_info = {
    "PC1_explained_variance_percent": float(
        pca.explained_variance_ratio_[0] * 100
    ),
    "PC2_explained_variance_percent": float(
        pca.explained_variance_ratio_[1] * 100
    ),
    "combined_explained_variance_percent": float(
        pca.explained_variance_ratio_.sum() * 100
    ),
}

with open(ADV_DIR / "pca_summary.json", "w", encoding="utf-8") as f:
    json.dump(pca_info, f, indent=2)

print("\nEXPERIMENT A — PCA")
print(f"PC1 variance: {pca_info['PC1_explained_variance_percent']:.2f}%")
print(f"PC2 variance: {pca_info['PC2_explained_variance_percent']:.2f}%")
print(
    "Combined variance: "
    f"{pca_info['combined_explained_variance_percent']:.2f}%"
)
print(f"Saved: {FIGURE_DIR / 'pca_bearing_classes.png'}")

# --------------------------------------------------
# EXPERIMENT B: Feature selection
# --------------------------------------------------
# Fit the selector only on training data.
# Median importance is used as a simple feature-selection threshold.

selector = SelectFromModel(
    estimator=RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    ),
    threshold="median",
)

selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.get_support()].tolist()

if not selected_features:
    raise RuntimeError("Feature selection returned no features.")

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Evaluate a selected-feature Random Forest with training CV.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

selected_rf = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    class_weight="balanced",
    n_jobs=-1,
)

selected_cv_scores = cross_val_score(
    selected_rf,
    X_train_selected,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=1,
)

selected_rf.fit(X_train_selected, y_train)
selected_predictions = selected_rf.predict(X_test_selected)

feature_result = {
    "original_feature_count": int(X_train.shape[1]),
    "selected_feature_count": len(selected_features),
    "selected_features": selected_features,
    "cv_macro_f1_mean": float(selected_cv_scores.mean()),
    "cv_macro_f1_std": float(selected_cv_scores.std()),
    "test_accuracy": float(
        accuracy_score(y_test, selected_predictions)
    ),
    "test_macro_f1": float(
        f1_score(y_test, selected_predictions, average="macro")
    ),
}

with open(
    ADV_DIR / "feature_selection_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(feature_result, f, indent=2)

pd.DataFrame({
    "Selected Feature": selected_features
}).to_csv(ADV_DIR / "selected_features.csv", index=False)

print("\nEXPERIMENT B — FEATURE SELECTION")
print(
    f"Features retained: {len(selected_features)} "
    f"out of {X_train.shape[1]}"
)
print("Selected features:", selected_features)
print(f"CV macro F1: {selected_cv_scores.mean():.4f}")
print(f"Test accuracy: {feature_result['test_accuracy']:.4f}")
print(f"Test macro F1: {feature_result['test_macro_f1']:.4f}")

# --------------------------------------------------
# EXPERIMENT C: Random Forest hyperparameter tuning
# --------------------------------------------------
# Hyperparameters are selected using training data only.
# The held-out test set is evaluated after the search is complete.

rf = RandomForestClassifier(
    random_state=SEED,
    class_weight="balanced",
    n_jobs=-1,
)

parameter_distributions = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.7],
    "criterion": ["gini", "entropy"],
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=parameter_distributions,
    n_iter=12,
    scoring="f1_macro",
    cv=cv,
    random_state=SEED,
    n_jobs=-1,
    verbose=1,
    refit=True,
)

print("\nEXPERIMENT C — RANDOM FOREST HYPERPARAMETER TUNING")
print("Searching 12 parameter combinations with 5-fold CV...")
search.fit(X_train, y_train)

best_rf = search.best_estimator_
tuned_predictions = best_rf.predict(X_test)

tuning_result = {
    "best_parameters": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_),
    "test_accuracy": float(
        accuracy_score(y_test, tuned_predictions)
    ),
    "test_macro_f1": float(
        f1_score(y_test, tuned_predictions, average="macro")
    ),
}

with open(
    ADV_DIR / "tuning_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(tuning_result, f, indent=2)

pd.DataFrame(search.cv_results_).sort_values(
    "rank_test_score"
).to_csv(ADV_DIR / "random_forest_search_results.csv", index=False)

joblib.dump(best_rf, MODEL_DIR / "tuned_random_forest.joblib")

print("\nBest parameters:")
print(json.dumps(search.best_params_, indent=2))
print(f"Best CV macro F1: {search.best_score_:.4f}")
print(f"Tuned test accuracy: {tuning_result['test_accuracy']:.4f}")
print(f"Tuned test macro F1: {tuning_result['test_macro_f1']:.4f}")

print("\nTUNED MODEL CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    tuned_predictions,
    zero_division=0,
))

print("\nSaved advanced experiment results in:")
print(ADV_DIR)
print("\nStep 6 complete.")
