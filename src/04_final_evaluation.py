
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
SEED = 42

# Load the previously fixed split
train = pd.read_csv(SPLIT_DIR / "train.csv")
test = pd.read_csv(SPLIT_DIR / "test.csv")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test features do not match.")

if set(y_train.unique()) != set(y_test.unique()):
    raise ValueError("Training and test classes do not match.")

classes = sorted(y_train.unique())

# Define the same four baseline model configurations
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=SEED,
    ),
}

# Model selection uses training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}

cv_results = []

print("=" * 65)
print("5-FOLD CROSS-VALIDATION ON TRAINING DATA")
print("=" * 65)

for name, model in models.items():
    print(f"\nEvaluating {name} ...")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    cv_results.append({
        "Model": name,
        "CV Accuracy Mean": scores["test_accuracy"].mean(),
        "CV Accuracy Std": scores["test_accuracy"].std(),
        "CV Macro F1 Mean": scores["test_macro_f1"].mean(),
        "CV Macro F1 Std": scores["test_macro_f1"].std(),
        "CV Weighted F1 Mean": scores["test_weighted_f1"].mean(),
    })

cv_df = pd.DataFrame(cv_results).sort_values(
    "CV Macro F1 Mean", ascending=False
)

cv_df.to_csv(MODEL_DIR / "cross_validation_comparison.csv", index=False)

print("\nCROSS-VALIDATION COMPARISON")
print(cv_df.to_string(
    index=False, float_format=lambda value: f"{value:.4f}"
))

# Choose using CV macro F1, not the test set
best_name = cv_df.iloc[0]["Model"]
best_model = models[best_name]

print(f"\nSelected using training CV macro F1: {best_name}")

# Fit on all training data; evaluate on held-out test data once
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

final_metrics = {
    "selected_model": best_name,
    "test_accuracy": float(accuracy_score(y_test, predictions)),
    "test_macro_precision": float(
        precision_score(y_test, predictions, average="macro", zero_division=0)
    ),
    "test_macro_recall": float(
        recall_score(y_test, predictions, average="macro", zero_division=0)
    ),
    "test_macro_f1": float(
        f1_score(y_test, predictions, average="macro", zero_division=0)
    ),
    "test_weighted_f1": float(
        f1_score(y_test, predictions, average="weighted", zero_division=0)
    ),
    "evaluation_method": "Stratified 80/20 segment-level split",
    "cv_folds": 5,
    "selection_metric": "Mean cross-validation macro F1",
    "random_state": SEED,
}

print("\nFINAL HELD-OUT TEST RESULTS")
print(json.dumps(final_metrics, indent=2))

report = classification_report(
    y_test,
    predictions,
    labels=classes,
    target_names=classes,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT")
print(report)

# Save final artifacts
joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

metadata = {
    "model_name": best_name,
    "features": list(X_train.columns),
    "classes": [str(c) for c in classes],
    "target": TARGET,
    "random_state": SEED,
    "selection_method": "5-fold stratified CV on training set",
    "evaluation_method": "Held-out stratified segment-level test set",
}

joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")

with open(
    MODEL_DIR / "final_metrics.json", "w", encoding="utf-8"
) as file:
    json.dump(final_metrics, file, indent=2)

with open(
    MODEL_DIR / "final_classification_report.txt",
    "w", encoding="utf-8"
) as file:
    file.write(report)

# Confusion matrix for the selected model
cm = confusion_matrix(y_test, predictions, labels=classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
plt.title(f"Final Test Confusion Matrix — {best_name}")
plt.xlabel("Predicted bearing condition")
plt.ylabel("Actual bearing condition")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "final_confusion_matrix.png", dpi=200)
plt.close()

# Feature importance when supported by the selected estimator
estimator = (
    best_model.named_steps["classifier"]
    if isinstance(best_model, Pipeline)
    else best_model
)

if hasattr(estimator, "feature_importances_"):
    importance = pd.Series(
        estimator.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    importance.to_csv(MODEL_DIR / "final_feature_importance.csv")

    plt.figure(figsize=(10, 8))
    importance.head(15).sort_values().plot(kind="barh")
    plt.title(f"Top 15 Features — {best_name}")
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_feature_importance.png", dpi=200)
    plt.close()

print("\nFinal model and evaluation artifacts saved.")
print("Step 4 complete.")
