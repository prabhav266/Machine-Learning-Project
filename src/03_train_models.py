
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
OUTPUT_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load the fixed training and test sets
# --------------------------------------------------

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test features do not match.")

if set(y_train.unique()) != set(y_test.unique()):
    raise ValueError("Training and test classes do not match.")

classes = sorted(y_train.unique())

print("=" * 65)
print("BEARING FAULT CLASSIFICATION — MODEL TRAINING")
print("=" * 65)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Features:         {X_train.shape[1]}")
print(f"Classes:          {classes}")

# --------------------------------------------------
# 3. Define classifiers
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    ),
}

# --------------------------------------------------
# 4. Train and evaluate
# --------------------------------------------------

results = []
reports = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    results.append(metrics)

    reports[name] = classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=classes,
        zero_division=0,
    )

    # Save each fitted model
    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, OUTPUT_DIR / f"{filename}.joblib")

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=classes)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"confusion_matrix_{filename}.png",
        dpi=200,
    )
    plt.close()

    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Macro F1: {metrics['Macro F1']:.4f}")

# --------------------------------------------------
# 5. Save comparison and reports
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "Macro F1", ascending=False
)

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

with open(
    OUTPUT_DIR / "classification_reports.txt",
    "w",
    encoding="utf-8",
) as file:
    for name, report in reports.items():
        file.write(f"\n{'=' * 65}\n{name}\n{'=' * 65}\n")
        file.write(report + "\n")

# --------------------------------------------------
# 6. Plot model comparison
# --------------------------------------------------

plot_df = results_df.set_index("Model")[
    ["Accuracy", "Macro F1", "Weighted F1"]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=20,
)

ax.set_title("Bearing Fault Classification — Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "model_comparison.png",
    dpi=200,
)
plt.close()

# --------------------------------------------------
# 7. Plot Random Forest feature importance
# --------------------------------------------------

rf = models["Random Forest"]

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)

importance.to_csv(OUTPUT_DIR / "feature_importance.csv")

plt.figure(figsize=(10, 8))
importance.head(15).sort_values().plot(kind="barh")
plt.title("Random Forest — Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "feature_importance.png",
    dpi=200,
)
plt.close()

# --------------------------------------------------
# 8. Save the best model by macro F1
# --------------------------------------------------

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

metadata = {
    "best_model": best_name,
    "target": TARGET,
    "features": list(X_train.columns),
    "classes": classes,
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "selection_metric": "Macro F1",
    "evaluation": "Stratified segment-level random split",
}

joblib.dump(metadata, OUTPUT_DIR / "model_metadata.joblib")

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSelected model: {best_name}")
print("Selection criterion: highest test-set macro F1")
print("\nSaved outputs:")
print(f" - {OUTPUT_DIR / 'model_comparison.csv'}")
print(f" - {OUTPUT_DIR / 'classification_reports.txt'}")
print(f" - {OUTPUT_DIR / 'best_model.joblib'}")
print(f" - {FIGURE_DIR / 'model_comparison.png'}")
print(f" - {FIGURE_DIR / 'feature_importance.png'}")
print("\nStep 3 complete.")
