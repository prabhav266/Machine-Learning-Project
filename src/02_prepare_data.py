import numpy as np
import pandas as pd
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "features_all_segments.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "splits"

TARGET = "label"
TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if df[TARGET].isna().any():
    raise ValueError("Missing target labels found.")


feature_columns = [col for col in df.columns if col != TARGET]

# Validate that all input features are numeric
if not all(
    pd.api.types.is_numeric_dtype(df[col])
    for col in feature_columns
):
    raise ValueError("All input features must be numeric.")

# Define X and y BEFORE validating feature values
X = df[feature_columns]
y = df[TARGET]

# Check for missing values
if X.isna().any().any():
    raise ValueError("Missing input feature values found.")

# Check for infinite values
if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite feature values found.")


# --------------------------------------------------
# 3. Stratified train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# --------------------------------------------------
# 4. Save the splits
# --------------------------------------------------

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# --------------------------------------------------
# 5. Generate a split report
# --------------------------------------------------

lines = [
    "BEARING FAULT CLASSIFICATION — DATA SPLIT REPORT",
    "=" * 55,
    f"Original dataset: {len(df)} rows",
    f"Input features: {len(feature_columns)}",
    f"Training rows: {len(train_df)}",
    f"Testing rows: {len(test_df)}",
    f"Test fraction: {TEST_SIZE:.0%}",
    f"Random state: {RANDOM_STATE}",
    "Split method: Stratified random split",
    "",
    "TRAINING CLASS DISTRIBUTION",
    "-" * 35,
    train_df[TARGET].value_counts().sort_index().to_string(),
    "",
    "TEST CLASS DISTRIBUTION",
    "-" * 35,
    test_df[TARGET].value_counts().sort_index().to_string(),
    "",
    "IMPORTANT LIMITATION",
    "-" * 35,
    "The dataset contains no recording/group identifiers.",
    "This is a segment-level random-split evaluation.",
    "Generalization to unseen recordings is not established.",
]

report = "\n".join(lines)
(OUTPUT_DIR / "split_report.txt").write_text(
    report, encoding="utf-8"
)

print(report)
print("\nSaved files:")
print(f" - {OUTPUT_DIR / 'train.csv'}")
print(f" - {OUTPUT_DIR / 'test.csv'}")
print(f" - {OUTPUT_DIR / 'split_report.txt'}")
print("\nStep 2 complete. No models have been trained.")
