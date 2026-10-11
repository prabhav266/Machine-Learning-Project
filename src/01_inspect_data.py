from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "features_all_segments.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

EXPECTED_FEATURES = [
    "rms", "std", "kurtosis", "skewness", "peak", "p2p", "crest", "shape", "energy",
    "spec_cent", "spec_bw", "spec_ent", "band0", "band1", "band2", "band3",
    "wE0", "wS0", "wE1", "wS1", "wE2", "wS2", "wE3", "wS3", "wE4", "wS4",
]
EXPECTED_CLASSES = ["Normal", "Ball_Fault", "Inner_Race", "Outer_Race"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place features_all_segments.csv in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    print("=" * 68)
    print("BEARING FAULT DATASET — INITIAL VALIDATION")
    print("=" * 68)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing cells: {int(df.isna().sum().sum())}")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    if "label" not in df.columns:
        raise ValueError("Expected target column 'label' was not found.")

    feature_columns = [column for column in df.columns if column != "label"]
    missing_features = sorted(set(EXPECTED_FEATURES) - set(feature_columns))
    unexpected_features = sorted(set(feature_columns) - set(EXPECTED_FEATURES))
    if missing_features or unexpected_features:
        print(f"Missing expected features: {missing_features or 'None'}")
        print(f"Unexpected features: {unexpected_features or 'None'}")
        raise ValueError("Feature columns do not match the documented 26-feature schema.")

    non_numeric = df[feature_columns].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric}")

    class_counts = df["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    class_counts["percentage"] = (class_counts["count"] / len(df) * 100).round(2)
    class_counts.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    df[feature_columns].describe().T.to_csv(OUTPUT_DIR / "feature_summary.csv")

    report_lines = [
        "Bearing fault dataset validation report",
        f"Dataset file: {DATA_PATH.name}",
        f"Rows: {len(df)}",
        f"Columns including target: {df.shape[1]}",
        f"Numerical features: {len(feature_columns)}",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Exact duplicate rows: {int(df.duplicated().sum())}",
        f"Labels observed: {sorted(df['label'].dropna().unique().tolist())}",
        "",
        "Class distribution:",
        class_counts.to_string(index=False),
        "",
        "Note: this file contains extracted features, not raw vibration waveforms.",
        "Check original recording/run identifiers before choosing the final validation split.",
    ]
    (OUTPUT_DIR / "dataset_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5.5))
    order = class_counts.sort_values("count", ascending=False)["label"].tolist()
    counts = df["label"].value_counts().reindex(order)
    ax = counts.plot(kind="bar", ax=plt.gca())
    ax.set_title("Bearing Condition Class Distribution")
    ax.set_xlabel("Bearing condition")
    ax.set_ylabel("Number of segments")
    plt.xticks(rotation=15, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 11))
    correlation = df[feature_columns].corr(numeric_only=True)
    sns.heatmap(correlation, cmap="coolwarm", center=0, linewidths=0.0, cbar_kws={"shrink": 0.75})
    plt.title("Correlation Heatmap of the 26 Input Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("\nClass distribution:")
    print(class_counts.to_string(index=False))
    print("\nCreated outputs:")
    for path in [
        OUTPUT_DIR / "dataset_report.txt",
        OUTPUT_DIR / "class_distribution.csv",
        OUTPUT_DIR / "feature_summary.csv",
        FIGURES_DIR / "class_distribution.png",
        FIGURES_DIR / "feature_correlation_heatmap.png",
    ]:
        print(f" - {path.relative_to(ROOT)}")
    print("\nStep 1 complete. Do not train models until validation strategy is decided.")


if __name__ == "__main__":
    main()
