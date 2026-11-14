"""
Guardian - Final Model Comparison
---------------------------------
Creates a clean comparison of all trained models using the final
test results already obtained.

This script:
- DOES NOT train any model.
- DOES NOT modify model weights.
- DOES NOT change thresholds.
- Creates a CSV and text report.

Run:
    python final_model_comparison.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

OUTPUT_CSV = RESULTS / "final_model_comparison.csv"
OUTPUT_TXT = RESULTS / "final_model_comparison.txt"

data = [
    ["Random Forest", 77.11, 59.30, 71.11, 64.67, None],
    ["MLP", 80.90, 66.93, 69.53, 68.21, None],
    ["CNN", 71.74, 51.31, 80.11, 62.56, None],
    ["GRU", 81.04, 70.83, 60.63, 65.33, None],
    ["Improved BiGRU", 82.99, 76.35, 61.24, 67.97, 7.93],
    ["CNN-GRU", 69.61, 49.08, 83.35, 61.78, None],
    ["Bidirectional LSTM", 81.58, 72.69, 60.05, 65.76, 9.43],
]

df = pd.DataFrame(
    data,
    columns=[
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "False Alarm (%)",
    ],
)

df["Rank (F1)"] = df["F1 (%)"].rank(
    ascending=False,
    method="min"
).astype(int)

df = df.sort_values(
    ["Rank (F1)", "Model"]
).reset_index(drop=True)

df = df[
    [
        "Rank (F1)",
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "False Alarm (%)",
    ]
]

RESULTS.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("=" * 95)
print("GUARDIAN - FINAL MODEL COMPARISON")
print("=" * 95)
print()
print(df.to_string(index=False))
print()

best_f1 = df.iloc[0]
best_accuracy = df.loc[df["Accuracy (%)"].idxmax()]
best_precision = df.loc[df["Precision (%)"].idxmax()]
best_recall = df.loc[df["Recall (%)"].idxmax()]

print("=" * 95)
print("HIGHLIGHTS")
print("=" * 95)

print(
    f"Highest F1        : {best_f1['Model']} "
    f"({best_f1['F1 (%)']:.2f}%)"
)

print(
    f"Highest Accuracy  : {best_accuracy['Model']} "
    f"({best_accuracy['Accuracy (%)']:.2f}%)"
)

print(
    f"Highest Precision : {best_precision['Model']} "
    f"({best_precision['Precision (%)']:.2f}%)"
)

print(
    f"Highest Recall    : {best_recall['Model']} "
    f"({best_recall['Recall (%)']:.2f}%)"
)

print()
print(
    "NOTE: False-alarm values are shown only where they were "
    "explicitly calculated in the corresponding final evaluation."
)

print(
    "BiGRU and LSTM used validation-selected thresholds before "
    "their final test evaluations."
)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("GUARDIAN - FINAL MODEL COMPARISON\n")
    f.write("=" * 70 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")
    f.write("HIGHLIGHTS\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Highest F1: {best_f1['Model']} "
        f"({best_f1['F1 (%)']:.2f}%)\n"
    )
    f.write(
        f"Highest Accuracy: {best_accuracy['Model']} "
        f"({best_accuracy['Accuracy (%)']:.2f}%)\n"
    )
    f.write(
        f"Highest Precision: {best_precision['Model']} "
        f"({best_precision['Precision (%)']:.2f}%)\n"
    )
    f.write(
        f"Highest Recall: {best_recall['Model']} "
        f"({best_recall['Recall (%)']:.2f}%)\n"
    )
    f.write("\n")
    f.write(
        "This report consolidates existing experiment results; "
        "it does not train or modify models.\n"
    )

print()
print("Saved:")
print(f"  {OUTPUT_CSV}")
print(f"  {OUTPUT_TXT}")
