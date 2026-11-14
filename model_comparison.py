
"""
Guardian - Confusion Matrix + False Alarm Report
=================================================
This script does NOT retrain any model.

It:
1. Uses the already obtained test confusion matrices.
2. Calculates False Alarm Rate for every model.
3. Saves a corrected model-comparison CSV.
4. Creates one colorful confusion-matrix PNG per model.
5. Creates a multi-page PDF containing all confusion matrices.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "guardian_confusion_matrices"
OUT.mkdir(parents=True, exist_ok=True)

# Format:
# Model, Accuracy, Precision, Recall, F1, TN, FP, FN, TP, Threshold
models = [
    ["Random Forest", 77.11, 59.30, 71.11, 64.67, 12416, 3180, 1882, 4633, 0.50],
    ["MLP", 80.90, 66.93, 69.53, 68.21, 13358, 2238, 1985, 4530, 0.50],
    ["CNN", 71.74, 51.31, 80.11, 62.56, 10644, 4952, 1296, 5219, 0.50],
    ["GRU", 81.04, 70.83, 60.63, 65.33, 13969, 1627, 2565, 3950, 0.50],
    ["Improved BiGRU", 82.99, 76.35, 61.24, 67.97, 14360, 1236, 2525, 3990, 0.60],
    ["CNN-GRU", 69.61, 49.08, 83.35, 61.78, 9962, 5634, 1085, 5430, 0.50],
    ["Bidirectional LSTM", 81.58, 72.69, 60.05, 65.76, 14126, 1470, 2603, 3912, 0.50],
]

columns = [
    "Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)",
    "TN", "FP", "FN", "TP", "Threshold"
]

df = pd.DataFrame(models, columns=columns)

# Calculate false-alarm rate for EVERY model.
df["False Alarm (%)"] = df["FP"] / (df["TN"] + df["FP"]) * 100

# Rank by F1.
df["Rank (F1)"] = (
    df["F1 (%)"]
    .rank(ascending=False, method="min")
    .astype(int)
)

df = df.sort_values(["Rank (F1)", "Model"]).reset_index(drop=True)

# Save corrected comparison.
comparison_columns = [
    "Rank (F1)", "Model", "Accuracy (%)", "Precision (%)",
    "Recall (%)", "F1 (%)", "False Alarm (%)"
]
df[comparison_columns].to_csv(
    OUT / "final_model_comparison_corrected.csv",
    index=False
)

# Save detailed matrix data.
df.to_csv(OUT / "confusion_matrix_data.csv", index=False)

# Colorful confusion matrices.
# Each matrix is a separate figure, so it can be inserted into a report.
cmap = "Blues"

pdf_path = OUT / "Guardian_All_Confusion_Matrices.pdf"

with PdfPages(pdf_path) as pdf:
    for _, row in df.iterrows():

        model = row["Model"]
        matrix = [
            [int(row["TN"]), int(row["FP"])],
            [int(row["FN"]), int(row["TP"])]
        ]

        fig, ax = plt.subplots(figsize=(8.5, 7))

        image = ax.imshow(matrix, cmap=cmap)

        # Color bar.
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Number of Samples", fontsize=11)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Normal", "Predicted Fall"])
        ax.set_yticklabels(["Actual Normal", "Actual Fall"])

        ax.set_xlabel("Predicted Class", fontsize=12)
        ax.set_ylabel("Actual Class", fontsize=12)

        ax.set_title(
            f"{model} - Confusion Matrix\n"
            f"Accuracy: {row['Accuracy (%)']:.2f}% | "
            f"F1: {row['F1 (%)']:.2f}% | "
            f"False Alarm: {row['False Alarm (%)']:.2f}%",
            fontsize=14,
            fontweight="bold",
            pad=15
        )

        # Put values and labels inside cells.
        max_value = max(max(x) for x in matrix)

        for i in range(2):
            for j in range(2):
                value = matrix[i][j]
                label = [
                    ["TN", "FP"],
                    ["FN", "TP"]
                ][i][j]

                # White text on dark cells, black on light cells.
                text_color = "white" if value > max_value * 0.55 else "black"

                ax.text(
                    j, i,
                    f"{label}\n{value:,}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=text_color
                )

        # Add metric box below the matrix.
        metrics = (
            f"Precision: {row['Precision (%)']:.2f}%    "
            f"Recall: {row['Recall (%)']:.2f}%    "
            f"F1: {row['F1 (%)']:.2f}%"
        )

        fig.text(
            0.5, 0.04,
            metrics,
            ha="center",
            fontsize=11
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        # Safe filename.
        safe_name = (
            model.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        png_path = OUT / f"{safe_name}_confusion_matrix.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# Create a text summary for viva/report use.
best_f1 = df.iloc[0]
lowest_far = df.loc[df["False Alarm (%)"].idxmin()]
highest_recall = df.loc[df["Recall (%)"].idxmax()]

with open(OUT / "confusion_matrix_summary.txt", "w", encoding="utf-8") as f:
    f.write("GUARDIAN - CONFUSION MATRIX SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("CORRECTED FALSE ALARM RATES\n")
    f.write("-" * 70 + "\n")

    for _, row in df.iterrows():
        f.write(
            f"{row['Model']}: "
            f"False Alarm Rate = {row['False Alarm (%)']:.2f}%\n"
        )

    f.write("\n")
    f.write(
        f"Highest F1: {best_f1['Model']} "
        f"({best_f1['F1 (%)']:.2f}%)\n"
    )
    f.write(
        f"Lowest False Alarm Rate: {lowest_far['Model']} "
        f"({lowest_far['False Alarm (%)']:.2f}%)\n"
    )
    f.write(
        f"Highest Recall: {highest_recall['Model']} "
        f"({highest_recall['Recall (%)']:.2f}%)\n"
    )

print("=" * 80)
print("GUARDIAN - CORRECTED MODEL COMPARISON")
print("=" * 80)
print()
print(
    df[comparison_columns].to_string(
        index=False,
        formatters={
            "Accuracy (%)": "{:.2f}".format,
            "Precision (%)": "{:.2f}".format,
            "Recall (%)": "{:.2f}".format,
            "F1 (%)": "{:.2f}".format,
            "False Alarm (%)": "{:.2f}".format,
        }
    )
)
print()
print("Files created in:")
print(OUT)
print()
print("Main PDF:")
print(pdf_path)
    