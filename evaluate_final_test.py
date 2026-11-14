"""
Guardian - Final Test Evaluation
--------------------------------
Final evaluation of the selected Improved BiGRU model.

IMPORTANT:
- The model and threshold were selected using the validation set.
- The TEST set is used only once here for final unbiased evaluation.
- Do NOT change the threshold after seeing these results.

Selected configuration:
    Improved BiGRU
    Probability threshold = 0.60

Run from the project root:
    python evaluate_final_test.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "processed_data" / "cnn"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODEL_DIR / "fall_detection_gru_improved_best.pth"

X_TEST_PATH = DATA_DIR / "X_test_raw.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"

MEAN_PATH = DATA_DIR / "gru_improved_channel_mean.npy"
STD_PATH = DATA_DIR / "gru_improved_channel_std.npy"

BATCH_SIZE = 512

# FINAL threshold selected from validation.
THRESHOLD = 0.60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMPROVED BIDIRECTIONAL GRU
# ============================================================

class ImprovedBiGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.classifier(out).squeeze(1)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = ImprovedBiGRU().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(model, X, mean, std):

    X = X.astype(np.float32)

    # Same normalization used during validation.
    X = (
        X - mean.reshape(1, 1, -1)
    ) / std.reshape(1, 1, -1)

    probabilities = []

    for start in range(0, len(X), BATCH_SIZE):

        batch = torch.from_numpy(
            X[start:start + BATCH_SIZE]
        ).to(DEVICE)

        logits = model(batch)

        probs = torch.sigmoid(logits)

        probabilities.append(
            probs.cpu().numpy()
        )

    return np.concatenate(probabilities)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GUARDIAN - FINAL TEST EVALUATION")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        MEAN_PATH,
        STD_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load TEST data
    # --------------------------------------------------------

    print("\nLoading TEST data...")

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    print(
        f"Normal samples: {np.sum(y_test == 0)}"
    )

    print(
        f"Fall samples: {np.sum(y_test == 1)}"
    )

    # --------------------------------------------------------
    # Load normalization
    # --------------------------------------------------------

    mean = np.load(MEAN_PATH).astype(
        np.float32
    )

    std = np.load(STD_PATH).astype(
        np.float32
    )

    # Protect against zero standard deviation.
    std = np.where(
        std == 0,
        1.0,
        std
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading final Improved BiGRU...")

    model = load_model()

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating TEST predictions..."
    )

    probabilities = predict(
        model,
        X_test,
        mean,
        std
    )

    # --------------------------------------------------------
    # Fixed threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    print(
        f"\nFINAL threshold: {THRESHOLD}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(
        f"\nAccuracy       : {accuracy * 100:.2f}%"
    )

    print(
        f"Fall Precision : {precision * 100:.2f}%"
    )

    print(
        f"Fall Recall    : {recall * 100:.2f}%"
    )

    print(
        f"Fall F1-score  : {f1 * 100:.2f}%"
    )

    print(
        f"False Alarm Rate: {false_alarm_rate * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(
        "                 Predicted"
    )
    print(
        "                 Normal   Fall"
    )
    print(
        f"Actual Normal    {tn:6d}   {fp:6d}"
    )
    print(
        f"Actual Fall      {fn:6d}   {tp:6d}"
    )

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Fall"
            ],
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save probabilities and predictions
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame({
        "actual_label": y_test,
        "fall_probability": probabilities,
        "predicted_label": predictions,
    })

    prediction_path = (
        RESULTS_DIR /
        "final_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # Save final report
    # --------------------------------------------------------

    report_path = (
        RESULTS_DIR /
        "final_test_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "GUARDIAN - FINAL TEST EVALUATION\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Model: Improved Bidirectional GRU\n"
        )

        f.write(
            "Dataset: SisFall\n"
        )

        f.write(
            "Evaluation split: TEST\n"
        )

        f.write(
            "Probability threshold: 0.60\n"
        )

        f.write(
            "Threshold selected using validation set only.\n\n"
        )

        f.write(
            f"Test samples: {len(y_test)}\n"
        )

        f.write(
            f"Normal samples: {np.sum(y_test == 0)}\n"
        )

        f.write(
            f"Fall samples: {np.sum(y_test == 1)}\n\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Fall Precision: {precision * 100:.2f}%\n"
        )

        f.write(
            f"Fall Recall: {recall * 100:.2f}%\n"
        )

        f.write(
            f"Fall F1-score: {f1 * 100:.2f}%\n"
        )

        f.write(
            f"False Alarm Rate: "
            f"{false_alarm_rate * 100:.2f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            f"TN = {tn}\n"
        )

        f.write(
            f"FP = {fp}\n"
        )

        f.write(
            f"FN = {fn}\n"
        )

        f.write(
            f"TP = {tp}\n"
        )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"\n{report_path}"
    )

    print(
        f"{prediction_path}"
    )

    print("\nThis is the FINAL unbiased test evaluation.")
    print(
        "Do not change the threshold based on these test results."
    )


if __name__ == "__main__":
    main()
