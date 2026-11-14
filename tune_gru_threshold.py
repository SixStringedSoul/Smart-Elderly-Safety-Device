import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "processed_data", "cnn")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fall_detection_gru_best.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# GRU MODEL
# Same architecture used during training
# ============================================================

class FallDetectionGRU(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        return self.classifier(last_output)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("GRU THRESHOLD TUNING")
print("=" * 60)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val = np.load(
    os.path.join(DATA_DIR, "X_validation_raw.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_validation.npy")
)

# Use the SAME normalization calculated during GRU training
channel_mean = np.load(
    os.path.join(DATA_DIR, "gru_channel_mean.npy")
)

channel_std = np.load(
    os.path.join(DATA_DIR, "gru_channel_std.npy")
)

print("Validation data shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)


# ============================================================
# NORMALIZE
# ============================================================

X_val = (
    X_val - channel_mean.reshape(1, 1, -1)
) / (
    channel_std.reshape(1, 1, -1) + 1e-8
)


# ============================================================
# PYTORCH TENSOR
# ============================================================

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD BEST GRU
# ============================================================

print("\nLoading trained GRU...")

model = FallDetectionGRU(
    input_size=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("GRU loaded successfully.")
print("Best training epoch:", checkpoint["best_epoch"])
print(
    "Best validation F1:",
    checkpoint["best_val_f1"]
)


# ============================================================
# CALCULATE VALIDATION PROBABILITIES
# ============================================================

print("\nCalculating validation probabilities...")

probabilities = []

batch_size = 256

with torch.no_grad():

    for i in range(
        0,
        len(X_val_tensor),
        batch_size
    ):

        batch = X_val_tensor[
            i:i + batch_size
        ].to(device)

        logits = model(batch)

        probs = torch.sigmoid(logits)

        probabilities.extend(
            probs.cpu().numpy().flatten()
        )

probabilities = np.array(probabilities)

print("Probability calculation complete.")


# ============================================================
# TEST THRESHOLDS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD RESULTS")
print("=" * 60)

results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_val,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    false_alarm_rate = fp / (fp + tn)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"False Alarm Rate: {false_alarm_rate:.4f}"
    )


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best = max(
    results,
    key=lambda x: x["f1"]
)

print("\n" + "=" * 60)
print("BEST THRESHOLD BY F1")
print("=" * 60)

print(
    f"Threshold       : {best['threshold']:.2f}"
)

print(
    f"Precision       : {best['precision']:.4f}"
)

print(
    f"Recall          : {best['recall']:.4f}"
)

print(
    f"F1 Score        : {best['f1']:.4f}"
)

print(
    f"False Alarm Rate: {best['false_alarm_rate']:.4f}"
)

print("\nConfusion Matrix:")

print(
    np.array([
        [best["tn"], best["fp"]],
        [best["fn"], best["tp"]]
    ])
)


# ============================================================
# BEST THRESHOLD WITH RECALL >= 70%
# ============================================================

recall_target = 0.70

eligible = [
    r for r in results
    if r["recall"] >= recall_target
]

if eligible:

    best_recall = max(
        eligible,
        key=lambda x: x["precision"]
    )

    print("\n" + "=" * 60)
    print("BEST THRESHOLD WITH RECALL >= 70%")
    print("=" * 60)

    print(
        f"Threshold       : {best_recall['threshold']:.2f}"
    )

    print(
        f"Precision       : {best_recall['precision']:.4f}"
    )

    print(
        f"Recall          : {best_recall['recall']:.4f}"
    )

    print(
        f"F1 Score        : {best_recall['f1']:.4f}"
    )

    print(
        f"False Alarm Rate: {best_recall['false_alarm_rate']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        np.array([
            [
                best_recall["tn"],
                best_recall["fp"]
            ],
            [
                best_recall["fn"],
                best_recall["tp"]
            ]
        ])
    )


# ============================================================
# BEST THRESHOLD WITH RECALL >= 80%
# ============================================================

recall_target_high = 0.80

eligible_high = [
    r for r in results
    if r["recall"] >= recall_target_high
]

if eligible_high:

    best_high = max(
        eligible_high,
        key=lambda x: x["precision"]
    )

    print("\n" + "=" * 60)
    print("BEST THRESHOLD WITH RECALL >= 80%")
    print("=" * 60)

    print(
        f"Threshold       : {best_high['threshold']:.2f}"
    )

    print(
        f"Precision       : {best_high['precision']:.4f}"
    )

    print(
        f"Recall          : {best_high['recall']:.4f}"
    )

    print(
        f"F1 Score        : {best_high['f1']:.4f}"
    )

    print(
        f"False Alarm Rate: {best_high['false_alarm_rate']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        np.array([
            [
                best_high["tn"],
                best_high["fp"]
            ],
            [
                best_high["fn"],
                best_high["tp"]
            ]
        ])
    )

else:

    print("\nNo threshold achieved recall >= 80%.")


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "gru_threshold_results.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("GRU THRESHOLD TUNING RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("ALL THRESHOLDS\n")
    f.write("-" * 60 + "\n")

    for r in results:

        f.write(
            f"Threshold: {r['threshold']:.2f}\n"
        )

        f.write(
            f"Precision: {r['precision']:.4f}\n"
        )

        f.write(
            f"Recall: {r['recall']:.4f}\n"
        )

        f.write(
            f"F1: {r['f1']:.4f}\n"
        )

        f.write(
            f"False Alarm Rate: "
            f"{r['false_alarm_rate']:.4f}\n"
        )

        f.write(
            f"TN: {r['tn']}, "
            f"FP: {r['fp']}, "
            f"FN: {r['fn']}, "
            f"TP: {r['tp']}\n\n"
        )

    f.write("=" * 60 + "\n")
    f.write("BEST THRESHOLD BY F1\n")
    f.write("=" * 60 + "\n")

    f.write(
        f"Threshold: {best['threshold']:.2f}\n"
    )

    f.write(
        f"Precision: {best['precision']:.4f}\n"
    )

    f.write(
        f"Recall: {best['recall']:.4f}\n"
    )

    f.write(
        f"F1: {best['f1']:.4f}\n"
    )

    f.write(
        f"False Alarm Rate: "
        f"{best['false_alarm_rate']:.4f}\n"
    )

    if eligible:

        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("BEST THRESHOLD WITH RECALL >= 70%\n")
        f.write("=" * 60 + "\n")

        f.write(
            f"Threshold: "
            f"{best_recall['threshold']:.2f}\n"
        )

        f.write(
            f"Precision: "
            f"{best_recall['precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{best_recall['recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{best_recall['f1']:.4f}\n"
        )

        f.write(
            f"False Alarm Rate: "
            f"{best_recall['false_alarm_rate']:.4f}\n"
        )


print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 60)
print("GRU THRESHOLD TUNING COMPLETE")
print("=" * 60)