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
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fall_detection_gru_improved_best.pth"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# IMPROVED BIDIRECTIONAL GRU
# Exact architecture used during training
# ============================================================

class ImprovedFallDetectionGRU(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        num_layers=2,
        dropout=0.4
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        return self.classifier(last_output)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("IMPROVED GRU THRESHOLD TUNING")
print("=" * 70)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val = np.load(
    os.path.join(
        DATA_DIR,
        "X_validation_raw.npy"
    )
)

y_val = np.load(
    os.path.join(
        DATA_DIR,
        "y_validation.npy"
    )
)


# ============================================================
# LOAD GRU NORMALIZATION
# ============================================================

channel_mean = np.load(
    os.path.join(
        DATA_DIR,
        "gru_improved_channel_mean.npy"
    )
)

channel_std = np.load(
    os.path.join(
        DATA_DIR,
        "gru_improved_channel_std.npy"
    )
)

print(
    "Validation data shape:",
    X_val.shape
)

print(
    "Validation labels shape:",
    y_val.shape
)


# ============================================================
# NORMALIZE VALIDATION DATA
# ============================================================

X_val = (
    X_val -
    channel_mean.reshape(1, 1, -1)
) / (
    channel_std.reshape(1, 1, -1) + 1e-8
)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading trained improved GRU...")

model = ImprovedFallDetectionGRU(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    dropout=0.4
).to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    "Improved GRU loaded successfully."
)

print(
    "Best training epoch:",
    checkpoint["best_epoch"]
)

print(
    "Best validation F1:",
    checkpoint["best_val_f1"]
)


# ============================================================
# CALCULATE VALIDATION PROBABILITIES
# ============================================================

print(
    "\nCalculating validation probabilities..."
)

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

        probs = torch.sigmoid(
            logits
        )

        probabilities.extend(
            probs.cpu()
            .numpy()
            .flatten()
        )


probabilities = np.array(
    probabilities
)


print(
    "Probability calculation complete."
)


# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "THRESHOLD RESULTS"
)

print(
    "=" * 70
)


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


    false_alarm_rate = (
        fp / (fp + tn)
    )


    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate":
                false_alarm_rate,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    )


    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"False Alarm Rate: "
        f"{false_alarm_rate:.4f}"
    )


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best = max(
    results,
    key=lambda x: x["f1"]
)


print(
    "\n" + "=" * 70
)

print(
    "BEST THRESHOLD BY F1"
)

print(
    "=" * 70
)


print(
    f"Threshold       : "
    f"{best['threshold']:.2f}"
)

print(
    f"Precision       : "
    f"{best['precision']:.4f}"
)

print(
    f"Recall          : "
    f"{best['recall']:.4f}"
)

print(
    f"F1 Score        : "
    f"{best['f1']:.4f}"
)

print(
    f"False Alarm Rate: "
    f"{best['false_alarm_rate']:.4f}"
)


print("\nConfusion Matrix:")

print(
    np.array(
        [
            [
                best["tn"],
                best["fp"]
            ],
            [
                best["fn"],
                best["tp"]
            ]
        ]
    )
)


# ============================================================
# BEST THRESHOLD WITH RECALL >= 70%
# ============================================================

recall_target = 0.70


eligible = [
    r
    for r in results
    if r["recall"] >= recall_target
]


if eligible:

    best_70 = max(
        eligible,
        key=lambda x: x["precision"]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "BEST THRESHOLD WITH RECALL >= 70%"
    )

    print(
        "=" * 70
    )


    print(
        f"Threshold       : "
        f"{best_70['threshold']:.2f}"
    )

    print(
        f"Precision       : "
        f"{best_70['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_70['recall']:.4f}"
    )

    print(
        f"F1 Score        : "
        f"{best_70['f1']:.4f}"
    )

    print(
        f"False Alarm Rate: "
        f"{best_70['false_alarm_rate']:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        np.array(
            [
                [
                    best_70["tn"],
                    best_70["fp"]
                ],
                [
                    best_70["fn"],
                    best_70["tp"]
                ]
            ]
        )
    )


# ============================================================
# BEST THRESHOLD WITH RECALL >= 75%
# ============================================================

recall_target_75 = 0.75


eligible_75 = [
    r
    for r in results
    if r["recall"] >= recall_target_75
]


if eligible_75:

    best_75 = max(
        eligible_75,
        key=lambda x: x["precision"]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "BEST THRESHOLD WITH RECALL >= 75%"
    )

    print(
        "=" * 70
    )


    print(
        f"Threshold       : "
        f"{best_75['threshold']:.2f}"
    )

    print(
        f"Precision       : "
        f"{best_75['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_75['recall']:.4f}"
    )

    print(
        f"F1 Score        : "
        f"{best_75['f1']:.4f}"
    )

    print(
        f"False Alarm Rate: "
        f"{best_75['false_alarm_rate']:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        np.array(
            [
                [
                    best_75["tn"],
                    best_75["fp"]
                ],
                [
                    best_75["fn"],
                    best_75["tp"]
                ]
            ]
        )
    )


# ============================================================
# BEST THRESHOLD WITH RECALL >= 80%
# ============================================================

recall_target_80 = 0.80


eligible_80 = [
    r
    for r in results
    if r["recall"] >= recall_target_80
]


if eligible_80:

    best_80 = max(
        eligible_80,
        key=lambda x: x["precision"]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "BEST THRESHOLD WITH RECALL >= 80%"
    )

    print(
        "=" * 70
    )


    print(
        f"Threshold       : "
        f"{best_80['threshold']:.2f}"
    )

    print(
        f"Precision       : "
        f"{best_80['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_80['recall']:.4f}"
    )

    print(
        f"F1 Score        : "
        f"{best_80['f1']:.4f}"
    )

    print(
        f"False Alarm Rate: "
        f"{best_80['false_alarm_rate']:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        np.array(
            [
                [
                    best_80["tn"],
                    best_80["fp"]
                ],
                [
                    best_80["fn"],
                    best_80["tp"]
                ]
            ]
        )
    )

else:

    print(
        "\nNo threshold achieved "
        "recall >= 80%."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "gru_improved_threshold_results.txt"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IMPROVED GRU THRESHOLD TUNING RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    f.write(
        "ALL THRESHOLDS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )


    for r in results:

        f.write(
            f"Threshold: "
            f"{r['threshold']:.2f}\n"
        )

        f.write(
            f"Precision: "
            f"{r['precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{r['recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{r['f1']:.4f}\n"
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


    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "BEST THRESHOLD BY F1\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        f"Threshold: "
        f"{best['threshold']:.2f}\n"
    )

    f.write(
        f"Precision: "
        f"{best['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best['recall']:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{best['f1']:.4f}\n"
    )

    f.write(
        f"False Alarm Rate: "
        f"{best['false_alarm_rate']:.4f}\n"
    )


    if eligible:

        f.write(
            "\n" + "=" * 70 + "\n"
        )

        f.write(
            "BEST THRESHOLD WITH RECALL >= 70%\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{best_70['threshold']:.2f}\n"
        )

        f.write(
            f"Precision: "
            f"{best_70['precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{best_70['recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{best_70['f1']:.4f}\n"
        )

        f.write(
            f"False Alarm Rate: "
            f"{best_70['false_alarm_rate']:.4f}\n"
        )


    if eligible_75:

        f.write(
            "\n" + "=" * 70 + "\n"
        )

        f.write(
            "BEST THRESHOLD WITH RECALL >= 75%\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{best_75['threshold']:.2f}\n"
        )

        f.write(
            f"Precision: "
            f"{best_75['precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{best_75['recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{best_75['f1']:.4f}\n"
        )

        f.write(
            f"False Alarm Rate: "
            f"{best_75['false_alarm_rate']:.4f}\n"
        )


    if eligible_80:

        f.write(
            "\n" + "=" * 70 + "\n"
        )

        f.write(
            "BEST THRESHOLD WITH RECALL >= 80%\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{best_80['threshold']:.2f}\n"
        )

        f.write(
            f"Precision: "
            f"{best_80['precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{best_80['recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{best_80['f1']:.4f}\n"
        )

        f.write(
            f"False Alarm Rate: "
            f"{best_80['false_alarm_rate']:.4f}\n"
        )


print("\nResults saved to:")
print(output_file)

print(
    "\n" + "=" * 70
)

print(
    "IMPROVED GRU THRESHOLD TUNING COMPLETE"
)

print(
    "=" * 70
)