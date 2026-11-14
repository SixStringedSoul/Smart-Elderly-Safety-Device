"""
Guardian - LSTM Threshold Tuning
Uses VALIDATION data only to select the best probability threshold.
Do NOT use the test set here.

Run:
    python tune_lstm_threshold.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "processed_data" / "cnn"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

X_VAL = DATA / "X_validation_raw.npy"
Y_VAL = DATA / "y_validation.npy"
MEAN = DATA / "lstm_channel_mean.npy"
STD = DATA / "lstm_channel_std.npy"
MODEL = MODELS / "fall_detection_lstm_best.pth"

BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
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
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)


@torch.no_grad()
def get_probabilities(model, X, mean, std):
    X = X.astype(np.float32)
    X = (X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    probs = []
    for start in range(0, len(X), BATCH_SIZE):
        batch = torch.from_numpy(X[start:start+BATCH_SIZE]).to(DEVICE)
        probs.append(torch.sigmoid(model(batch)).cpu().numpy())

    return np.concatenate(probs)


def metrics(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    print("=" * 80)
    print("GUARDIAN - LSTM THRESHOLD TUNING")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for path in [X_VAL, Y_VAL, MEAN, STD, MODEL]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("\nLoading VALIDATION data...")
    X = np.load(X_VAL)
    y = np.load(Y_VAL)
    mean = np.load(MEAN).astype(np.float32)
    std = np.load(STD).astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    print(f"X_validation shape: {X.shape}")
    print(f"y_validation shape: {y.shape}")

    model = LSTMModel().to(DEVICE)
    checkpoint = torch.load(MODEL, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    print("\nGenerating VALIDATION probabilities...")
    probs = get_probabilities(model, X, mean, std)

    print(f"Probability range: {probs.min():.4f} - {probs.max():.4f}")

    rows = []
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP")
    print("=" * 80)
    print("\nThreshold | Precision | Recall | F1 | False Alarm")
    print("-" * 60)

    for t in np.round(np.arange(0.10, 0.901, 0.01), 2):
        r = metrics(y, probs, float(t))
        rows.append(r)
        print(
            f"{t:8.2f} | "
            f"{r['precision']*100:8.2f}% | "
            f"{r['recall']*100:6.2f}% | "
            f"{r['f1']*100:5.2f}% | "
            f"{r['false_alarm_rate']*100:10.2f}%"
        )

    df = pd.DataFrame(rows)
    best = df.loc[df["f1"].idxmax()]

    print("\n" + "=" * 80)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 80)
    print(f"\nThreshold       : {best['threshold']:.2f}")
    print(f"Accuracy        : {best['accuracy']*100:.2f}%")
    print(f"Precision       : {best['precision']*100:.2f}%")
    print(f"Recall          : {best['recall']*100:.2f}%")
    print(f"F1-score        : {best['f1']*100:.2f}%")
    print(f"False Alarm Rate: {best['false_alarm_rate']*100:.2f}%")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Normal   Fall")
    print(f"Actual Normal    {int(best['tn']):6d}   {int(best['fp']):6d}")
    print(f"Actual Fall      {int(best['fn']):6d}   {int(best['tp']):6d}")

    print("\n" + "=" * 80)
    print("BEST F1 AT RECALL TARGETS")
    print("=" * 80)

    for target in [0.60, 0.65, 0.70, 0.75, 0.80]:
        eligible = df[df["recall"] >= target]
        if len(eligible):
            r = eligible.loc[eligible["f1"].idxmax()]
            print(
                f"Recall >= {target*100:.0f}%: "
                f"threshold={r['threshold']:.2f}, "
                f"precision={r['precision']*100:.2f}%, "
                f"recall={r['recall']*100:.2f}%, "
                f"F1={r['f1']*100:.2f}%, "
                f"false alarms={r['false_alarm_rate']*100:.2f}%"
            )

    RESULTS.mkdir(parents=True, exist_ok=True)

    sweep_path = RESULTS / "lstm_threshold_validation.csv"
    summary_path = RESULTS / "lstm_best_threshold.txt"

    df.to_csv(sweep_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GUARDIAN - LSTM VALIDATION THRESHOLD TUNING\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model: Bidirectional LSTM\n")
        f.write("Split: VALIDATION ONLY\n\n")
        f.write(f"Best threshold: {best['threshold']:.2f}\n")
        f.write(f"Accuracy: {best['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {best['precision']*100:.2f}%\n")
        f.write(f"Recall: {best['recall']*100:.2f}%\n")
        f.write(f"F1: {best['f1']*100:.2f}%\n")
        f.write(f"False Alarm Rate: {best['false_alarm_rate']*100:.2f}%\n")
        f.write(f"TN: {int(best['tn'])}\n")
        f.write(f"FP: {int(best['fp'])}\n")
        f.write(f"FN: {int(best['fn'])}\n")
        f.write(f"TP: {int(best['tp'])}\n\n")
        f.write("Threshold selected using validation only.\n")
        f.write("Do not tune using the test set.\n")

    print("\nSaved:")
    print(f"  {sweep_path}")
    print(f"  {summary_path}")



if __name__ == "__main__":
    main()
