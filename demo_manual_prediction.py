import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"models\fall_detection_gru_improved_best.pth"

SEQ_LEN = 200
STEP = 100
THRESHOLD = 0.60

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# BiGRU MODEL
# ============================================================

class BiGRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.gru(x)

        x = output[:, -1, :]

        return self.classifier(x).squeeze(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Guardian BiGRU model...")

model = BiGRU().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

else:

    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)


# ============================================================
# GET FILE PATH FROM USER
# ============================================================

print("\n" + "=" * 65)
print("          GUARDIAN FALL DETECTION")
print("=" * 65)

print("""
Paste the FULL path of any SisFall .txt recording.

Example:

C:\\Users\\aryan\\OneDrive\\Desktop\\CAPSTONE PJT-1\\datasets\\SisFall_dataset\\Sisfall\\SA01\\F01_SA01_R01.txt
""")

file_path = input(
    "\nEnter recording path: "
).strip().strip('"')


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.isfile(file_path):

    print("\nERROR: File not found!")

    print("\nYou entered:")
    print(file_path)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# GET ACTUAL CLASS
# ============================================================

filename = os.path.basename(file_path)

if filename.upper().startswith("F"):

    actual_class = "FALL"

elif filename.upper().startswith("D"):

    actual_class = "NORMAL"

else:

    actual_class = "UNKNOWN"


# ============================================================
# READ SISFALL FILE
# ============================================================

def read_sisfall(filepath):

    with open(
        filepath,
        "r",
        errors="ignore"
    ) as f:

        text = f.read()

    samples = []

    for sample in text.split(";"):

        sample = sample.strip()

        if not sample:
            continue

        values = sample.split(",")

        if len(values) != 9:
            continue

        try:

            values = [
                float(v.strip())
                for v in values
            ]

            # SisFall original channels:
            #
            # 0,1,2 = acceleration XYZ
            # 6,7,8 = gyroscope XYZ
            #
            # Guardian input:
            #
            # AccX AccY AccZ GyroX GyroY GyroZ

            samples.append([
                values[0],
                values[1],
                values[2],
                values[6],
                values[7],
                values[8]
            ])

        except ValueError:

            continue

    return np.array(
        samples,
        dtype=np.float32
    )


# ============================================================
# READ RECORDING
# ============================================================

print("\nReading recording...")

data = read_sisfall(file_path)

print(
    "Samples found:",
    len(data)
)

print(
    "Actual dataset class:",
    actual_class
)


if len(data) < SEQ_LEN:

    print(
        "\nERROR: Recording contains fewer than 200 samples."
    )

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# CREATE WINDOWS
# ============================================================

windows = []

window_starts = []

for start in range(
    0,
    len(data) - SEQ_LEN + 1,
    STEP
):

    window = data[
        start:start + SEQ_LEN
    ]

    windows.append(window)

    window_starts.append(start)


print(
    "Windows created:",
    len(windows)
)


# ============================================================
# PREDICT ALL WINDOWS
# ============================================================

print("\nRunning BiGRU prediction...")


results = []


for window, start in zip(
    windows,
    window_starts
):

    # --------------------------------------------------------
    # Normalize window
    # --------------------------------------------------------

    mean = window.mean(axis=0)

    std = window.std(axis=0)

    std[std == 0] = 1.0

    normalized = (
        window - mean
    ) / std


    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    X = torch.tensor(
        normalized,
        dtype=torch.float32
    )

    X = X.unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(X)

        probability = torch.sigmoid(
            output
        ).item()


    prediction = (
        "FALL"
        if probability >= THRESHOLD
        else "NORMAL"
    )


    results.append({
        "start": start,
        "probability": probability,
        "prediction": prediction
    })


# ============================================================
# FIND HIGHEST FALL PROBABILITY
# ============================================================

best = max(
    results,
    key=lambda x: x["probability"]
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 65)
print("                    RESULT")
print("=" * 65)

print(
    "Recording:",
    filename
)

print(
    "Actual dataset class:",
    actual_class
)

print(
    "Total samples:",
    len(data)
)

print(
    "Total windows:",
    len(results)
)

print()

print(
    "Highest fall probability:"
)

print(
    f"{best['probability']:.4f}"
)

print(
    f"({best['probability'] * 100:.2f}%)"
)

print(
    "Detected at sample:",
    best["start"]
)

print(
    "Threshold:",
    THRESHOLD
)

print()

if best["probability"] >= THRESHOLD:

    print(
        "MODEL PREDICTION: FALL"
    )

else:

    print(
        "MODEL PREDICTION: NORMAL"
    )

print("=" * 65)


# ============================================================
# COMPARE WITH DATASET LABEL
# ============================================================

if actual_class != "UNKNOWN":

    if best["prediction"] == actual_class:

        print(
            "\n✓ Prediction matches the dataset recording label."
        )

    else:

        print(
            "\n✗ Prediction does not match the dataset recording label."
        )


# ============================================================
# SHOW TOP 5 WINDOWS
# ============================================================

print("\nTop 5 windows:")

print("-" * 65)

top5 = sorted(
    results,
    key=lambda x: x["probability"],
    reverse=True
)[:5]


for result in top5:

    print(
        f"Sample {result['start']:6d} | "
        f"Probability: {result['probability']:.4f} | "
        f"{result['prediction']}"
    )


print("\n" + "=" * 65)
print("Demo completed.")
print("=" * 65)

input("\nPress Enter to exit...")