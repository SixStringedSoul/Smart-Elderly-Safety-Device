import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

SISFALL_DIR = Path("datasets/SisFall_dataset/Sisfall")
OUTPUT_DIR = Path("processed_data/cnn")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 200
WINDOW_SIZE = 200       # 1 second
STEP_SIZE = 100         # 50% overlap

# SisFall:
# Columns 0-2 = ADXL345 accelerometer X/Y/Z
# Columns 6-8 = ITG3200 gyroscope X/Y/Z
SELECTED_COLUMNS = [0, 1, 2, 6, 7, 8]


# ============================================================
# READ ONE SISFALL FILE
# ============================================================

def read_sisfall_file(file_path):

    data = []

    with open(file_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Remove semicolon at end
            line = line.replace(";", "")

            values = [
                float(x.strip())
                for x in line.split(",")
            ]

            if len(values) >= 9:
                data.append(values)

    return np.array(data, dtype=np.float32)


# ============================================================
# EXTRACT WINDOWS
# ============================================================

def create_windows(data):

    windows = []

    for start in range(
        0,
        len(data) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE

        window = data[
            start:end,
            SELECTED_COLUMNS
        ]

        if window.shape == (WINDOW_SIZE, 6):
            windows.append(window)

    return windows


# ============================================================
# COLLECT RECORDINGS
# ============================================================

print("=" * 60)
print("PREPARING RAW SISFALL DATA FOR 1D CNN")
print("=" * 60)

print("\nDataset location:")
print(SISFALL_DIR)

recordings = []

for subject_dir in sorted(SISFALL_DIR.iterdir()):

    if not subject_dir.is_dir():
        continue

    subject = subject_dir.name

    for file_path in sorted(subject_dir.glob("*.txt")):

        filename = file_path.name

        # Skip README
        if not filename.startswith(("D", "F")):
            continue

        if filename.startswith("F"):
            label = 1
            activity = "Fall"

        elif filename.startswith("D"):
            label = 0
            activity = "Normal"

        else:
            continue

        recordings.append({
            "path": file_path,
            "subject": subject,
            "recording": filename,
            "label": label,
            "activity": activity
        })


print("\nTotal recordings:", len(recordings))

print(
    "Normal recordings:",
    sum(r["label"] == 0 for r in recordings)
)

print(
    "Fall recordings:",
    sum(r["label"] == 1 for r in recordings)
)


# ============================================================
# CREATE ALL WINDOWS
# ============================================================

all_windows = []
metadata = []

print("\nCreating time-series windows...")

for i, recording in enumerate(recordings):

    data = read_sisfall_file(recording["path"])

    windows = create_windows(data)

    for window_number, window in enumerate(windows):

        all_windows.append(window)

        metadata.append([
            recording["subject"],
            recording["recording"],
            recording["label"],
            recording["activity"],
            window_number
        ])

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1}/{len(recordings)} recordings..."
        )


X = np.array(all_windows, dtype=np.float32)

metadata = pd.DataFrame(
    metadata,
    columns=[
        "subject",
        "recording",
        "label",
        "activity",
        "window"
    ]
)


print("\n" + "=" * 60)
print("WINDOW CREATION COMPLETE")
print("=" * 60)

print("X shape:", X.shape)
print("Metadata shape:", metadata.shape)

print("\nExpected window shape:")
print("(samples, time_steps, channels)")

print(
    f"Actual: ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})"
)


# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING SUBJECT-WISE SPLIT")
print("=" * 60)

subjects = metadata["subject"].values

# First split: 70% train / 30% temporary
splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    splitter_1.split(
        X,
        metadata["label"].values,
        groups=subjects
    )
)

# Second split: temporary → 50% validation / 50% test
splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_relative_idx, test_relative_idx = next(
    splitter_2.split(
        X[temp_idx],
        metadata.iloc[temp_idx]["label"].values,
        groups=subjects[temp_idx]
    )
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]


# ============================================================
# CREATE SPLITS
# ============================================================

X_train = X[train_idx]
y_train = metadata.iloc[train_idx]["label"].values

X_val = X[val_idx]
y_val = metadata.iloc[val_idx]["label"].values

X_test = X[test_idx]
y_test = metadata.iloc[test_idx]["label"].values


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

train_subjects = sorted(
    metadata.iloc[train_idx]["subject"].unique()
)

val_subjects = sorted(
    metadata.iloc[val_idx]["subject"].unique()
)

test_subjects = sorted(
    metadata.iloc[test_idx]["subject"].unique()
)

print("\nTRAINING SET")
print("Samples:", len(X_train))
print("Subjects:", len(train_subjects))
print("Subjects:", train_subjects)

print("\nVALIDATION SET")
print("Samples:", len(X_val))
print("Subjects:", len(val_subjects))
print("Subjects:", val_subjects)

print("\nTEST SET")
print("Samples:", len(X_test))
print("Subjects:", len(test_subjects))
print("Subjects:", test_subjects)


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATA LEAKAGE CHECK")
print("=" * 60)

train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

print(
    "Train ∩ Validation:",
    train_set.intersection(val_set)
)

print(
    "Train ∩ Test:",
    train_set.intersection(test_set)
)

print(
    "Validation ∩ Test:",
    val_set.intersection(test_set)
)

if (
    not train_set.intersection(val_set)
    and not train_set.intersection(test_set)
    and not val_set.intersection(test_set)
):
    print("NO SUBJECT LEAKAGE DETECTED!")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(
    "TRAIN:",
    "Normal", np.sum(y_train == 0),
    "Fall", np.sum(y_train == 1)
)

print(
    "VALIDATION:",
    "Normal", np.sum(y_val == 0),
    "Fall", np.sum(y_val == 1)
)

print(
    "TEST:",
    "Normal", np.sum(y_test == 0),
    "Fall", np.sum(y_test == 1)
)


# ============================================================
# SAVE DATA
# ============================================================

print("\n" + "=" * 60)
print("SAVING CNN DATA")
print("=" * 60)

np.save(
    OUTPUT_DIR / "X_train_raw.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "X_validation_raw.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "y_validation.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "X_test_raw.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)

metadata.iloc[train_idx].to_csv(
    OUTPUT_DIR / "train_metadata.csv",
    index=False
)

metadata.iloc[val_idx].to_csv(
    OUTPUT_DIR / "validation_metadata.csv",
    index=False
)

metadata.iloc[test_idx].to_csv(
    OUTPUT_DIR / "test_metadata.csv",
    index=False
)


# ============================================================
# SAVE CHANNEL INFORMATION
# ============================================================

with open(
    OUTPUT_DIR / "channel_names.txt",
    "w"
) as f:

    f.write("acc_x\n")
    f.write("acc_y\n")
    f.write("acc_z\n")
    f.write("gyro_x\n")
    f.write("gyro_y\n")
    f.write("gyro_z\n")


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

print(
    "Training:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)

print("\nChannel order:")
print("1. Accelerometer X")
print("2. Accelerometer Y")
print("3. Accelerometer Z")
print("4. Gyroscope X")
print("5. Gyroscope Y")
print("6. Gyroscope Z")

print("\nSampling rate:", SAMPLE_RATE, "Hz")
print("Window size:", WINDOW_SIZE, "samples")
print("Window duration:", WINDOW_SIZE / SAMPLE_RATE, "second")
print("Overlap:", "50%")

print("\nSaved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 60)
print("CNN DATA PREPARATION COMPLETE")
print("=" * 60)