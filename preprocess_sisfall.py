from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

SISFALL_PATH = Path("datasets/SisFall_dataset/Sisfall")
OUTPUT_PATH = Path("processed_data")

OUTPUT_PATH.mkdir(exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

# SisFall sampling frequency
SAMPLING_RATE = 200

# 1-second window = 200 samples
WINDOW_SIZE = 200

# 50% overlap
STEP_SIZE = 100


# ============================================================
# READ ONE SISFALL RECORDING
# ============================================================

def read_sisfall_file(file_path):

    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Remove semicolon at the end
            line = line.replace(";", "")

            values = line.split(",")

            # Every valid sensor row must contain 9 values
            if len(values) != 9:
                continue

            try:
                values = [float(value.strip()) for value in values]
                rows.append(values)

            except ValueError:
                continue

    return np.asarray(rows, dtype=np.float32)


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(window):

    # --------------------------------------------------------
    # SisFall format:
    #
    # 0,1,2 = Accelerometer 1 (ADXL345)
    # 3,4,5 = Accelerometer 2 (MMA8451Q)
    # 6,7,8 = Gyroscope
    #
    # We use Accelerometer 1 + Gyroscope.
    #
    # This gives us the same 6-channel concept as MPU6050:
    #
    # Acc X Y Z
    # Gyro X Y Z
    # --------------------------------------------------------

    acc_x = window[:, 0]
    acc_y = window[:, 1]
    acc_z = window[:, 2]

    gyro_x = window[:, 6]
    gyro_y = window[:, 7]
    gyro_z = window[:, 8]

    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    acc_mag = np.sqrt(
        acc_x**2 +
        acc_y**2 +
        acc_z**2
    )

    gyro_mag = np.sqrt(
        gyro_x**2 +
        gyro_y**2 +
        gyro_z**2
    )

    signals = {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag
    }

    features = {}

    # --------------------------------------------------------
    # Statistical features
    # --------------------------------------------------------

    for name, signal in signals.items():

        features[f"{name}_mean"] = np.mean(signal)

        features[f"{name}_std"] = np.std(signal)

        features[f"{name}_min"] = np.min(signal)

        features[f"{name}_max"] = np.max(signal)

        features[f"{name}_range"] = (
            np.max(signal) - np.min(signal)
        )

    # --------------------------------------------------------
    # Jerk
    # --------------------------------------------------------

    jerk = np.diff(acc_mag)

    features["jerk_mean"] = np.mean(np.abs(jerk))

    features["jerk_max"] = np.max(np.abs(jerk))

    return features


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print("=" * 70)
    print("SISFALL PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset path
    # --------------------------------------------------------

    if not SISFALL_PATH.exists():

        print("ERROR: SisFall dataset not found!")

        print("Expected path:")
        print(SISFALL_PATH)

        return

    # --------------------------------------------------------
    # Find subject folders
    # --------------------------------------------------------

    subjects = sorted(
        [
            folder
            for folder in SISFALL_PATH.iterdir()
            if folder.is_dir()
            and (
                folder.name.startswith("SA")
                or folder.name.startswith("SE")
            )
        ]
    )

    print("Subjects found:", len(subjects))

    # --------------------------------------------------------
    # Find recordings
    # --------------------------------------------------------

    recordings = []

    for subject_folder in subjects:

        subject_id = subject_folder.name

        for file_path in subject_folder.glob("*.txt"):

            filename = file_path.name

            # Normal activity
            if filename.startswith("D"):

                label = 0
                activity = "Normal"

            # Fall
            elif filename.startswith("F"):

                label = 1
                activity = "Fall"

            else:

                # Ignore Readme etc.
                continue

            recordings.append(
                {
                    "path": file_path,
                    "subject": subject_id,
                    "label": label,
                    "activity": activity
                }
            )

    print("Total recordings:", len(recordings))

    normal_recordings = sum(
        r["label"] == 0
        for r in recordings
    )

    fall_recordings = sum(
        r["label"] == 1
        for r in recordings
    )

    print("Normal recordings:", normal_recordings)
    print("Fall recordings:", fall_recordings)

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    dataset = []

    for index, recording in enumerate(recordings):

        file_path = recording["path"]

        data = read_sisfall_file(file_path)

        # Skip very short recordings
        if len(data) < WINDOW_SIZE:

            continue

        window_number = 0

        # ----------------------------------------------------
        # Sliding window
        # ----------------------------------------------------

        for start in range(
            0,
            len(data) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            end = start + WINDOW_SIZE

            window = data[start:end]

            features = extract_features(window)

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            features["label"] = recording["label"]

            features["activity"] = recording["activity"]

            features["subject"] = recording["subject"]

            features["recording"] = file_path.stem

            features["window"] = window_number

            dataset.append(features)

            window_number += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{len(recordings)} recordings"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    print("\nCreating final DataFrame...")

    df = pd.DataFrame(dataset)

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    print("\nTotal ML windows:", len(df))

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nActivity distribution:")

    print(
        df["activity"]
        .value_counts()
    )

    print("\nSubjects represented:", df["subject"].nunique())

    print("\nNumber of features:")

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "label",
            "activity",
            "subject",
            "recording",
            "window"
        ]
    ]

    print(len(feature_columns))

    print("\nFeature columns:")

    for column in feature_columns:

        print(" -", column)

    # ========================================================
    # SAVE DATASET
    # ========================================================

    output_file = OUTPUT_PATH / "sisfall_features.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print("\nSaved file:")
    print(output_file)

    print("\nFinal shape:")
    print(df.shape)

    print("\nFirst 5 rows:")

    print(df.head())


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()