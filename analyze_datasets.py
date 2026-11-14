from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

SISFALL_PATH = Path("datasets/SisFall_dataset/Sisfall")
HAR70_PATH = Path("datasets/har70/har70plus")


# ============================================================
# SISFALL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SISFALL ANALYSIS")
print("=" * 70)

# Only actual subject folders
subject_folders = [
    p for p in SISFALL_PATH.iterdir()
    if p.is_dir() and (p.name.startswith("SA") or p.name.startswith("SE"))
]

print("Subject folders:", len(subject_folders))

# Only D and F recordings
d_files = []
f_files = []

for subject in subject_folders:
    for file in subject.glob("*.txt"):

        if file.name.startswith("D"):
            d_files.append(file)

        elif file.name.startswith("F"):
            f_files.append(file)

print("Normal activity recordings (D):", len(d_files))
print("Fall recordings (F):", len(f_files))
print("Total actual recordings:", len(d_files) + len(f_files))


# ------------------------------------------------------------
# Show one normal recording
# ------------------------------------------------------------

if d_files:

    normal_file = d_files[0]

    print("\n--- NORMAL ACTIVITY SAMPLE ---")
    print("File:", normal_file)

    with open(normal_file, "r") as f:

        for i in range(5):
            line = f.readline().strip()

            if line:
                print(line)


# ------------------------------------------------------------
# Show one fall recording
# ------------------------------------------------------------

if f_files:

    fall_file = f_files[0]

    print("\n--- FALL SAMPLE ---")
    print("File:", fall_file)

    with open(fall_file, "r") as f:

        for i in range(5):
            line = f.readline().strip()

            if line:
                print(line)


# ============================================================
# HAR70+ ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HAR70+ ANALYSIS")
print("=" * 70)

csv_files = sorted(HAR70_PATH.glob("*.csv"))

print("Number of CSV files:", len(csv_files))

if csv_files:

    # Combine only labels first
    label_counts = {}

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)

        print(
            f"{csv_file.name}: "
            f"{len(df):,} rows"
        )

        counts = df["label"].value_counts()

        for label, count in counts.items():

            label_counts[label] = (
                label_counts.get(label, 0) + count
            )

    print("\nHAR70+ label counts:")

    for label, count in sorted(label_counts.items()):

        print(f"Label {label}: {count:,} samples")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)