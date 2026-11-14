from pathlib import Path
import pandas as pd

# Your exact dataset locations
SISFALL_PATH = Path("datasets/SisFall_dataset/Sisfall")
HAR70_PATH = Path("datasets/har70/har70plus")

# ============================================================
# SISFALL
# ============================================================

print("\n" + "=" * 60)
print("SISFALL DATASET")
print("=" * 60)

if not SISFALL_PATH.exists():
    print("ERROR: SisFall folder not found!")
    print("Expected:", SISFALL_PATH)
else:
    subjects = [p for p in SISFALL_PATH.iterdir() if p.is_dir()]

    print("Location:", SISFALL_PATH)
    print("Subject folders:", len(subjects))

    sa_subjects = [p for p in subjects if p.name.startswith("SA")]
    se_subjects = [p for p in subjects if p.name.startswith("SE")]

    print("SA subjects:", len(sa_subjects))
    print("SE subjects:", len(se_subjects))

    all_files = list(SISFALL_PATH.rglob("*.txt"))

    d_files = [f for f in all_files if f.name.startswith("D")]
    f_files = [f for f in all_files if f.name.startswith("F")]

    print("Total TXT recordings:", len(all_files))
    print("Normal activity (D):", len(d_files))
    print("Fall recordings (F):", len(f_files))

    if all_files:
        sample = all_files[0]

        print("\nSample file:", sample)

        with open(sample, "r") as file:
            print("First 5 lines:")

            for _ in range(5):
                print(file.readline().strip())


# ============================================================
# HAR70+
# ============================================================

print("\n" + "=" * 60)
print("HAR70+ DATASET")
print("=" * 60)

if not HAR70_PATH.exists():
    print("ERROR: HAR70+ folder not found!")
    print("Expected:", HAR70_PATH)
else:
    csv_files = sorted(HAR70_PATH.glob("*.csv"))

    print("Location:", HAR70_PATH)
    print("CSV files:", len(csv_files))

    if csv_files:
        sample_csv = csv_files[0]

        print("\nSample CSV:", sample_csv.name)

        df = pd.read_csv(sample_csv)

        print("Shape:", df.shape)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETE")
print("=" * 60)