from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("processed_data/sisfall_features.csv")
OUTPUT_DIR = Path("processed_data/split")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# 70% training
# 15% validation
# 15% testing


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SUBJECT-WISE DATASET SPLITTING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total samples:", len(df))
print("Total subjects:", df["subject"].nunique())


# ============================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_indices, temp_indices = next(
    splitter.split(
        df,
        groups=df["subject"]
    )
)

train_df = df.iloc[train_indices].copy()
temp_df = df.iloc[temp_indices].copy()


# ============================================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ============================================================

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_indices, test_indices = next(
    splitter2.split(
        temp_df,
        groups=temp_df["subject"]
    )
)

val_df = temp_df.iloc[val_indices].copy()
test_df = temp_df.iloc[test_indices].copy()


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print("\nTRAINING SET")
print("Samples:", len(train_df))
print("Subjects:", train_df["subject"].nunique())
print("Subjects:", sorted(train_df["subject"].unique()))

print("\nVALIDATION SET")
print("Samples:", len(val_df))
print("Subjects:", val_df["subject"].nunique())
print("Subjects:", sorted(val_df["subject"].unique()))

print("\nTEST SET")
print("Samples:", len(test_df))
print("Subjects:", test_df["subject"].nunique())
print("Subjects:", sorted(test_df["subject"].unique()))


# ============================================================
# CHECK FOR DATA LEAKAGE
# ============================================================

train_subjects = set(train_df["subject"])
val_subjects = set(val_df["subject"])
test_subjects = set(test_df["subject"])

train_val_overlap = train_subjects & val_subjects
train_test_overlap = train_subjects & test_subjects
val_test_overlap = val_subjects & test_subjects

print("\n" + "=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

print("Train ∩ Validation:", train_val_overlap)
print("Train ∩ Test:", train_test_overlap)
print("Validation ∩ Test:", val_test_overlap)


if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print("\n✅ NO SUBJECT LEAKAGE DETECTED!")

else:

    print("\n❌ WARNING: SUBJECT OVERLAP DETECTED!")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for name, dataset in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    counts = dataset["label"].value_counts()

    print("Normal (0):", counts.get(0, 0))
    print("Fall   (1):", counts.get(1, 0))


# ============================================================
# SAVE DATASETS
# ============================================================

train_file = OUTPUT_DIR / "train.csv"
val_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASETS SAVED")
print("=" * 70)

print("\nTraining:")
print(train_file)

print("\nValidation:")
print(val_file)

print("\nTesting:")
print(test_file)

print("\nTotal samples check:")

print(
    len(train_df)
    + len(val_df)
    + len(test_df),
    "==",
    len(df)
)

print("\n✅ SPLITTING COMPLETE!")