import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ==============================
# PATHS
# ==============================

DATA_DIR = Path("processed_data/split")
OUTPUT_DIR = Path("processed_data/prepared")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

print("Loading datasets...")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train:      {train.shape}")
print(f"Validation: {validation.shape}")
print(f"Test:       {test.shape}")

# ==============================
# SEPARATE FEATURES AND LABEL
# ==============================

# These columns are metadata and must NOT be used as ML features
metadata_columns = [
    "label",
    "activity",
    "subject",
    "recording",
    "window"
]

feature_columns = [
    col for col in train.columns
    if col not in metadata_columns
]

print("\nNumber of features:", len(feature_columns))
print("Features:")
print(feature_columns)

# ==============================
# CREATE X AND y
# ==============================

X_train = train[feature_columns].values
y_train = train["label"].values

X_val = validation[feature_columns].values
y_val = validation["label"].values

X_test = test[feature_columns].values
y_test = test["label"].values

print("\nBefore scaling:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)

# ==============================
# FEATURE SCALING
# ==============================

print("\nFitting StandardScaler on TRAINING data only...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# IMPORTANT:
# Validation and test are ONLY transformed.
# They are NOT used to fit the scaler.
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==============================
# SAVE PREPARED DATA
# ==============================

np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_validation.npy", X_val_scaled)
np.save(OUTPUT_DIR / "y_validation.npy", y_val)

np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# Save scaler for future inference
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

# Save feature names
with open(OUTPUT_DIR / "feature_names.txt", "w") as f:
    for feature in feature_columns:
        f.write(feature + "\n")

# ==============================
# VERIFY
# ==============================

print("\n" + "=" * 50)
print("DATA PREPARATION COMPLETE")
print("=" * 50)

print(f"X_train: {X_train_scaled.shape}")
print(f"y_train: {y_train.shape}")

print(f"X_validation: {X_val_scaled.shape}")
print(f"y_validation: {y_val.shape}")

print(f"X_test: {X_test_scaled.shape}")
print(f"y_test: {y_test.shape}")

print("\nScaled training mean:", np.mean(X_train_scaled))
print("Scaled training std: ", np.std(X_train_scaled))

print("\nSaved files in:")
print(OUTPUT_DIR)

print("\nFiles created:")
for file in OUTPUT_DIR.iterdir():
    print(" -", file.name)