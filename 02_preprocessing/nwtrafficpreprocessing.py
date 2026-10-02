# ============================================================
# CICEVSE2024 - EVSE-A Network Traffic Preprocessing
# Master's Thesis
#
# Dataset:
#   EVSE-A Charging + Idle
#
# Target:
#   Benign = 0
#   Attack = 1
# ============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = os.path.join("network traffic", "EVSE A")

OUTPUT_DIR = "processed_EVSE_A"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. FIND ALL CSV FILES
# ============================================================

csv_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "**",
            "*.csv"
        ),
        recursive=True
    )
)

print("=" * 70)
print("CICEVSE2024 EVSE-A NETWORK TRAFFIC PREPROCESSING")
print("=" * 70)

print(
    f"\nCSV files found: {len(csv_files)}"
)

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found inside {INPUT_DIR}"
    )

for file in csv_files:
    print(
        " -",
        os.path.relpath(
            file,
            INPUT_DIR
        )
    )


# ============================================================
# 2. READ FILES
# ============================================================

frames = []

for file in csv_files:

    print(
        "\nReading:",
        os.path.basename(file)
    )

    temp = pd.read_csv(
        file,
        low_memory=False
    )

    # Preserve origin
    temp["source_file"] = os.path.basename(
        file
    )

    # Preserve directory/state
    relative_path = os.path.relpath(
        file,
        INPUT_DIR
    )

    if relative_path.lower().startswith(
        "charging"
    ):
        temp["dataset_state"] = "charging"

    elif relative_path.lower().startswith(
        "idle"
    ):
        temp["dataset_state"] = "idle"

    else:
        temp["dataset_state"] = "unknown"

    frames.append(temp)


# ============================================================
# 3. COMBINE
# ============================================================

df = pd.concat(
    frames,
    ignore_index=True,
    sort=False
)

print("\nCombined dataset:")
print(
    "Rows   :",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
)


# ============================================================
# 5. REMOVE CSV ARTIFACT COLUMNS
# ============================================================

artifact_columns = [
    c
    for c in df.columns
    if c.lower().startswith(
        "unnamed:"
    )
]

print(
    "\nRemoving artifact columns:",
    len(artifact_columns)
)

df.drop(
    columns=artifact_columns,
    inplace=True,
    errors="ignore"
)


# ============================================================
# 6. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_columns = [
    c
    for c in df.columns
    if df[c].isna().all()
]

print(
    "Completely empty columns:",
    len(empty_columns)
)

df.drop(
    columns=empty_columns,
    inplace=True
)


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

duplicates_before = len(df)

df = df.drop_duplicates()

duplicates_removed = (
    duplicates_before -
    len(df)
)

print(
    "Duplicate rows removed:",
    duplicates_removed
)


# ============================================================
# 8. IDENTIFY / DERIVE LABEL COLUMN
# ============================================================

# These CSVs have no Label column; derive from filename:
#   *benign*  -> benign
#   otherwise -> attack

if "Label" not in df.columns:

    print(
        "\nNo Label column found. "
        "Deriving labels from source filenames."
    )

    source_lower = (
        df["source_file"]
        .astype(str)
        .str.lower()
    )

    df["Label"] = np.where(
        source_lower.str.contains(
            "benign"
        ),
        "benign",
        "attack"
    )

    # Attack / scenario name from filename
    # e.g. EVSE-A-charging-syn-flood.csv -> syn-flood
    def attack_from_filename(name):

        name = (
            str(name)
            .lower()
            .replace(".csv", "")
        )

        for prefix in [
            "evse-a-charging-",
            "evse-a-idle-",
            "evse-a-",
        ]:
            if name.startswith(prefix):
                name = name[
                    len(prefix):
                ]
                break

        if "benign" in name:
            return "benign"

        return name

    df["Attack"] = (
        df["source_file"]
        .map(attack_from_filename)
    )

else:

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


print("\nOriginal label distribution:")

print(
    df["Label"].value_counts(
        dropna=False
    )
)


# ============================================================
# 9. KEEP ONLY BENIGN / ATTACK
# ============================================================

valid_labels = [
    "benign",
    "attack"
]

unexpected_labels = sorted(
    set(df["Label"].unique())
    - set(valid_labels)
)

if unexpected_labels:

    print(
        "\nUnexpected labels:",
        unexpected_labels
    )

    df = df[
        df["Label"].isin(
            valid_labels
        )
    ].copy()


# ============================================================
# 10. CREATE BINARY TARGET
# ============================================================

df["target"] = (
    df["Label"] == "attack"
).astype(
    np.int8
)


# ============================================================
# 11. DETERMINE STATE
# ============================================================

if "State" in df.columns:

    df["State"] = (
        df["State"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

else:

    # Use folder-derived state
    df["State"] = (
        df["dataset_state"]
    )


print("\nState distribution:")

print(
    df["State"].value_counts()
)


# ============================================================
# 12. TIME COLUMN
# ============================================================

# Possible timestamp column names
possible_time_columns = [
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "bidirectional_first_seen_ms",
    "src2dst_first_seen_ms",
]

time_column = None

for column in possible_time_columns:

    if column in df.columns:

        time_column = column
        break


if time_column is not None:

    print(
        "\nTime column:",
        time_column
    )

    # Preserve original
    df["timestamp_original"] = (
        df[time_column]
    )

    # Convert timestamp
    df["timestamp_parsed"] = pd.to_datetime(
        df[time_column],
        errors="coerce"
    )

    invalid_time = (
        df["timestamp_parsed"]
        .isna()
        .sum()
    )

    print(
        "Invalid timestamps:",
        invalid_time
    )

else:

    print(
        "\nWARNING: No timestamp column found."
    )


# ============================================================
# 13. NUMERIC FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# 14. COLUMNS THAT MUST NOT BE FEATURES
# ============================================================

metadata_columns = [
    "target",
    "Label",
    "Attack",
    "Scenario",
    "State",
    "interface",
    "source_file",
    "dataset_state",
    "timestamp_original",
    "timestamp_parsed"
]


# ============================================================
# 15. IDENTIFY POSSIBLE IDENTIFIER COLUMNS
# ============================================================

identifier_keywords = [
    "flow id",
    "flow_id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",
    "src_oui",
    "dst_oui",
]

# Exact-match identifier / non-feature columns (NFStream exports)
exact_identifier_columns = [
    "id",
    "expiration_id",
]

identifier_columns = []

for column in df.columns:

    column_lower = (
        column.lower()
        .strip()
    )

    if column_lower in exact_identifier_columns:

        identifier_columns.append(
            column
        )
        continue

    for keyword in identifier_keywords:

        if keyword in column_lower:

            identifier_columns.append(
                column
            )

            break


identifier_columns = list(
    dict.fromkeys(
        identifier_columns
    )
)


print(
    "\nPotential identifier columns:"
)

for column in identifier_columns:

    print(
        " -",
        column
    )


# ============================================================
# 16. BUILD INITIAL FEATURE LIST
# ============================================================

feature_columns = []

for column in numeric_columns:

    if column in metadata_columns:
        continue

    if column in identifier_columns:
        continue

    feature_columns.append(
        column
    )


print(
    "\nInitial numerical feature count:",
    len(feature_columns)
)


# ============================================================
# 17. REPLACE INFINITE VALUES
# ============================================================

df[feature_columns] = (
    df[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# 18. REMOVE FEATURES WITH >50% MISSING
# ============================================================

missing_ratio = (
    df[feature_columns]
    .isna()
    .mean()
)

high_missing = (
    missing_ratio[
        missing_ratio > 0.50
    ]
    .index
    .tolist()
)

print(
    "\nFeatures with >50% missing:",
    len(high_missing)
)

feature_columns = [
    c
    for c in feature_columns
    if c not in high_missing
]


# ============================================================
# 19. TEMPORAL ORDERING
# ============================================================

if "timestamp_parsed" in df.columns:

    df = df.sort_values(
        by=[
            "source_file",
            "timestamp_parsed"
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# 20. SAVE CLEAN DATASET
# ============================================================

metadata_to_save = [
    c
    for c in [
        "timestamp_original",
        "timestamp_parsed",
        "State",
        "Attack",
        "Scenario",
        "Label",
        "target",
        "interface",
        "source_file"
    ]
    if c in df.columns
]

final_columns = (
    metadata_to_save
    + feature_columns
)

final_columns = list(
    dict.fromkeys(
        final_columns
    )
)

clean_df = df[
    final_columns
].copy()


clean_path = os.path.join(
    OUTPUT_DIR,
    "EVSE_A_Network_Clean.csv"
)

clean_df.to_csv(
    clean_path,
    index=False
)

print(
    "\nClean dataset saved:"
)

print(
    clean_path
)


# ============================================================
# 21. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(
    "\nShape:",
    clean_df.shape
)

print(
    "\nLabels:"
)

print(
    clean_df["Label"]
    .value_counts()
)

print(
    "\nBinary target:"
)

print(
    clean_df["target"]
    .value_counts()
)

print(
    "\nState:"
)

print(
    clean_df["State"]
    .value_counts()
)

if "Attack" in clean_df.columns:

    print(
        "\nAttack types:"
    )

    print(
        clean_df["Attack"]
        .value_counts()
        .head(30)
    )

print(
    "\nNumber of ML features:",
    len(feature_columns)
)

print(
    "\nCleaning complete."
)