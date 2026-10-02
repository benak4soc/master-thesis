# ============================================================
# CICEVSE2024 - EVSE-B Power Consumption Preprocessing
# Master's Thesis
#
# Dataset:
#   power consumption / EVSE-B-PowerCombined.csv
#
# Targets:
#   Binary  : Benign = 0, Attack = 1
#   Multi   : Benign / Recon / DoS / Cryptojacking / Backdoor
#             (Attack_Group for supplementary multi-class LDA)
#
# Note:
#   Power telemetry has few numeric features (not the main DR
#   source). It provides cyber-physical context and contains
#   Backdoor samples missing from the host HPC file.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = os.path.join(
    "power consumption",
    "EVSE-B-PowerCombined.csv"
)

OUTPUT_DIR = "processed_EVSE_B_Power"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Source Attack-Group / Attack -> thesis families
GROUP_MAP = {
    "none": "Benign",
    "benign": "Benign",
    "recon": "Recon",
    "dos": "DoS",
    "cryptojacking": "Cryptojacking",
    "backdoor": "Backdoor",
    # power file uses host-attack for Backdoor + cryptojacking
    "host-attack": None,  # resolve via Attack name
}

ATTACK_TO_GROUP = {
    "none": "Benign",
    "vuln-scan": "Recon",
    "syn-stealth": "Recon",
    "tcp-flood": "DoS",
    "syn-flood": "DoS",
    "cryptojacking": "Cryptojacking",
    "backdoor": "Backdoor",
}

MULTI_CLASS_ORDER = [
    "Benign",
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]


print("=" * 70)
print("CICEVSE2024 EVSE-B POWER CONSUMPTION PREPROCESSING")
print("=" * 70)

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

print(f"\nReading: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Raw shape:", df.shape)


# ============================================================
# 1. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# 2. REMOVE ARTIFACT / EMPTY COLUMNS
# ============================================================

artifact_columns = [
    c
    for c in df.columns
    if c.lower().startswith("unnamed:")
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
# 3. NORMALIZE METADATA
# ============================================================

for column in [
    "Label",
    "Attack",
    "Attack-Group",
    "State",
    "interface",
]:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

df["Label"] = df["Label"].str.lower()
df["Attack"] = df["Attack"].str.lower()
df["Attack-Group"] = (
    df["Attack-Group"]
    .str.lower()
)
df["State"] = df["State"].str.lower()
df["interface"] = (
    df["interface"]
    .str.lower()
    .str.replace(r"\s+", "", regex=True)
)


# ============================================================
# 4. KEEP VALID LABELS
# ============================================================

valid_mask = df["Label"].isin(
    ["benign", "attack"]
)

print(
    "\nInvalid Label rows dropped:",
    int((~valid_mask).sum())
)

df = df[valid_mask].copy()

print("Rows after label filter:", len(df))
print("\nLabel distribution:")
print(df["Label"].value_counts())


# ============================================================
# 5. DERIVE Attack_Group (thesis multi-class families)
# ============================================================

def map_attack_group(row):
    attack = str(row["Attack"]).lower()
    group = str(row["Attack-Group"]).lower()

    # Prefer fine-grained Attack when available
    if attack in ATTACK_TO_GROUP:
        return ATTACK_TO_GROUP[attack]

    mapped = GROUP_MAP.get(group)
    if mapped is not None:
        return mapped

    # host-attack without recognized Attack name
    if group == "host-attack":
        if "backdoor" in attack:
            return "Backdoor"
        if "crypto" in attack:
            return "Cryptojacking"
        return "Unknown"

    if str(row["Label"]).lower() == "attack":
        return "Unknown"

    return "Benign"


df["Attack_Group"] = df.apply(
    map_attack_group,
    axis=1
)

# Keep original column as Attack_Group_source for traceability
df["Attack_Group_source"] = df["Attack-Group"]

print("\nAttack_Group distribution:")
print(
    df["Attack_Group"]
    .value_counts()
    .reindex(MULTI_CLASS_ORDER, fill_value=0)
)

unknown = int(
    (df["Attack_Group"] == "Unknown").sum()
)
if unknown:
    print(
        "\nWARNING: Unknown Attack_Group rows:",
        unknown
    )


# ============================================================
# 6. BINARY + MULTI-CLASS TARGETS
# ============================================================

df["target"] = (
    df["Label"] == "attack"
).astype(np.int8)

group_to_id = {
    name: idx
    for idx, name in enumerate(MULTI_CLASS_ORDER)
}

df["target_multiclass"] = (
    df["Attack_Group"]
    .map(group_to_id)
)

unmapped = int(
    df["target_multiclass"].isna().sum()
)
if unmapped:
    print(
        "Dropping unmapped multi-class rows:",
        unmapped
    )
    df = df[
        df["target_multiclass"].notna()
    ].copy()

df["target_multiclass"] = (
    df["target_multiclass"]
    .astype(np.int8)
)

attack_family_order = [
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]
attack_family_to_id = {
    name: idx
    for idx, name in enumerate(attack_family_order)
}

df["target_attack_family"] = (
    df["Attack_Group"]
    .map(attack_family_to_id)
)

print("\nBinary target:")
print(df["target"].value_counts())


# ============================================================
# 7. PARSE TIME
# ============================================================

df["timestamp_parsed"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

invalid_time = int(
    df["timestamp_parsed"].isna().sum()
)
print("\nInvalid timestamps:", invalid_time)

df["timestamp_original"] = df["time"]

# Sort for chronological / session-aware splits later
df = df.sort_values(
    by=[
        "Attack_Group",
        "Attack",
        "State",
        "interface",
        "timestamp_parsed",
    ]
).reset_index(drop=True)


# ============================================================
# 8. NUMERIC FEATURES
# ============================================================

feature_columns = [
    "shunt_voltage",
    "bus_voltage_V",
    "current_mA",
    "power_mW",
]

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[feature_columns] = (
    df[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
)

missing_ratio = df[feature_columns].isna().mean()
print("\nFeature missing ratios:")
print(missing_ratio)

# Drop rows with missing features
before = len(df)
df = df.dropna(subset=feature_columns).copy()
print(
    "Rows dropped for missing features:",
    before - len(df)
)


# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

before = len(df)
df = df.drop_duplicates()
print(
    "Duplicate rows removed:",
    before - len(df)
)


# ============================================================
# 10. SAVE
# ============================================================

metadata_to_save = [
    "timestamp_original",
    "timestamp_parsed",
    "State",
    "interface",
    "Attack",
    "Attack_Group_source",
    "Attack_Group",
    "Label",
    "target",
    "target_multiclass",
    "target_attack_family",
]

final_columns = metadata_to_save + feature_columns
clean_df = df[final_columns].copy()

clean_path = os.path.join(
    OUTPUT_DIR,
    "EVSE_B_Power_Clean.csv"
)
clean_df.to_csv(clean_path, index=False)

attack_only = clean_df[
    clean_df["target"] == 1
].copy()
attack_only = attack_only[
    attack_only["Attack_Group"].isin(
        attack_family_order
    )
].copy()

attack_path = os.path.join(
    OUTPUT_DIR,
    "EVSE_B_Power_AttackFamily.csv"
)
attack_only.to_csv(attack_path, index=False)

label_map_path = os.path.join(
    OUTPUT_DIR,
    "label_maps.txt"
)

with open(
    label_map_path,
    "w",
    encoding="utf-8"
) as handle:
    handle.write("Binary target\n")
    handle.write("  0 = Benign\n")
    handle.write("  1 = Attack\n\n")
    handle.write(
        "target_multiclass (Attack_Group)\n"
    )
    for name, idx in group_to_id.items():
        handle.write(f"  {idx} = {name}\n")
    handle.write(
        "\ntarget_attack_family "
        "(attack rows only)\n"
    )
    for name, idx in attack_family_to_id.items():
        handle.write(f"  {idx} = {name}\n")


print("\n" + "=" * 70)
print("FINAL POWER DATASET SUMMARY")
print("=" * 70)
print("\nClean shape:", clean_df.shape)
print("\nBinary Label:")
print(clean_df["Label"].value_counts())
print("\nAttack_Group:")
print(
    clean_df["Attack_Group"]
    .value_counts()
    .reindex(MULTI_CLASS_ORDER, fill_value=0)
)
print("\nState:")
print(clean_df["State"].value_counts())
print("\nML features:", feature_columns)
print("\nSaved:")
print(" ", clean_path)
print(" ", attack_path)
print(" ", label_map_path)
print(
    "\nNote: Power has Backdoor samples "
    f"({int((clean_df['Attack_Group']=='Backdoor').sum())}); "
    "host HPC file does not."
)
print("\nPower-consumption cleaning complete.")
