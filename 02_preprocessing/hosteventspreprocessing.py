# ============================================================
# CICEVSE2024 - EVSE-B HPC / Kernel Events Preprocessing
# Master's Thesis
#
# Dataset:
#   host events / EVSE-B-HPC-Kernel-Events-Combined.csv
#
# Targets:
#   Binary  : Benign = 0, Attack = 1
#   Multi   : Benign / Recon / DoS / Cryptojacking / Backdoor
#             (Attack_Group for supplementary multi-class LDA)
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = os.path.join(
    "host events",
    "EVSE-B-HPC-Kernel-Events-Combined.csv"
)

OUTPUT_DIR = "processed_EVSE_B_Host"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Attack name -> Attack_Group (aligned with power Attack-Group /
# thesis multi-class LDA: Recon / DoS / Cryptojacking / Backdoor)
ATTACK_TO_GROUP = {
    # benign
    "none": "Benign",
    "benign": "Benign",

    # host attacks
    "cryptojacking": "Cryptojacking",
    "backdoor": "Backdoor",

    # reconnaissance
    "aggressive-scan": "Recon",
    "os-fingerprinting": "Recon",
    "os-scan": "Recon",
    "port-scan": "Recon",
    "service-detection": "Recon",
    "serice-detection": "Recon",  # typo in source
    "syn-stealth": "Recon",
    "vuln-scan": "Recon",
    "vulnerability-scan": "Recon",

    # denial of service
    "icmp-flood": "DoS",
    "icmp-fragmentation": "DoS",
    "icmp-fragmentation_old": "DoS",
    "push-ack-flood": "DoS",
    "syn-flood": "DoS",
    "synonymous-ip-flood": "DoS",
    "tcp-flood": "DoS",
    "udp-flood": "DoS",
}


SCENARIO_TO_GROUP = {
    "benign": "Benign",
    "recon": "Recon",
    "dos": "DoS",
    "cryptojacking": "Cryptojacking",
    "backdoor": "Backdoor",
    "host-attack": "Cryptojacking",  # power naming fallback
}


MULTI_CLASS_ORDER = [
    "Benign",
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]


# ============================================================
# 1. LOAD
# ============================================================

print("=" * 70)
print("CICEVSE2024 EVSE-B HPC / KERNEL EVENTS PREPROCESSING")
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

print(
    "Raw shape:",
    df.shape
)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# 3. REMOVE CSV ARTIFACT COLUMNS
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


# ============================================================
# 4. DROP REPEATED HEADER / GARBAGE ROWS
# ============================================================

rows_before = len(df)

# Repeated header row (feature names written into values)
header_mask = (
    df["time"].astype(str).str.lower().eq("time")
    | df["Label"].astype(str).str.lower().eq("label")
)

# Feature-name leakage into label/meta fields
garbage_mask = (
    df["Label"].astype(str).str.contains(":", na=False)
    | df["Attack"].astype(str).str.contains(":", na=False)
    | df["Scenario"].astype(str).str.contains(":", na=False)
    | df["State"].astype(str).str.contains(":", na=False)
)

df = df[
    ~(header_mask | garbage_mask)
].copy()

print(
    "Header / garbage rows removed:",
    rows_before - len(df)
)


# ============================================================
# 5. NORMALIZE METADATA STRINGS
# ============================================================

for column in [
    "Label",
    "Attack",
    "Scenario",
    "State",
    "interface",
]:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )


# Fix known Attack typo
df["Attack"] = (
    df["Attack"]
    .str.lower()
    .replace(
        {
            "serice-detection": "service-detection",
            "icmp-fragmentation_old": "icmp-fragmentation",
        }
    )
)

df["Label"] = df["Label"].str.lower()
df["Scenario"] = df["Scenario"].str.strip()
df["State"] = df["State"].str.strip()
df["interface"] = (
    df["interface"]
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "", regex=True)
)


# ============================================================
# 6. KEEP ONLY VALID LABELED ROWS
# ============================================================

# Invalid Label values include nan / 0 / 0.0 / empty — drop them.
# Those trailing blocks have State/Attack/Scenario = 0 and are
# unsafe for supervised learning (shifted / unlabeled metadata).

valid_label_mask = df["Label"].isin(
    ["benign", "attack"]
)

invalid_count = (
    (~valid_label_mask).sum()
)

print(
    "\nRows with invalid Label dropped:",
    int(invalid_count)
)

df = df[valid_label_mask].copy()

print(
    "Rows after label filter:",
    len(df)
)

print("\nLabel distribution:")
print(
    df["Label"].value_counts(
        dropna=False
    )
)

print("\nScenario distribution:")
print(
    df["Scenario"].value_counts(
        dropna=False
    )
)


# ============================================================
# 7. DERIVE Attack_Group (multi-class LDA families)
# ============================================================

def map_attack_group(row):
    """
    Prefer Scenario when it already encodes the thesis families.
    Fall back to Attack name mapping.
    """

    scenario_key = (
        str(row["Scenario"])
        .strip()
        .lower()
    )

    if scenario_key in SCENARIO_TO_GROUP:
        return SCENARIO_TO_GROUP[scenario_key]

    attack_key = (
        str(row["Attack"])
        .strip()
        .lower()
    )

    if attack_key in ATTACK_TO_GROUP:
        return ATTACK_TO_GROUP[attack_key]

    # Last resort: if labeled attack but unknown type
    if str(row["Label"]).lower() == "attack":
        return "Unknown"

    return "Benign"


df["Attack_Group"] = df.apply(
    map_attack_group,
    axis=1
)

print("\nAttack_Group distribution:")
print(
    df["Attack_Group"].value_counts(
        dropna=False
    )
)

unknown_groups = df[
    df["Attack_Group"] == "Unknown"
]

if len(unknown_groups) > 0:
    print(
        "\nWARNING: Unknown Attack_Group rows:",
        len(unknown_groups)
    )
    print(
        unknown_groups[
            ["Attack", "Scenario", "Label"]
        ]
        .value_counts()
        .head(20)
    )


# ============================================================
# 8. BINARY TARGET + MULTI-CLASS TARGETS
# ============================================================

# Binary: thesis primary task
df["target"] = (
    df["Label"] == "attack"
).astype(
    np.int8
)

# Multi-class integer codes for LDA / classifiers
# 0=Benign, 1=Recon, 2=DoS, 3=Cryptojacking, 4=Backdoor
group_to_id = {
    name: idx
    for idx, name in enumerate(
        MULTI_CLASS_ORDER
    )
}

df["target_multiclass"] = (
    df["Attack_Group"]
    .map(group_to_id)
)

unmapped_mc = df[
    "target_multiclass"
].isna().sum()

if unmapped_mc > 0:
    print(
        "\nDropping rows with unmapped "
        "multi-class target:",
        int(unmapped_mc)
    )
    df = df[
        df["target_multiclass"].notna()
    ].copy()

df["target_multiclass"] = (
    df["target_multiclass"]
    .astype(np.int8)
)

# Attack-only multi-class (no Benign) — useful for
# supplementary multi-class LDA among attack families.
# Codes: 0=Recon, 1=DoS, 2=Cryptojacking, 3=Backdoor
attack_family_order = [
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]

attack_family_to_id = {
    name: idx
    for idx, name in enumerate(
        attack_family_order
    )
}

df["target_attack_family"] = (
    df["Attack_Group"]
    .map(attack_family_to_id)
)

print("\nBinary target:")
print(
    df["target"].value_counts()
)

print("\nMulti-class target (incl. Benign):")
print(
    df["Attack_Group"]
    .value_counts()
    .reindex(
        MULTI_CLASS_ORDER,
        fill_value=0
    )
)

print(
    "\nNote: Backdoor count in host HPC file:",
    int(
        (df["Attack_Group"] == "Backdoor").sum()
    )
)
print(
    "  (Backdoor appears in power-consumption "
    "telemetry; host file has Recon/DoS/"
    "Cryptojacking only.)"
)


# ============================================================
# 9. NORMALIZE STATE
# ============================================================

df["State"] = (
    df["State"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df.loc[
    df["State"].isin(
        ["charging"]
    ),
    "State"
] = "charging"

df.loc[
    df["State"].isin(
        ["idle"]
    ),
    "State"
] = "idle"

print("\nState distribution:")
print(
    df["State"].value_counts()
)


# ============================================================
# 10. TIME COLUMN
# ============================================================

df["time"] = pd.to_numeric(
    df["time"],
    errors="coerce"
)

invalid_time = df["time"].isna().sum()

print(
    "\nInvalid time values:",
    int(invalid_time)
)

if invalid_time > 0:
    df = df[df["time"].notna()].copy()

# Relative elapsed seconds within each scenario block;
# keep original for chronological ordering.
df["timestamp_original"] = df["time"]


# ============================================================
# 11. CONVERT FEATURES TO NUMERIC
# ============================================================

metadata_columns = [
    "target",
    "target_multiclass",
    "target_attack_family",
    "Label",
    "Attack",
    "Attack_Group",
    "Scenario",
    "State",
    "interface",
    "time",
    "timestamp_original",
]

feature_candidates = [
    c
    for c in df.columns
    if c not in metadata_columns
]

print(
    "\nConverting feature columns to numeric:",
    len(feature_candidates)
)

for column in feature_candidates:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 12. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_columns = [
    c
    for c in feature_candidates
    if df[c].isna().all()
]

print(
    "Completely empty feature columns:",
    len(empty_columns)
)

df.drop(
    columns=empty_columns,
    inplace=True
)

feature_candidates = [
    c
    for c in feature_candidates
    if c not in empty_columns
]


# ============================================================
# 13. REMOVE DUPLICATES
# ============================================================

duplicates_before = len(df)

df = df.drop_duplicates()

print(
    "Duplicate rows removed:",
    duplicates_before - len(df)
)


# ============================================================
# 14. REPLACE INFINITE VALUES
# ============================================================

df[feature_candidates] = (
    df[feature_candidates]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# 15. REMOVE FEATURES WITH >50% MISSING
# ============================================================

missing_ratio = (
    df[feature_candidates]
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

feature_candidates = [
    c
    for c in feature_candidates
    if c not in high_missing
]


# ============================================================
# 16. REMOVE ZERO-VARIANCE FEATURES
# ============================================================

# Fill NaN temporarily for variance check; real imputation
# for models must be fit on training split only later.
var_check = (
    df[feature_candidates]
    .fillna(0.0)
)

selector = VarianceThreshold(
    threshold=0.0
)

selector.fit(var_check)

kept_mask = selector.get_support()

zero_var_columns = [
    c
    for c, keep in zip(
        feature_candidates,
        kept_mask
    )
    if not keep
]

print(
    "Zero-variance features removed:",
    len(zero_var_columns)
)

feature_columns = [
    c
    for c, keep in zip(
        feature_candidates,
        kept_mask
    )
    if keep
]

print(
    "Remaining ML features:",
    len(feature_columns)
)


# ============================================================
# 17. TEMPORAL ORDERING
# ============================================================

# Order by scenario / attack session then elapsed time
# to support chronological / session-aware splits later.
sort_keys = [
    c
    for c in [
        "Scenario",
        "Attack",
        "State",
        "interface",
        "time",
    ]
    if c in df.columns
]

df = df.sort_values(
    by=sort_keys
).reset_index(
    drop=True
)


# ============================================================
# 18. SAVE CLEAN DATASET
# ============================================================

metadata_to_save = [
    c
    for c in [
        "timestamp_original",
        "time",
        "State",
        "interface",
        "Attack",
        "Scenario",
        "Attack_Group",
        "Label",
        "target",
        "target_multiclass",
        "target_attack_family",
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
    "EVSE_B_Host_Events_Clean.csv"
)

clean_df.to_csv(
    clean_path,
    index=False
)

print(
    "\nClean dataset saved:"
)
print(clean_path)


# Attack-family-only subset (no Benign) for multi-class LDA
attack_only = clean_df[
    clean_df["target"] == 1
].copy()

# Drop rows whose Attack_Group is not one of the 4 families
attack_only = attack_only[
    attack_only["Attack_Group"].isin(
        attack_family_order
    )
].copy()

attack_only_path = os.path.join(
    OUTPUT_DIR,
    "EVSE_B_Host_Events_AttackFamily.csv"
)

attack_only.to_csv(
    attack_only_path,
    index=False
)

print(
    "Attack-family dataset saved:"
)
print(attack_only_path)


# Label maps for later LDA / training scripts
label_map_path = os.path.join(
    OUTPUT_DIR,
    "label_maps.txt"
)

with open(
    label_map_path,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "Binary target\n"
    )
    handle.write(
        "  0 = Benign\n"
    )
    handle.write(
        "  1 = Attack\n\n"
    )

    handle.write(
        "target_multiclass "
        "(Attack_Group incl. Benign)\n"
    )
    for name, idx in group_to_id.items():
        handle.write(
            f"  {idx} = {name}\n"
        )

    handle.write(
        "\ntarget_attack_family "
        "(attack rows only; NaN for Benign)\n"
    )
    for name, idx in attack_family_to_id.items():
        handle.write(
            f"  {idx} = {name}\n"
        )

print(
    "Label maps saved:"
)
print(label_map_path)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(
    "\nFull clean shape:",
    clean_df.shape
)

print(
    "\nBinary Label:"
)
print(
    clean_df["Label"].value_counts()
)

print(
    "\nBinary target:"
)
print(
    clean_df["target"].value_counts()
)

print(
    "\nAttack_Group (multi-class LDA families):"
)
print(
    clean_df["Attack_Group"]
    .value_counts()
    .reindex(
        MULTI_CLASS_ORDER,
        fill_value=0
    )
)

print(
    "\nAttack-family-only shape:",
    attack_only.shape
)

print(
    "\nAttack-family Attack_Group:"
)
print(
    attack_only["Attack_Group"]
    .value_counts()
    .reindex(
        attack_family_order,
        fill_value=0
    )
)

print(
    "\nState:"
)
print(
    clean_df["State"].value_counts()
)

print(
    "\nNumber of ML features:",
    len(feature_columns)
)

print(
    "\nIMPORTANT (train/test later):"
)
print(
    "  Fit scaler / PCA / LDA on TRAINING data only."
)
print(
    "  Use scenario/session-aware or chronological split."
)
print(
    "  For multi-class LDA use Attack_Group / "
    "target_multiclass or attack-family subset."
)

print(
    "\nHost-events cleaning complete."
)
