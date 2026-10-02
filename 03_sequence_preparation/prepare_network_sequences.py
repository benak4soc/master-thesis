# ============================================================
# CICEVSE2024 - EVSE-A Network Traffic: sequences for Pipelines A–E
#
# Input : processed_EVSE_A/EVSE_A_Network_Clean.csv
# Output: processed_EVSE_A/sequences/
#
# Notes:
#   - Extreme class imbalance (82 benign vs ~548k attack)
#   - Undersample attacks (keep ALL benign)
#   - Window size 5 (benign files are small)
#   - Session = source_file
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


INPUT_CSV = os.path.join(
    "processed_EVSE_A",
    "EVSE_A_Network_Clean.csv",
)

OUTPUT_DIR = os.path.join(
    "processed_EVSE_A",
    "sequences",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 5
STRIDE = 1
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Cap attack rows per Attack_Group after keeping all benign
MAX_ATTACK_PER_GROUP = 4000

METADATA = [
    "timestamp_original",
    "timestamp_parsed",
    "State",
    "Attack",
    "Attack_Group",
    "Label",
    "target",
    "target_multiclass",
    "source_file",
]

ATTACK_TO_GROUP = {
    "benign": "Benign",
    # Recon / scans
    "portscan": "Recon",
    "vulnerability-scan": "Recon",
    "aggressive-scan": "Recon",
    "os-fingerprinting": "Recon",
    "syn-stealth": "Recon",
    "syn-stealth-scan": "Recon",
    "service-detection": "Recon",
    "slowloris-scan": "Recon",
    # DoS / floods
    "syn-flood": "DoS",
    "synonymous-ip": "DoS",
    "tcp-flood": "DoS",
    "push-ack-flood": "DoS",
    "udp-flood": "DoS",
    "icmp-fragmentation": "DoS",
    "icmp-flood": "DoS",
}

MULTI_CLASS_ORDER = [
    "Benign",
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]


print("=" * 70)
print("NETWORK EVSE-A — SEQUENCE PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)
print(f"Raw rows: {len(df)}")

# Attack_Group
atk = df["Attack"].astype(str).str.lower().str.strip()
df["Attack_Group"] = atk.map(ATTACK_TO_GROUP).fillna("Unknown")

unknown = int((df["Attack_Group"] == "Unknown").sum())
if unknown:
    print("WARNING Unknown Attack_Group rows:", unknown)
    print(df.loc[df["Attack_Group"] == "Unknown", "Attack"].value_counts().head())

group_to_id = {n: i for i, n in enumerate(MULTI_CLASS_ORDER)}
df["target_multiclass"] = df["Attack_Group"].map(group_to_id)
df = df[df["target_multiclass"].notna()].copy()
df["target_multiclass"] = df["target_multiclass"].astype(np.int8)

print("\nAttack_Group (full):")
print(df["Attack_Group"].value_counts())

# ------------------------------------------------------------
# Undersample attacks (keep all benign)
# ------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)

benign = df[df["target"] == 0]
attacks = df[df["target"] == 1]

sampled_parts = [benign]
print(f"\nKeeping all benign: {len(benign)}")

for group_name, part in attacks.groupby("Attack_Group"):
    n = len(part)
    keep = min(n, MAX_ATTACK_PER_GROUP)
    if keep < n:
        idx = rng.choice(part.index.to_numpy(), size=keep, replace=False)
        part = part.loc[idx]
    sampled_parts.append(part)
    print(f"  {group_name}: kept {len(part)} / {n}")

df = (
    pd.concat(sampled_parts, axis=0)
    .sort_values(["source_file", "timestamp_parsed", "bidirectional_first_seen_ms"], kind="mergesort")
    .reset_index(drop=True)
)

print(f"\nAfter undersample: {len(df)}")
print(df["target"].value_counts())
print(df["Attack_Group"].value_counts())

feature_columns = [
    c for c in df.columns if c not in METADATA
]
# drop non-numeric leftovers if any
for c in list(feature_columns):
    if not np.issubdtype(df[c].dtype, np.number):
        # try coerce
        df[c] = pd.to_numeric(df[c], errors="coerce")

feature_columns = [
    c for c in feature_columns
    if np.issubdtype(df[c].dtype, np.number)
]

print(f"Features: {len(feature_columns)}")

# ------------------------------------------------------------
# Sessions = source_file
# ------------------------------------------------------------
df["session_id"] = df["source_file"].astype(str)

sessions = (
    df.groupby("session_id", sort=False)
    .agg(
        n_rows=("target", "size"),
        Attack_Group=("Attack_Group", "first"),
        target=("target", "first"),
        State=("State", "first"),
    )
    .reset_index()
)

print(f"Sessions: {len(sessions)}")
print(sessions.groupby("Attack_Group")["n_rows"].agg(["count", "sum"]))

usable = sessions[sessions["n_rows"] >= WINDOW_SIZE].copy()
print(f"Usable sessions (>= {WINDOW_SIZE}): {len(usable)}")

# ------------------------------------------------------------
# Hybrid split (same idea as host)
# ------------------------------------------------------------
df["split"] = None
RARE_MAX = 3


def assign_group_sessions(group_df):
    ids = group_df["session_id"].tolist()
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * TRAIN_FRAC)))
    n_val = max(1, int(round(n * VAL_FRAC)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = 1
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def chrono_labels(n):
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    if len(labels) < n:
        labels += ["test"] * (n - len(labels))
    return labels[:n]


train_ids, val_ids, test_ids, chrono_ids = [], [], [], []

print("\nSplit strategy:")
for gname, gdf in usable.groupby("Attack_Group", sort=False):
    if len(gdf) <= RARE_MAX:
        print(f"  {gname:12s} chronological within {len(gdf)} session(s)")
        for sid in gdf["session_id"]:
            chrono_ids.append(sid)
            mask = df["session_id"] == sid
            ordered = df.loc[mask].sort_values(
                ["timestamp_parsed", "bidirectional_first_seen_ms"]
            ).index
            df.loc[ordered, "split"] = chrono_labels(len(ordered))
    else:
        tr, va, te = assign_group_sessions(gdf)
        train_ids.extend(tr)
        val_ids.extend(va)
        test_ids.extend(te)
        print(f"  {gname:12s} session-aware train={len(tr)} val={len(va)} test={len(te)}")

for sid in train_ids:
    df.loc[df["session_id"] == sid, "split"] = "train"
for sid in val_ids:
    df.loc[df["session_id"] == sid, "split"] = "val"
for sid in test_ids:
    df.loc[df["session_id"] == sid, "split"] = "test"

df = df[df["split"].notna()].copy()

for name in ["train", "val", "test"]:
    part = df[df["split"] == name]
    print(
        f"  {name}: rows={len(part)}  "
        f"binary={part['target'].value_counts().to_dict()}  "
        f"groups={part['Attack_Group'].value_counts().to_dict()}"
    )

# ------------------------------------------------------------
# Scaler on train only
# ------------------------------------------------------------
train_mask = df["split"] == "train"
scaler = StandardScaler()
scaler.fit(df.loc[train_mask, feature_columns].fillna(0.0))
df[feature_columns] = scaler.transform(
    df[feature_columns].fillna(0.0)
)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

# ------------------------------------------------------------
# Windows
# ------------------------------------------------------------

def build_windows(split_name):
    X_list, yb, ym, meta = [], [], [], []
    part = df[df["split"] == split_name]
    for sid, g in part.groupby("session_id", sort=False):
        g = g.sort_values(
            ["timestamp_parsed", "bidirectional_first_seen_ms"]
        )
        feats = g[feature_columns].to_numpy(np.float32)
        y_bin = g["target"].to_numpy(np.int64)
        y_mc = g["target_multiclass"].to_numpy(np.int64)
        n = len(g)
        if n < WINDOW_SIZE:
            continue
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            last = end - 1
            X_list.append(feats[start:end])
            yb.append(y_bin[last])
            ym.append(y_mc[last])
            row = g.iloc[last]
            meta.append(
                {
                    "session_id": sid,
                    "Attack_Group": row["Attack_Group"],
                    "Attack": row["Attack"],
                    "State": row["State"],
                    "source_file": row["source_file"],
                }
            )
    if not X_list:
        raise RuntimeError(f"No windows for {split_name}")
    return {
        "X": np.stack(X_list),
        "y_binary": np.asarray(yb, np.int64),
        "y_multiclass": np.asarray(ym, np.int64),
        "meta": meta,
    }


datasets = {}
for split_name in ["train", "val", "test"]:
    data = build_windows(split_name)
    datasets[split_name] = data
    # attack family placeholder for compatibility with DR script
    y_af = np.full(len(data["y_binary"]), np.nan, dtype=np.float64)
    # map non-benign multiclass to attack-family ids if desired later
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"{split_name}_windows.npz"),
        X=data["X"],
        y_binary=data["y_binary"],
        y_multiclass=data["y_multiclass"],
        y_attack_family=y_af,
    )
    with open(
        os.path.join(OUTPUT_DIR, f"{split_name}_meta.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(data["meta"], handle)
    print(
        f"{split_name.upper()} windows: {data['X'].shape}  "
        f"y={np.bincount(data['y_binary'])}"
    )

with open(
    os.path.join(OUTPUT_DIR, "feature_columns.txt"),
    "w",
    encoding="utf-8",
) as handle:
    for c in feature_columns:
        handle.write(c + "\n")

config = {
    "modality": "network_EVSE_A",
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "n_features": len(feature_columns),
    "max_attack_per_group": MAX_ATTACK_PER_GROUP,
    "random_state": RANDOM_STATE,
    "shapes": {
        name: {
            "X": list(datasets[name]["X"].shape),
            "y_binary_counts": {
                "benign": int((datasets[name]["y_binary"] == 0).sum()),
                "attack": int((datasets[name]["y_binary"] == 1).sum()),
            },
        }
        for name in ["train", "val", "test"]
    },
    "note": (
        "Attacks undersampled; all benign kept. "
        "Scores may still be optimistic due to imbalance / window overlap."
    ),
}

with open(
    os.path.join(OUTPUT_DIR, "sequence_config.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(config, handle, indent=2)

print("\nNetwork sequence preparation complete.")
print(f"Output: {OUTPUT_DIR}")
