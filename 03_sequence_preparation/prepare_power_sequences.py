# ============================================================
# CICEVSE2024 - EVSE-B Power: sequences for Pipelines A–E
#
# Input : processed_EVSE_B_Power/EVSE_B_Power_Clean.csv
# Output: processed_EVSE_B_Power/sequences/
#
# Includes Backdoor (absent from host HPC / network EVSE-A).
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
    "processed_EVSE_B_Power",
    "EVSE_B_Power_Clean.csv",
)
OUTPUT_DIR = os.path.join(
    "processed_EVSE_B_Power",
    "sequences",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 10
STRIDE = 1
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
MAX_PER_GROUP = 6000  # balance + speed; all groups capped

FEATURE_COLUMNS = [
    "shunt_voltage",
    "bus_voltage_V",
    "current_mA",
    "power_mW",
]

MULTI_CLASS_ORDER = [
    "Benign",
    "Recon",
    "DoS",
    "Cryptojacking",
    "Backdoor",
]


print("=" * 70)
print("POWER EVSE-B — SEQUENCE PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)
df["timestamp_parsed"] = pd.to_datetime(
    df["timestamp_parsed"],
    errors="coerce",
)
print(f"Raw rows: {len(df)}")
print(df["Attack_Group"].value_counts())

# Cap per Attack_Group
rng = np.random.default_rng(RANDOM_STATE)
parts = []
for gname, part in df.groupby("Attack_Group"):
    n = len(part)
    keep = min(n, MAX_PER_GROUP)
    if keep < n:
        idx = rng.choice(part.index.to_numpy(), size=keep, replace=False)
        part = part.loc[idx]
    parts.append(part)
    print(f"  {gname}: kept {len(part)} / {n}")

df = (
    pd.concat(parts, axis=0)
    .sort_values(
        ["Attack_Group", "Attack", "State", "interface", "timestamp_parsed"]
    )
    .reset_index(drop=True)
)
print(f"After subsample: {len(df)}")
print(df["target"].value_counts())

df["session_id"] = (
    df[["Attack_Group", "Attack", "State", "interface"]]
    .astype(str)
    .agg("||".join, axis=1)
)

sessions = (
    df.groupby("session_id", sort=False)
    .agg(
        n_rows=("target", "size"),
        Attack_Group=("Attack_Group", "first"),
        target=("target", "first"),
    )
    .reset_index()
)
print(f"Sessions: {len(sessions)}")
print(sessions.groupby("Attack_Group")["n_rows"].agg(["count", "sum"]))

usable = sessions[sessions["n_rows"] >= WINDOW_SIZE].copy()
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


train_ids, val_ids, test_ids = [], [], []

print("\nSplit strategy:")
for gname, gdf in usable.groupby("Attack_Group", sort=False):
    if len(gdf) <= RARE_MAX:
        print(f"  {gname:14s} chronological within {len(gdf)} session(s)")
        for sid in gdf["session_id"]:
            mask = df["session_id"] == sid
            ordered = df.loc[mask].sort_values("timestamp_parsed").index
            df.loc[ordered, "split"] = chrono_labels(len(ordered))
    else:
        tr, va, te = assign_group_sessions(gdf)
        train_ids.extend(tr)
        val_ids.extend(va)
        test_ids.extend(te)
        print(
            f"  {gname:14s} session-aware "
            f"train={len(tr)} val={len(va)} test={len(te)}"
        )

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

# Scaler train-only
train_mask = df["split"] == "train"
scaler = StandardScaler()
scaler.fit(df.loc[train_mask, FEATURE_COLUMNS].fillna(0.0))
df[FEATURE_COLUMNS] = scaler.transform(
    df[FEATURE_COLUMNS].fillna(0.0)
)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))


def build_windows(split_name):
    X_list, yb, ym, yaf, meta = [], [], [], [], []
    part = df[df["split"] == split_name]
    for sid, g in part.groupby("session_id", sort=False):
        g = g.sort_values("timestamp_parsed")
        feats = g[FEATURE_COLUMNS].to_numpy(np.float32)
        y_bin = g["target"].to_numpy(np.int64)
        y_mc = g["target_multiclass"].to_numpy(np.int64)
        y_af = g["target_attack_family"].to_numpy(np.float64)
        n = len(g)
        if n < WINDOW_SIZE:
            continue
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            last = end - 1
            X_list.append(feats[start:end])
            yb.append(y_bin[last])
            ym.append(y_mc[last])
            yaf.append(y_af[last])
            row = g.iloc[last]
            meta.append(
                {
                    "session_id": sid,
                    "Attack_Group": row["Attack_Group"],
                    "Attack": row["Attack"],
                    "State": row["State"],
                    "interface": row["interface"],
                }
            )
    if not X_list:
        raise RuntimeError(f"No windows for {split_name}")
    return {
        "X": np.stack(X_list),
        "y_binary": np.asarray(yb, np.int64),
        "y_multiclass": np.asarray(ym, np.int64),
        "y_attack_family": np.asarray(yaf, np.float64),
        "meta": meta,
    }


datasets = {}
for split_name in ["train", "val", "test"]:
    data = build_windows(split_name)
    datasets[split_name] = data
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"{split_name}_windows.npz"),
        X=data["X"],
        y_binary=data["y_binary"],
        y_multiclass=data["y_multiclass"],
        y_attack_family=data["y_attack_family"],
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
    for c in FEATURE_COLUMNS:
        handle.write(c + "\n")

config = {
    "modality": "power_EVSE_B",
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "n_features": len(FEATURE_COLUMNS),
    "max_per_group": MAX_PER_GROUP,
    "features": FEATURE_COLUMNS,
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
    "note": "Includes Backdoor; full 5-class Attack_Group available for LDA.",
}

with open(
    os.path.join(OUTPUT_DIR, "sequence_config.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(config, handle, indent=2)

print("\nPower sequence preparation complete.")
print(f"Output: {OUTPUT_DIR}")
