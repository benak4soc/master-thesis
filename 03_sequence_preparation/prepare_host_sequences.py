# ============================================================
# CICEVSE2024 - Host Events: Split / Scale / Temporal Windows
# Master's Thesis
#
# Input :
#   processed_EVSE_B_Host/EVSE_B_Host_Events_Clean.csv
#
# Output:
#   processed_EVSE_B_Host/sequences/
#     train_windows.npz / val_windows.npz / test_windows.npz
#     scaler.joblib
#     feature_columns.txt
#     split_summary.txt
#
# Split strategy (thesis §11):
#   Session-aware split by (Scenario, Attack, State, interface)
#   so the same attack session does not leak across sets.
#   Scaler is fit on TRAINING rows only.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = os.path.join(
    "processed_EVSE_B_Host",
    "EVSE_B_Host_Events_Clean.csv"
)

OUTPUT_DIR = os.path.join(
    "processed_EVSE_B_Host",
    "sequences"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 10
STRIDE = 1
RANDOM_STATE = 42

# Fraction of sessions (not rows)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = remaining ~0.15

METADATA = [
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


print("=" * 70)
print("HOST EVENTS — SEQUENCE PREPARATION")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"Missing cleaned host file: {INPUT_CSV}\n"
        "Run hosteventspreprocessing.py first."
    )

df = pd.read_csv(INPUT_CSV)

feature_columns = [
    c
    for c in df.columns
    if c not in METADATA
]

print(f"\nRows            : {len(df)}")
print(f"Feature columns : {len(feature_columns)}")
print(f"Window size     : {WINDOW_SIZE}")
print(f"Stride          : {STRIDE}")


# ============================================================
# 2. DEFINE SESSIONS
# ============================================================

session_keys = [
    "Scenario",
    "Attack",
    "State",
    "interface",
]

df["session_id"] = (
    df[session_keys]
    .astype(str)
    .agg("||".join, axis=1)
)

# Chronological within each session
df = (
    df.sort_values(
        by=["session_id", "time"]
    )
    .reset_index(drop=True)
)

sessions = (
    df.groupby("session_id", sort=False)
    .agg(
        n_rows=("target", "size"),
        Attack_Group=("Attack_Group", "first"),
        target=("target", "first"),
        Scenario=("Scenario", "first"),
        Attack=("Attack", "first"),
        State=("State", "first"),
        interface=("interface", "first"),
    )
    .reset_index()
)

print(f"\nSessions found  : {len(sessions)}")
print(sessions.groupby("Attack_Group")["n_rows"].agg(["count", "sum"]))


# ============================================================
# 3. HYBRID SPLIT (session-aware + chronological for rare groups)
# ============================================================

# Sessions shorter than WINDOW_SIZE cannot form a window
usable = sessions[
    sessions["n_rows"] >= WINDOW_SIZE
].copy()

dropped = len(sessions) - len(usable)

print(
    f"\nSessions usable (>= {WINDOW_SIZE} rows): "
    f"{len(usable)}  (dropped {dropped})"
)

rng = np.random.default_rng(RANDOM_STATE)

# Row-level split labels
df["split"] = None

# Groups with few sessions cannot populate train/val/test
# without reuse. For those, use chronological within-session
# split (thesis §11 allows chronological when needed).
RARE_GROUP_MAX_SESSIONS = 3


def assign_group_sessions(group_df):
    """Session-level assignment for groups with enough sessions."""

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

    return (
        ids[:n_train],
        ids[n_train:n_train + n_val],
        ids[n_train + n_val:],
    )


def assign_chrono_within_session(session_rows):
    """
    Chronological 70/15/15 split inside one session.
    Returns boolean masks aligned to session_rows index order.
    """

    n = len(session_rows)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(1, int(n * VAL_FRAC))

    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    n_test = n - n_train - n_val

    if n_test < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    labels = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * n_test
    )

    # Pad / trim if rounding left a gap (should not happen)
    if len(labels) < n:
        labels += ["test"] * (n - len(labels))
    labels = labels[:n]

    return labels


print("\nSplit strategy by Attack_Group:")

train_session_ids = []
val_session_ids = []
test_session_ids = []
chrono_sessions = []

for group_name, group_df in usable.groupby(
    "Attack_Group",
    sort=False
):
    n_sess = len(group_df)

    if n_sess <= RARE_GROUP_MAX_SESSIONS:
        print(
            f"  {group_name:14s} chronological "
            f"within {n_sess} session(s)"
        )

        for sid in group_df["session_id"]:
            chrono_sessions.append(sid)
            mask = df["session_id"] == sid
            ordered_idx = (
                df.loc[mask]
                .sort_values("time")
                .index
            )
            labels = assign_chrono_within_session(
                ordered_idx
            )
            df.loc[ordered_idx, "split"] = labels

    else:
        tr, va, te = assign_group_sessions(group_df)
        train_session_ids.extend(tr)
        val_session_ids.extend(va)
        test_session_ids.extend(te)

        print(
            f"  {group_name:14s} session-aware "
            f"train={len(tr)} val={len(va)} test={len(te)}"
        )

# Apply session-level assignments
for sid in train_session_ids:
    df.loc[df["session_id"] == sid, "split"] = "train"

for sid in val_session_ids:
    df.loc[df["session_id"] == sid, "split"] = "val"

for sid in test_session_ids:
    df.loc[df["session_id"] == sid, "split"] = "test"

# Keep only rows with a split
df = df[df["split"].notna()].copy()

# Build summary tables from row assignments
def summarize_split(name):
    part = df[df["split"] == name]
    sess_count = part["session_id"].nunique()
    print(
        f"  {name:5s}: {sess_count:2d} sessions "
        f"(incl. chrono partial), "
        f"{len(part):5d} rows"
    )
    print(
        "         groups:",
        part["Attack_Group"]
        .value_counts()
        .to_dict()
    )
    print(
        "         binary:",
        part["target"]
        .value_counts()
        .to_dict()
    )


print("\nRow assignment totals:")
for name in ["train", "val", "test"]:
    summarize_split(name)

# Keep these for summary file compatibility
train_sessions = usable[
    usable["session_id"].isin(
        train_session_ids + chrono_sessions
    )
].copy()
train_sessions["note"] = train_sessions[
    "session_id"
].map(
    lambda s: (
        "chrono-partial"
        if s in chrono_sessions
        else "full-session"
    )
)

val_sessions = usable[
    usable["session_id"].isin(
        val_session_ids + chrono_sessions
    )
].copy()
val_sessions["note"] = val_sessions[
    "session_id"
].map(
    lambda s: (
        "chrono-partial"
        if s in chrono_sessions
        else "full-session"
    )
)

test_sessions = usable[
    usable["session_id"].isin(
        test_session_ids + chrono_sessions
    )
].copy()
test_sessions["note"] = test_sessions[
    "session_id"
].map(
    lambda s: (
        "chrono-partial"
        if s in chrono_sessions
        else "full-session"
    )
)


# ============================================================
# 4. FIT SCALER ON TRAIN ROWS ONLY
# ============================================================

train_mask = df["split"] == "train"

scaler = StandardScaler()
scaler.fit(
    df.loc[train_mask, feature_columns]
    .fillna(0.0)
)

scaled = scaler.transform(
    df[feature_columns].fillna(0.0)
)

df_scaled = df.copy()
df_scaled[feature_columns] = scaled

scaler_path = os.path.join(
    OUTPUT_DIR,
    "scaler.joblib"
)

joblib.dump(scaler, scaler_path)

print(f"\nScaler fit on train rows only.")
print(f"Saved: {scaler_path}")


# ============================================================
# 5. BUILD SLIDING WINDOWS PER SESSION
# ============================================================

def build_windows_for_split(split_name):
    """
    Sliding windows inside each session assigned to split_name.
    Label of a window = label of the LAST timestep (common IDS practice).
    """

    X_list = []
    y_bin_list = []
    y_mc_list = []
    y_af_list = []
    meta_list = []

    part = df_scaled[
        df_scaled["split"] == split_name
    ]

    for session_id, group in part.groupby(
        "session_id",
        sort=False
    ):
        group = group.sort_values("time")

        feats = group[feature_columns].to_numpy(
            dtype=np.float32
        )
        y_bin = group["target"].to_numpy(
            dtype=np.int64
        )
        y_mc = group["target_multiclass"].to_numpy(
            dtype=np.int64
        )
        y_af = group["target_attack_family"].to_numpy(
            dtype=np.float64
        )  # may contain NaN for benign

        n = len(group)

        if n < WINDOW_SIZE:
            continue

        for start in range(
            0,
            n - WINDOW_SIZE + 1,
            STRIDE
        ):
            end = start + WINDOW_SIZE
            last = end - 1

            X_list.append(
                feats[start:end]
            )
            y_bin_list.append(
                y_bin[last]
            )
            y_mc_list.append(
                y_mc[last]
            )
            y_af_list.append(
                y_af[last]
            )

            row = group.iloc[last]
            meta_list.append(
                {
                    "session_id": session_id,
                    "Attack_Group": row["Attack_Group"],
                    "Attack": row["Attack"],
                    "Scenario": row["Scenario"],
                    "State": row["State"],
                    "interface": row["interface"],
                    "time_end": float(row["time"]),
                }
            )

    if not X_list:
        raise RuntimeError(
            f"No windows created for split={split_name}. "
            "Reduce WINDOW_SIZE or check session lengths."
        )

    return {
        "X": np.stack(X_list, axis=0),
        "y_binary": np.asarray(y_bin_list, dtype=np.int64),
        "y_multiclass": np.asarray(y_mc_list, dtype=np.int64),
        "y_attack_family": np.asarray(y_af_list, dtype=np.float64),
        "meta": meta_list,
    }


datasets = {}

for split_name in ["train", "val", "test"]:
    data = build_windows_for_split(split_name)
    datasets[split_name] = data

    out_npz = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_windows.npz"
    )

    np.savez_compressed(
        out_npz,
        X=data["X"],
        y_binary=data["y_binary"],
        y_multiclass=data["y_multiclass"],
        y_attack_family=data["y_attack_family"],
    )

    meta_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_meta.json"
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            data["meta"],
            handle
        )

    print(
        f"\n{split_name.upper()} windows: "
        f"X={data['X'].shape}  "
        f"y_binary={np.bincount(data['y_binary'])}"
    )


# ============================================================
# 6. SAVE FEATURE LIST + SUMMARY
# ============================================================

feat_path = os.path.join(
    OUTPUT_DIR,
    "feature_columns.txt"
)

with open(feat_path, "w", encoding="utf-8") as handle:
    for column in feature_columns:
        handle.write(column + "\n")

config = {
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "n_features": len(feature_columns),
    "random_state": RANDOM_STATE,
    "train_frac": TRAIN_FRAC,
    "val_frac": VAL_FRAC,
    "n_train_sessions": int(len(train_sessions)),
    "n_val_sessions": int(len(val_sessions)),
    "n_test_sessions": int(len(test_sessions)),
    "shapes": {
        name: {
            "X": list(datasets[name]["X"].shape),
            "y_binary_counts": {
                "benign": int(
                    (datasets[name]["y_binary"] == 0).sum()
                ),
                "attack": int(
                    (datasets[name]["y_binary"] == 1).sum()
                ),
            },
        }
        for name in ["train", "val", "test"]
    },
}

config_path = os.path.join(
    OUTPUT_DIR,
    "sequence_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as handle:
    json.dump(config, handle, indent=2)

summary_path = os.path.join(
    OUTPUT_DIR,
    "split_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "Session-aware split summary\n"
    )
    handle.write(
        f"WINDOW_SIZE={WINDOW_SIZE}  STRIDE={STRIDE}\n\n"
    )

    for name, part in [
        ("TRAIN", train_sessions),
        ("VAL", val_sessions),
        ("TEST", test_sessions),
    ]:
        handle.write(f"=== {name} ===\n")
        handle.write(
            part[
                [
                    "session_id",
                    "Attack_Group",
                    "Attack",
                    "State",
                    "n_rows",
                ]
            ].to_string(index=False)
        )
        handle.write("\n\n")

print("\n" + "=" * 70)
print("SEQUENCE PREPARATION COMPLETE")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print(
    "Artifacts: *_windows.npz, scaler.joblib, "
    "feature_columns.txt, sequence_config.json"
)
print(
    "\nNext: train_lstm_baseline.py  (Pipeline A)"
)
