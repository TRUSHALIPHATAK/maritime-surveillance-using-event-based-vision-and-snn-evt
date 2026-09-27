import os
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)


TRACK_FILE = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5",
    "ship_045_tracks.csv"
)

GT_FILE = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_045_tracking_ground_truth.csv"
)


# ============================================================
# LOAD
# ============================================================

tracks = pd.read_csv(
    TRACK_FILE
)

gt = pd.read_csv(
    GT_FILE
)


track = tracks[
    tracks["track_id"] == 5
].copy()


gt = gt[
    (gt["target_id"] == 3)
    &
    (gt["visible"] == 1)
].copy()


# ============================================================
# GT CENTERS
# ============================================================

gt["gt_x"] = gt["center_x"]
gt["gt_y"] = gt["center_y"]


# ============================================================
# COMPARE
# ============================================================

rows = []


for _, t in track.iterrows():

    bin_id = int(
        t["bin_id"]
    )

    gt_bin = gt[
        gt["bin_id"] == bin_id
    ]


    if len(gt_bin) == 0:

        continue


    target = gt_bin.iloc[0]


    tx = float(
        t["x"]
    )

    ty = float(
        t["y"]
    )

    gx = float(
        target["gt_x"]
    )

    gy = float(
        target["gt_y"]
    )


    error = np.sqrt(
        (tx - gx) ** 2
        +
        (ty - gy) ** 2
    )


    rows.append(
        {
            "bin": bin_id,

            "track_x": tx,

            "track_y": ty,

            "gt_x": gx,

            "gt_y": gy,

            "error": error,

            "score":
                float(t["score"])
        }
    )


result = pd.DataFrame(
    rows
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("TRACK 5 vs GT TARGET 3")
print("=" * 80)

print(
    result.to_string(
        index=False,
        formatters={
            "track_x": "{:.2f}".format,
            "track_y": "{:.2f}".format,
            "gt_x": "{:.2f}".format,
            "gt_y": "{:.2f}".format,
            "error": "{:.2f}".format,
            "score": "{:.3f}".format
        }
    )
)


print()
print("=" * 80)
print("ERROR SUMMARY")
print("=" * 80)

print(
    f"Mean error   : "
    f"{result['error'].mean():.2f} px"
)

print(
    f"Median error : "
    f"{result['error'].median():.2f} px"
)

print(
    f"<= 4 px      : "
    f"{(result['error'] <= 4).sum()}/"
    f"{len(result)}"
)

print(
    f"<= 8 px      : "
    f"{(result['error'] <= 8).sum()}/"
    f"{len(result)}"
)

print(
    f"> 8 px       : "
    f"{(result['error'] > 8).sum()}/"
    f"{len(result)}"
)

print(
    f"> 15 px      : "
    f"{(result['error'] > 15).sum()}/"
    f"{len(result)}"
)


# ============================================================
# WORST BINS
# ============================================================

print()
print("=" * 80)
print("WORST 10 BINS")
print("=" * 80)

worst = result.sort_values(
    "error",
    ascending=False
).head(10)


print(
    worst.to_string(
        index=False,
        formatters={
            "track_x": "{:.2f}".format,
            "track_y": "{:.2f}".format,
            "gt_x": "{:.2f}".format,
            "gt_y": "{:.2f}".format,
            "error": "{:.2f}".format,
            "score": "{:.3f}".format
        }
    )
)