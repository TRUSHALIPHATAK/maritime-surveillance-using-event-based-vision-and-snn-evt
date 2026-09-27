import os
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

PRED_045 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_045_predicted_heatmaps_v3.npy"
)

GT_045 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_045_tracking_ground_truth.csv"
)


predictions = np.load(PRED_045)
gt = pd.read_csv(GT_045)

print()
print("=" * 70)
print("SNN SCORE AT EACH 045 GROUND-TRUTH SHIP CENTER")
print("=" * 70)


for bin_id in [5, 10, 20, 30, 40, 49]:

    heatmap = predictions[bin_id]

    rows = gt[
        (gt["bin_id"] == bin_id)
        &
        (gt["visible"] == 1)
    ]

    print()
    print(f"BIN {bin_id}")
    print("-" * 50)

    for _, row in rows.iterrows():

        x = float(row["center_x"])
        y = float(row["center_y"])

        xi = int(round(x))
        yi = int(round(y))

        score = float(
            heatmap[yi, xi]
        )

        print(
            f"Target {int(row['target_id'])}: "
            f"GT=({x:.2f},{y:.2f}) "
            f"score={score:.4f}"
        )