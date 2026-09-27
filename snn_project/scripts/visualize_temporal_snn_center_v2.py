import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SNN_PROJECT_DIR)

# Voxel grids
VOXEL_045 = os.path.join(
    PROJECT_ROOT,
    "ship_045_event_dataset",
    "3_voxel_grid",
    "ship_045_voxel_grid.npy"
)

VOXEL_047 = os.path.join(
    SNN_PROJECT_DIR,
    "ship_047_event_dataset",
    "3_voxel_grid",
    "ship_047_voxel_grid.npy"
)

# GT heatmaps
HEATMAP_045 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_045_center_heatmaps.npy"
)

HEATMAP_047 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_047_center_heatmaps.npy"
)

# GT label CSVs
GT_045 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_045_event_gt_labels.csv"
)

GT_047 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_047_event_gt_labels.csv"
)

# V2 predictions
PRED_045 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v2",
    "ship_045_predicted_heatmaps_v2.npy"
)

PRED_047 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v2",
    "ship_047_predicted_heatmaps_v2.npy"
)

# Output
OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_visualization_v2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SELECTED_BINS = [0, 5, 10, 20, 30, 40, 49]

THRESHOLD = 0.50
MIN_DISTANCE = 6
MAX_PEAKS = 10


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V2 - VISUALIZATION")
print("=" * 70)

files = {
    "045 voxel": VOXEL_045,
    "047 voxel": VOXEL_047,
    "045 GT heatmap": HEATMAP_045,
    "047 GT heatmap": HEATMAP_047,
    "045 GT labels": GT_045,
    "047 GT labels": GT_047,
    "045 prediction": PRED_045,
    "047 prediction": PRED_047,
}

for name, path in files.items():
    status = "FOUND" if os.path.exists(path) else "NOT FOUND"
    print(f"{name:<20}: {status}")

missing = [path for path in files.values() if not os.path.exists(path)]

if missing:
    print("\nERROR: Required files are missing.")
    for path in missing:
        if not os.path.exists(path):
            print(path)
    raise FileNotFoundError("Missing required files.")


# ============================================================
# PEAK DETECTION
# ============================================================

def detect_peaks(
    heatmap,
    threshold=THRESHOLD,
    min_distance=MIN_DISTANCE,
    max_peaks=MAX_PEAKS
):
    """
    Detect local maxima using only NumPy.
    No scipy required.
    """

    candidates = []

    height, width = heatmap.shape

    for y in range(height):

        for x in range(width):

            score = float(heatmap[y, x])

            if score < threshold:
                continue

            y0 = max(0, y - min_distance)
            y1 = min(height, y + min_distance + 1)

            x0 = max(0, x - min_distance)
            x1 = min(width, x + min_distance + 1)

            neighborhood = heatmap[y0:y1, x0:x1]

            if score >= neighborhood.max():

                candidates.append(
                    (score, x, y)
                )

    # Highest scores first
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    for score, x, y in candidates:

        too_close = False

        for _, px, py in selected:

            distance = np.sqrt(
                (x - px) ** 2 +
                (y - py) ** 2
            )

            if distance < min_distance:

                too_close = True
                break

        if not too_close:

            selected.append(
                (score, x, y)
            )

        if len(selected) >= max_peaks:
            break

    return selected


# ============================================================
# LOAD GT CENTERS
# ============================================================

def get_gt_centers(gt_csv, bin_id):

    df = pd.read_csv(gt_csv)

    rows = df[
        (df["bin_id"] == bin_id) &
        (df["visible"] == 1)
    ]

    centers = []

    for _, row in rows.iterrows():

        cx = (
            float(row["x_128"]) +
            float(row["w_128"]) / 2
        )

        cy = (
            float(row["y_128"]) +
            float(row["h_128"]) / 2
        )

        centers.append(
            (cx, cy)
        )

    return centers


# ============================================================
# VISUALIZE ONE SEQUENCE
# ============================================================

def visualize_sequence(
    sequence_name,
    voxel_file,
    gt_heatmap_file,
    gt_csv,
    prediction_file
):

    print("\n" + "=" * 70)
    print(f"PROCESSING {sequence_name}")
    print("=" * 70)

    voxel = np.load(voxel_file)
    gt_heatmap = np.load(gt_heatmap_file)
    prediction = np.load(prediction_file)

    print("Voxel shape       :", voxel.shape)
    print("GT heatmap shape  :", gt_heatmap.shape)
    print("Prediction shape  :", prediction.shape)

    sequence_output = os.path.join(
        OUTPUT_DIR,
        sequence_name
    )

    os.makedirs(
        sequence_output,
        exist_ok=True
    )

    for bin_id in SELECTED_BINS:

        print(f"\nCreating bin {bin_id:02d}")

        # ----------------------------------------------------
        # Event representation
        # ----------------------------------------------------

        positive_events = voxel[
            bin_id,
            0
        ]

        negative_events = voxel[
            bin_id,
            1
        ]

        event_image = (
            positive_events +
            negative_events
        )

        # Normalize only for visualization
        max_event = event_image.max()

        if max_event > 0:
            event_image = event_image / max_event

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt = gt_heatmap[bin_id]

        gt_centers = get_gt_centers(
            gt_csv,
            bin_id
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred = prediction[bin_id]

        predicted_peaks = detect_peaks(
            pred
        )

        print(
            f"GT centers       : {len(gt_centers)}"
        )

        print(
            f"Predicted peaks  : {len(predicted_peaks)}"
        )

        for score, x, y in predicted_peaks:

            print(
                f"  Prediction: "
                f"x={x}, y={y}, "
                f"score={score:.3f}"
            )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 5)
        )

        # ====================================================
        # 1. EVENT INPUT
        # ====================================================

        axes[0].imshow(
            event_image,
            cmap="gray"
        )

        axes[0].set_title(
            f"{sequence_name} - Bin {bin_id:02d}\n"
            "Event Input"
        )

        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # ====================================================
        # 2. GROUND TRUTH HEATMAP
        # ====================================================

        axes[1].imshow(
            gt,
            cmap="hot",
            vmin=0,
            vmax=1
        )

        for cx, cy in gt_centers:

            axes[1].plot(
                cx,
                cy,
                marker="x",
                markersize=10,
                markeredgewidth=2
            )

        axes[1].set_title(
            "Ground Truth Centers"
        )

        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

        # ====================================================
        # 3. V2 PREDICTION HEATMAP
        # ====================================================

        axes[2].imshow(
            pred,
            cmap="hot",
            vmin=0,
            vmax=1
        )

        for score, x, y in predicted_peaks:

            axes[2].plot(
                x,
                y,
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgewidth=2
            )

            axes[2].text(
                x + 2,
                y + 2,
                f"{score:.2f}",
                fontsize=8
            )

        axes[2].set_title(
            f"V2 Prediction\n"
            f"Threshold={THRESHOLD}"
        )

        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")

        # ====================================================
        # 4. OVERLAY
        # ====================================================

        axes[3].imshow(
            event_image,
            cmap="gray"
        )

        # GT = X
        for cx, cy in gt_centers:

            axes[3].plot(
                cx,
                cy,
                marker="x",
                markersize=11,
                markeredgewidth=2
            )

        # Prediction = circle
        for score, x, y in predicted_peaks:

            axes[3].plot(
                x,
                y,
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgewidth=2
            )

            axes[3].text(
                x + 2,
                y + 2,
                f"{score:.2f}",
                fontsize=8
            )

        axes[3].set_title(
            "GT vs V2 Prediction\n"
            "X = GT, O = Prediction"
        )

        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")

        # ====================================================
        # SAVE
        # ====================================================

        plt.tight_layout()

        output_file = os.path.join(
            sequence_output,
            f"bin_{bin_id:02d}.png"
        )

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )


# ============================================================
# RUN
# ============================================================

visualize_sequence(
    "ship_045",
    VOXEL_045,
    HEATMAP_045,
    GT_045,
    PRED_045
)

visualize_sequence(
    "ship_047",
    VOXEL_047,
    HEATMAP_047,
    GT_047,
    PRED_047
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated folders:")
print(
    os.path.join(
        OUTPUT_DIR,
        "ship_045"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "ship_047"
    )
)