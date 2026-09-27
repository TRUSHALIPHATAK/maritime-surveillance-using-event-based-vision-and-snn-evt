import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# TEMPORAL SNN CENTER DETECTOR V3 - VISUALIZATION
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V3 - VISUALIZATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "data"
)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3"
)

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_visualization_v3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FILES
# ============================================================

PRED_045 = os.path.join(
    RESULT_DIR,
    "ship_045_predicted_heatmaps_v3.npy"
)

PRED_047 = os.path.join(
    RESULT_DIR,
    "ship_047_predicted_heatmaps_v3.npy"
)

GT_045 = os.path.join(
    DATA_DIR,
    "ship_045_event_gt_labels.csv"
)

GT_047 = os.path.join(
    DATA_DIR,
    "ship_047_event_gt_labels.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

THRESHOLD = 0.50
MIN_DISTANCE = 6
MAX_PEAKS = 10

SELECTED_BINS = [
    0,
    5,
    10,
    20,
    30,
    40,
    49
]


# ============================================================
# FILE CHECK
# ============================================================

print("\nChecking files...")

files = {
    "045 prediction": PRED_045,
    "047 prediction": PRED_047,
    "045 GT": GT_045,
    "047 GT": GT_047
}

for name, path in files.items():

    if os.path.exists(path):
        print(f"{name:<20}: FOUND")

    else:
        print(f"{name:<20}: NOT FOUND")
        print(path)
        raise FileNotFoundError(path)


# ============================================================
# PEAK DETECTION
# ============================================================

def detect_peaks(
    heatmap,
    threshold=THRESHOLD,
    min_distance=MIN_DISTANCE,
    max_peaks=MAX_PEAKS
):

    candidates = []

    height, width = heatmap.shape

    for y in range(height):

        for x in range(width):

            score = float(
                heatmap[y, x]
            )

            if score < threshold:
                continue

            y0 = max(
                0,
                y - min_distance
            )

            y1 = min(
                height,
                y + min_distance + 1
            )

            x0 = max(
                0,
                x - min_distance
            )

            x1 = min(
                width,
                x + min_distance + 1
            )

            neighborhood = heatmap[
                y0:y1,
                x0:x1
            ]

            if score >= float(
                neighborhood.max()
            ):

                candidates.append(
                    (
                        score,
                        x,
                        y
                    )
                )

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
                (
                    score,
                    x,
                    y
                )
            )

        if len(selected) >= max_peaks:
            break

    return selected


# ============================================================
# LOAD GT CENTERS
# ============================================================

def load_gt_centers(gt_file):

    df = pd.read_csv(
        gt_file
    )

    centers_by_bin = {}

    for _, row in df.iterrows():

        if int(row["visible"]) == 0:
            continue

        bin_id = int(
            row["bin_id"]
        )

        x = float(
            row["x_128"]
        )

        y = float(
            row["y_128"]
        )

        w = float(
            row["w_128"]
        )

        h = float(
            row["h_128"]
        )

        center_x = x + w / 2.0
        center_y = y + h / 2.0

        if bin_id not in centers_by_bin:

            centers_by_bin[bin_id] = []

        centers_by_bin[bin_id].append(
            (
                center_x,
                center_y
            )
        )

    return centers_by_bin


# ============================================================
# VISUALIZE ONE SEQUENCE
# ============================================================

def visualize_sequence(
    sequence_name,
    prediction_file,
    gt_file
):

    print("\n" + "=" * 70)
    print(
        f"VISUALIZING {sequence_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    predictions = np.load(
        prediction_file
    )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt_by_bin = load_gt_centers(
        gt_file
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    sequence_output = os.path.join(
        OUTPUT_DIR,
        sequence_name
    )

    os.makedirs(
        sequence_output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process selected bins
    # --------------------------------------------------------

    for bin_id in SELECTED_BINS:

        heatmap = predictions[
            bin_id
        ]

        gt_centers = gt_by_bin.get(
            bin_id,
            []
        )

        detected_peaks = detect_peaks(
            heatmap
        )

        print(
            f"\nBin {bin_id:02d}"
        )

        print(
            f"GT centers: "
            f"{len(gt_centers)}"
        )

        print(
            f"Predictions: "
            f"{len(detected_peaks)}"
        )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6)
        )

        # ====================================================
        # 1. GT HEATMAP
        # ====================================================

        axes[0].imshow(
            heatmap,
            cmap="gray"
        )

        # We will overlay GT centers
        for gx, gy in gt_centers:

            axes[0].scatter(
                gx,
                gy,
                marker="x",
                s=100,
                linewidths=2,
                label="GT"
            )

            axes[0].text(
                gx + 2,
                gy - 2,
                f"({gx:.1f},{gy:.1f})",
                fontsize=7
            )

        axes[0].set_title(
            f"{sequence_name} - Bin {bin_id:02d}\n"
            f"Ground Truth"
        )

        axes[0].set_xlim(
            0,
            127
        )

        axes[0].set_ylim(
            127,
            0
        )

        # ====================================================
        # 2. PREDICTED HEATMAP
        # ====================================================

        axes[1].imshow(
            heatmap,
            cmap="hot",
            vmin=0,
            vmax=1
        )

        for score, x, y in detected_peaks:

            axes[1].scatter(
                x,
                y,
                marker="o",
                facecolors="none",
                edgecolors="cyan",
                s=100,
                linewidths=2
            )

            axes[1].text(
                x + 2,
                y - 2,
                f"{score:.2f}",
                fontsize=8
            )

        axes[1].set_title(
            f"{sequence_name} - Bin {bin_id:02d}\n"
            f"V3 Prediction"
        )

        axes[1].set_xlim(
            0,
            127
        )

        axes[1].set_ylim(
            127,
            0
        )

        # ====================================================
        # 3. OVERLAY
        # ====================================================

        axes[2].imshow(
            heatmap,
            cmap="hot",
            vmin=0,
            vmax=1
        )

        # Ground truth = X
        for gx, gy in gt_centers:

            axes[2].scatter(
                gx,
                gy,
                marker="x",
                s=120,
                linewidths=3,
                label="Ground Truth"
            )

        # Prediction = circle
        for score, x, y in detected_peaks:

            axes[2].scatter(
                x,
                y,
                marker="o",
                facecolors="none",
                edgecolors="cyan",
                s=120,
                linewidths=2,
                label="Prediction"
            )

            axes[2].text(
                x + 2,
                y - 2,
                f"{score:.2f}",
                fontsize=8
            )

        axes[2].set_title(
            f"{sequence_name} - Bin {bin_id:02d}\n"
            f"GT + V3 Prediction"
        )

        axes[2].set_xlim(
            0,
            127
        )

        axes[2].set_ylim(
            127,
            0
        )

        # ----------------------------------------------------
        # Overall figure
        # ----------------------------------------------------

        fig.suptitle(
            f"Temporal SNN V3 Center Detection | "
            f"{sequence_name} | Bin {bin_id:02d}",
            fontsize=14
        )

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

    print(
        f"\nVisualization directory:"
    )

    print(
        sequence_output
    )


# ============================================================
# RUN
# ============================================================

visualize_sequence(
    "ship_045",
    PRED_045,
    GT_045
)

visualize_sequence(
    "ship_047",
    PRED_047,
    GT_047
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V3 VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_DIR
)