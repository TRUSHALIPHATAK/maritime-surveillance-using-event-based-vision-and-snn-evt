import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = ".."

PRED_045 = os.path.join(
    BASE_DIR,
    "results",
    "temporal_snn_inference",
    "ship_045_predicted_heatmaps.npy"
)

PRED_047 = os.path.join(
    BASE_DIR,
    "results",
    "temporal_snn_inference",
    "ship_047_predicted_heatmaps.npy"
)

GT_HEATMAP_045 = os.path.join(
    BASE_DIR,
    "data",
    "ship_045_center_heatmaps.npy"
)

GT_HEATMAP_047 = os.path.join(
    BASE_DIR,
    "data",
    "ship_047_center_heatmaps.npy"
)

GT_LABEL_045 = os.path.join(
    BASE_DIR,
    "data",
    "ship_045_event_gt_labels.csv"
)

GT_LABEL_047 = os.path.join(
    BASE_DIR,
    "data",
    "ship_047_event_gt_labels.csv"
)

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "results",
    "temporal_snn_visualization"
)


# ============================================================
# SETTINGS
# ============================================================

SELECTED_BINS = [0, 5, 10, 20, 30, 40, 49]

THRESHOLD = 0.30

MIN_DISTANCE = 5

MAX_PEAKS = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_BASE, exist_ok=True)


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

            score = float(heatmap[y, x])

            if score < threshold:
                continue

            y0 = max(0, y - min_distance)
            y1 = min(height, y + min_distance + 1)

            x0 = max(0, x - min_distance)
            x1 = min(width, x + min_distance + 1)

            neighborhood = heatmap[
                y0:y1,
                x0:x1
            ]

            if score >= neighborhood.max():

                candidates.append(
                    (
                        score,
                        x,
                        y
                    )
                )

    candidates.sort(
        key=lambda p: p[0],
        reverse=True
    )

    selected = []

    for score, x, y in candidates:

        too_close = False

        for _, px, py in selected:

            distance = (
                (x - px) ** 2 +
                (y - py) ** 2
            ) ** 0.5

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
# GET GT CENTERS
# ============================================================

def get_gt_centers(gt, bin_id):

    rows = gt[
        (gt["bin_id"] == bin_id) &
        (gt["visible"] == 1)
    ]

    centers = []

    for _, row in rows.iterrows():

        center_x = (
            float(row["x_128"]) +
            float(row["w_128"]) / 2.0
        )

        center_y = (
            float(row["y_128"]) +
            float(row["h_128"]) / 2.0
        )

        centers.append(
            (
                center_x,
                center_y
            )
        )

    return centers


# ============================================================
# DRAW CENTERS
# ============================================================

def draw_gt_centers(ax, centers):

    for i, (x, y) in enumerate(centers):

        ax.scatter(
            x,
            y,
            marker="x",
            s=100,
            linewidths=2
        )

        ax.text(
            x + 2,
            y - 2,
            f"GT{i + 1}",
            fontsize=8
        )


def draw_pred_centers(ax, predictions):

    for i, (score, x, y) in enumerate(predictions):

        ax.scatter(
            x,
            y,
            marker="o",
            facecolors="none",
            s=100,
            linewidths=2
        )

        ax.text(
            x + 2,
            y + 3,
            f"P{i + 1} ({score:.2f})",
            fontsize=8
        )


# ============================================================
# VISUALIZE ONE SEQUENCE
# ============================================================

def visualize_sequence(
    sequence_name,
    prediction_file,
    gt_heatmap_file,
    gt_label_file
):

    print()
    print("=" * 70)
    print(f"Visualizing: {sequence_name}")
    print("=" * 70)

    predictions = np.load(prediction_file)

    gt_heatmaps = np.load(gt_heatmap_file)

    gt = pd.read_csv(gt_label_file)

    print("Prediction shape:", predictions.shape)

    print("GT heatmap shape:", gt_heatmaps.shape)

    print("GT labels:", gt.shape)

    output_dir = os.path.join(
        OUTPUT_BASE,
        sequence_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for bin_id in SELECTED_BINS:

        pred_heatmap = predictions[bin_id]

        gt_heatmap = gt_heatmaps[bin_id]

        pred_peaks = detect_peaks(
            pred_heatmap
        )

        gt_centers = get_gt_centers(
            gt,
            bin_id
        )

        print()
        print(f"Bin {bin_id:02d}")

        print(
            "GT centers:",
            [
                (round(x, 2), round(y, 2))
                for x, y in gt_centers
            ]
        )

        print(
            "Predicted:",
            [
                (
                    x,
                    y,
                    round(score, 3)
                )
                for score, x, y in pred_peaks
            ]
        )

        # ----------------------------------------------------
        # CREATE FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        # ----------------------------------------------------
        # GT HEATMAP
        # ----------------------------------------------------

        axes[0].imshow(
            gt_heatmap,
            origin="upper"
        )

        draw_gt_centers(
            axes[0],
            gt_centers
        )

        axes[0].set_title(
            f"GT Center Heatmap\nBin {bin_id}"
        )

        axes[0].set_xlim(
            0,
            127
        )

        axes[0].set_ylim(
            127,
            0
        )

        # ----------------------------------------------------
        # PREDICTED HEATMAP
        # ----------------------------------------------------

        axes[1].imshow(
            pred_heatmap,
            origin="upper"
        )

        draw_pred_centers(
            axes[1],
            pred_peaks
        )

        axes[1].set_title(
            f"SNN Predicted Heatmap\nBin {bin_id}"
        )

        axes[1].set_xlim(
            0,
            127
        )

        axes[1].set_ylim(
            127,
            0
        )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        axes[2].imshow(
            pred_heatmap,
            origin="upper"
        )

        draw_gt_centers(
            axes[2],
            gt_centers
        )

        draw_pred_centers(
            axes[2],
            pred_peaks
        )

        axes[2].set_title(
            f"Overlay\nBin {bin_id}"
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
        # SAVE
        # ----------------------------------------------------

        output_file = os.path.join(
            output_dir,
            f"bin_{bin_id:02d}.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Saved:",
            output_file
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    visualize_sequence(
        "ship_045",
        PRED_045,
        GT_HEATMAP_045,
        GT_LABEL_045
    )

    visualize_sequence(
        "ship_047",
        PRED_047,
        GT_HEATMAP_047,
        GT_LABEL_047
    )

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Images saved to:"
    )

    print(
        os.path.abspath(OUTPUT_BASE)
    )