import numpy as np
import pandas as pd


HEATMAP_045 = "../data/ship_045_center_heatmaps.npy"
HEATMAP_047 = "../data/ship_047_center_heatmaps.npy"

GT_045 = "../data/ship_045_event_gt_labels.csv"
GT_047 = "../data/ship_047_event_gt_labels.csv"


def find_peaks(heatmap, threshold=0.30, min_distance=5):

    work = heatmap.copy()

    peaks = []

    for _ in range(10):

        y, x = np.unravel_index(
            np.argmax(work),
            work.shape
        )

        value = work[y, x]

        if value < threshold:
            break

        peaks.append(
            (x, y, float(value))
        )

        # Suppress neighborhood around detected peak
        y0 = max(
            0,
            y - min_distance
        )

        y1 = min(
            work.shape[0],
            y + min_distance + 1
        )

        x0 = max(
            0,
            x - min_distance
        )

        x1 = min(
            work.shape[1],
            x + min_distance + 1
        )

        work[
            y0:y1,
            x0:x1
        ] = 0

    return peaks


def verify(
    heatmap_file,
    gt_file,
    sequence
):

    heatmaps = np.load(
        heatmap_file
    )

    gt = pd.read_csv(
        gt_file
    )

    print()
    print("=" * 60)
    print(sequence)
    print("=" * 60)

    print(
        "Heatmap shape:",
        heatmaps.shape
    )

    for bin_id in [0, 5, 10, 20, 30, 40, 49]:

        heatmap = heatmaps[
            bin_id
        ]

        peaks = find_peaks(
            heatmap
        )

        gt_rows = gt[
            (gt["bin_id"] == bin_id)
            &
            (gt["visible"] == 1)
        ]

        print()
        print(
            f"Bin {bin_id:02d}"
        )

        print(
            "Visible GT ships:",
            len(gt_rows)
        )

        print(
            "GT centers:"
        )

        for _, row in gt_rows.iterrows():

            cx = (
                float(row["x_128"])
                +
                float(row["w_128"]) / 2
            )

            cy = (
                float(row["y_128"])
                +
                float(row["h_128"]) / 2
            )

            print(
                f"  Target {int(row['target_id'])}: "
                f"({cx:.2f}, {cy:.2f})"
            )

        print(
            "Detected heatmap peaks:"
        )

        for x, y, value in peaks:

            print(
                f"  ({x:.2f}, {y:.2f}) "
                f"value={value:.3f}"
            )


verify(
    HEATMAP_045,
    GT_045,
    "SHIP 045"
)

verify(
    HEATMAP_047,
    GT_047,
    "SHIP 047"
)