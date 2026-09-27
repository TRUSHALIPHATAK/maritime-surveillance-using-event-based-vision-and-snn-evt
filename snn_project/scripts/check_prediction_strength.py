import os
import numpy as np
import pandas as pd


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

GT_045 = os.path.join(
    BASE_DIR,
    "data",
    "ship_045_event_gt_labels.csv"
)

GT_047 = os.path.join(
    BASE_DIR,
    "data",
    "ship_047_event_gt_labels.csv"
)


def get_gt_centers(gt, bin_id):

    rows = gt[
        (gt["bin_id"] == bin_id) &
        (gt["visible"] == 1)
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

        centers.append((cx, cy))

    return centers


def check_sequence(name, prediction_file, gt_file):

    predictions = np.load(prediction_file)

    gt = pd.read_csv(gt_file)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    values_at_gt = []

    for bin_id in range(predictions.shape[0]):

        heatmap = predictions[bin_id]

        centers = get_gt_centers(
            gt,
            bin_id
        )

        for cx, cy in centers:

            x = int(round(cx))
            y = int(round(cy))

            value = float(
                heatmap[y, x]
            )

            values_at_gt.append(value)

    values_at_gt = np.array(values_at_gt)

    print()
    print("Prediction statistics:")
    print("Min:", predictions.min())
    print("Max:", predictions.max())
    print("Mean:", predictions.mean())

    print()
    print("Prediction at GT ship centers:")

    print(
        "Min:",
        values_at_gt.min()
    )

    print(
        "Max:",
        values_at_gt.max()
    )

    print(
        "Mean:",
        values_at_gt.mean()
    )

    print(
        "Median:",
        np.median(values_at_gt)
    )

    print()

    for threshold in [
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70
    ]:

        percentage = (
            np.sum(values_at_gt >= threshold)
            /
            len(values_at_gt)
            *
            100
        )

        print(
            f"GT centers >= {threshold:.2f}: "
            f"{percentage:.2f}%"
        )


if __name__ == "__main__":

    check_sequence(
        "ship_045",
        PRED_045,
        GT_045
    )

    check_sequence(
        "ship_047",
        PRED_047,
        GT_047
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)