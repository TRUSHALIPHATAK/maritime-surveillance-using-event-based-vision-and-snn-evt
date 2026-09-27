import os
import numpy as np
import pandas as pd


# ============================================================
# V3 THRESHOLD SWEEP
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V3 - THRESHOLD SWEEP")
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

OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "v3_threshold_sweep.csv"
)


# ============================================================
# FIXED PARAMETERS
# ============================================================

MIN_DISTANCE = 6
MAX_PEAKS = 5
MATCH_DISTANCE = 8.0

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.62,
    0.64,
    0.65,
    0.66,
    0.68,
    0.70,
    0.72,
    0.74,
    0.75
]


# ============================================================
# PEAK DETECTOR
# ============================================================

def detect_peaks(
    heatmap,
    threshold,
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
        key=lambda p: p[0],
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
# LOAD GT
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

        x = float(row["x_128"])
        y = float(row["y_128"])

        w = float(row["w_128"])
        h = float(row["h_128"])

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
# MATCHING
# ============================================================

def match_predictions(
    predictions,
    gt_centers
):

    pairs = []

    for pred_index, (
        score,
        px,
        py
    ) in enumerate(predictions):

        for gt_index, (
            gx,
            gy
        ) in enumerate(gt_centers):

            distance = np.sqrt(
                (px - gx) ** 2 +
                (py - gy) ** 2
            )

            if distance <= MATCH_DISTANCE:

                pairs.append(
                    (
                        distance,
                        pred_index,
                        gt_index
                    )
                )

    pairs.sort(
        key=lambda x: x[0]
    )

    used_predictions = set()
    used_gt = set()

    errors = []

    for (
        distance,
        pred_index,
        gt_index
    ) in pairs:

        if pred_index in used_predictions:
            continue

        if gt_index in used_gt:
            continue

        used_predictions.add(
            pred_index
        )

        used_gt.add(
            gt_index
        )

        errors.append(
            distance
        )

    return errors


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(
    prediction_file,
    gt_file,
    threshold
):

    heatmaps = np.load(
        prediction_file
    )

    gt_by_bin = load_gt_centers(
        gt_file
    )

    total_gt = 0
    total_predictions = 0
    total_matches = 0

    all_errors = []

    for bin_id in range(
        heatmaps.shape[0]
    ):

        gt_centers = gt_by_bin.get(
            bin_id,
            []
        )

        predictions = detect_peaks(
            heatmaps[bin_id],
            threshold
        )

        errors = match_predictions(
            predictions,
            gt_centers
        )

        total_gt += len(
            gt_centers
        )

        total_predictions += len(
            predictions
        )

        total_matches += len(
            errors
        )

        all_errors.extend(
            errors
        )

    precision = (
        total_matches /
        total_predictions
        if total_predictions > 0
        else 0
    )

    recall = (
        total_matches /
        total_gt
        if total_gt > 0
        else 0
    )

    mean_error = (
        np.mean(all_errors)
        if len(all_errors) > 0
        else np.nan
    )

    median_error = (
        np.median(all_errors)
        if len(all_errors) > 0
        else np.nan
    )

    # F1 score
    if precision + recall > 0:

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:

        f1 = 0

    return {
        "threshold": threshold,
        "gt": total_gt,
        "predictions": total_predictions,
        "matches": total_matches,
        "false_positives":
            total_predictions - total_matches,
        "false_negatives":
            total_gt - total_matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error": mean_error,
        "median_error": median_error
    }


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

print(
    f"045 prediction: "
    f"{os.path.exists(PRED_045)}"
)

print(
    f"047 prediction: "
    f"{os.path.exists(PRED_047)}"
)

print(
    f"045 GT: "
    f"{os.path.exists(GT_045)}"
)

print(
    f"047 GT: "
    f"{os.path.exists(GT_047)}"
)


# ============================================================
# RUN SWEEP
# ============================================================

all_results = []

for threshold in THRESHOLDS:

    print(
        "\n" +
        "-" * 70
    )

    print(
        f"THRESHOLD = {threshold:.2f}"
    )

    result_045 = evaluate_threshold(
        PRED_045,
        GT_045,
        threshold
    )

    result_045["sequence"] = "ship_045"

    result_047 = evaluate_threshold(
        PRED_047,
        GT_047,
        threshold
    )

    result_047["sequence"] = "ship_047"

    all_results.append(
        result_045
    )

    all_results.append(
        result_047
    )

    print(
        f"045: "
        f"Precision={result_045['precision']*100:.2f}% | "
        f"Recall={result_045['recall']*100:.2f}% | "
        f"F1={result_045['f1']:.4f} | "
        f"Error={result_045['mean_error']:.2f}px"
    )

    print(
        f"047: "
        f"Precision={result_047['precision']*100:.2f}% | "
        f"Recall={result_047['recall']*100:.2f}% | "
        f"F1={result_047['f1']:.4f} | "
        f"Error={result_047['mean_error']:.2f}px"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df[
    [
        "sequence",
        "threshold",
        "gt",
        "predictions",
        "matches",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_error",
        "median_error"
    ]
]

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL THRESHOLD COMPARISON")
print("=" * 70)

for sequence in [
    "ship_045",
    "ship_047"
]:

    print(
        f"\n{sequence}"
    )

    seq = results_df[
        results_df["sequence"] == sequence
    ].copy()

    display_df = seq[
        [
            "threshold",
            "predictions",
            "matches",
            "precision",
            "recall",
            "f1",
            "mean_error"
        ]
    ].copy()

    display_df["precision"] *= 100
    display_df["recall"] *= 100

    print(
        display_df.to_string(
            index=False,
            formatters={
                "precision": "{:.2f}".format,
                "recall": "{:.2f}".format,
                "f1": "{:.4f}".format,
                "mean_error": "{:.2f}".format
            }
        )
    )


# ============================================================
# BEST F1 FOR EACH SEQUENCE
# ============================================================

print("\n" + "=" * 70)
print("BEST F1 RESULTS")
print("=" * 70)

for sequence in [
    "ship_045",
    "ship_047"
]:

    seq = results_df[
        results_df["sequence"] == sequence
    ]

    best = seq.loc[
        seq["f1"].idxmax()
    ]

    print(
        f"\n{sequence}"
    )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{best['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )

    print(
        f"Mean error: "
        f"{best['mean_error']:.2f} px"
    )


print("\n" + "=" * 70)
print("THRESHOLD SWEEP COMPLETE")
print("=" * 70)

print(
    f"\nSaved:"
)

print(
    OUTPUT_FILE
)