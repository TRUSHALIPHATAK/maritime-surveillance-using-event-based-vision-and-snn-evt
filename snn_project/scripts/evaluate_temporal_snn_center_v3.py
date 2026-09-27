import os
import numpy as np
import pandas as pd


# ============================================================
# TEMPORAL SNN CENTER DETECTOR V3 - EVALUATION
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V3 - EVALUATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(SNN_PROJECT_DIR, "data")
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

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "v3_center_detection_summary.csv"
)


# ============================================================
# EVALUATION PARAMETERS
# ============================================================

THRESHOLD = 0.50
MIN_DISTANCE = 6
MAX_PEAKS = 5
MATCH_DISTANCE = 8.0


print("\nEvaluation parameters:")
print(f"Threshold:        {THRESHOLD}")
print(f"Min peak distance:{MIN_DISTANCE} pixels")
print(f"Max peaks/bin:    {MAX_PEAKS}")
print(f"Match distance:   {MATCH_DISTANCE} pixels")


# ============================================================
# FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)

files = {
    "045 prediction": PRED_045,
    "047 prediction": PRED_047,
    "045 GT": GT_045,
    "047 GT": GT_047,
}

for name, path in files.items():

    if os.path.exists(path):
        print(f"{name:<20}: FOUND")
    else:
        print(f"{name:<20}: NOT FOUND")
        print(f"Path: {path}")
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
    """
    Detect local maxima from a heatmap using NumPy only.

    Returns:
        List of tuples:
        (score, x, y)
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

            if score >= float(neighborhood.max()):

                candidates.append(
                    (
                        score,
                        int(x),
                        int(y)
                    )
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
# LOAD GROUND TRUTH CENTERS
# ============================================================

def load_gt_centers(gt_file):

    df = pd.read_csv(gt_file)

    centers_by_bin = {}

    for _, row in df.iterrows():

        visible = int(row["visible"])

        if visible == 0:
            continue

        bin_id = int(row["bin_id"])

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
# MATCH PREDICTIONS TO GROUND TRUTH
# ============================================================

def match_predictions(
    predictions,
    ground_truth,
    match_distance=MATCH_DISTANCE
):
    """
    Greedy nearest-neighbour matching.

    Each GT and prediction can only be matched once.
    """

    if len(predictions) == 0 or len(ground_truth) == 0:

        return [], len(predictions), len(ground_truth)

    pairs = []

    for pred_index, pred in enumerate(predictions):

        _, px, py = pred

        for gt_index, gt in enumerate(ground_truth):

            gx, gy = gt

            distance = np.sqrt(
                (px - gx) ** 2 +
                (py - gy) ** 2
            )

            if distance <= match_distance:

                pairs.append(
                    (
                        distance,
                        pred_index,
                        gt_index
                    )
                )

    # Closest pairs first
    pairs.sort(
        key=lambda item: item[0]
    )

    used_predictions = set()
    used_gt = set()

    matches = []

    for distance, pred_index, gt_index in pairs:

        if pred_index in used_predictions:
            continue

        if gt_index in used_gt:
            continue

        used_predictions.add(pred_index)
        used_gt.add(gt_index)

        matches.append(distance)

    false_positives = (
        len(predictions) -
        len(matches)
    )

    false_negatives = (
        len(ground_truth) -
        len(matches)
    )

    return (
        matches,
        false_positives,
        false_negatives
    )


# ============================================================
# EVALUATE ONE SEQUENCE
# ============================================================

def evaluate_sequence(
    sequence_name,
    prediction_file,
    gt_file
):

    print("\n" + "=" * 70)
    print(f"EVALUATING {sequence_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    predictions_heatmap = np.load(
        prediction_file
    )

    print(
        f"Prediction shape: {predictions_heatmap.shape}"
    )

    # Expected:
    # (50, 128, 128)

    if predictions_heatmap.ndim != 3:

        raise ValueError(
            f"Unexpected prediction shape: "
            f"{predictions_heatmap.shape}"
        )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt_by_bin = load_gt_centers(
        gt_file
    )

    print(
        f"GT bins: {len(gt_by_bin)}"
    )

    # --------------------------------------------------------
    # Evaluation counters
    # --------------------------------------------------------

    total_gt = 0
    total_predictions = 0
    total_matches = 0

    all_errors = []

    per_bin_results = []

    # --------------------------------------------------------
    # Process each temporal bin
    # --------------------------------------------------------

    num_bins = predictions_heatmap.shape[0]

    for bin_id in range(num_bins):

        heatmap = predictions_heatmap[
            bin_id
        ]

        gt_centers = gt_by_bin.get(
            bin_id,
            []
        )

        # Detect peaks
        detected_peaks = detect_peaks(
            heatmap
        )

        # Match
        matches, false_positives, false_negatives = (
            match_predictions(
                detected_peaks,
                gt_centers
            )
        )

        total_gt += len(gt_centers)

        total_predictions += len(
            detected_peaks
        )

        total_matches += len(matches)

        all_errors.extend(matches)

        per_bin_results.append(
            {
                "sequence": sequence_name,
                "bin_id": bin_id,
                "gt_count": len(gt_centers),
                "prediction_count": len(detected_peaks),
                "matches": len(matches),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "mean_error": (
                    np.mean(matches)
                    if len(matches) > 0
                    else np.nan
                )
            }
        )

    # ========================================================
    # METRICS
    # ========================================================

    precision = (
        total_matches / total_predictions
        if total_predictions > 0
        else 0.0
    )

    recall = (
        total_matches / total_gt
        if total_gt > 0
        else 0.0
    )

    mean_error = (
        float(np.mean(all_errors))
        if len(all_errors) > 0
        else np.nan
    )

    median_error = (
        float(np.median(all_errors))
        if len(all_errors) > 0
        else np.nan
    )

    false_positives = (
        total_predictions -
        total_matches
    )

    false_negatives = (
        total_gt -
        total_matches
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nResults:")
    print("-" * 50)

    print(
        f"GT ships:           {total_gt}"
    )

    print(
        f"Predictions:        {total_predictions}"
    )

    print(
        f"Matches:            {total_matches}"
    )

    print(
        f"False positives:    {false_positives}"
    )

    print(
        f"False negatives:    {false_negatives}"
    )

    print(
        f"Precision:          {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall:             {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    if not np.isnan(mean_error):

        print(
            f"Mean center error:  "
            f"{mean_error:.2f} pixels"
        )

        print(
            f"Median center error:"
            f" {median_error:.2f} pixels"
        )

    else:

        print(
            "Mean center error:  N/A"
        )

        print(
            "Median center error: N/A"
        )

    # ========================================================
    # SAVE PER-BIN RESULTS
    # ========================================================

    bin_results_file = os.path.join(
        RESULT_DIR,
        f"{sequence_name}_v3_evaluation.csv"
    )

    pd.DataFrame(
        per_bin_results
    ).to_csv(
        bin_results_file,
        index=False
    )

    print(
        f"\nPer-bin results saved:"
    )
    print(bin_results_file)

    return {
        "sequence": sequence_name,
        "gt_ships": total_gt,
        "predictions": total_predictions,
        "matches": total_matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_center_error": mean_error,
        "median_center_error": median_error
    }


# ============================================================
# RUN EVALUATION
# ============================================================

results = []

results.append(
    evaluate_sequence(
        "ship_045",
        PRED_045,
        GT_045
    )
)

results.append(
    evaluate_sequence(
        "ship_047",
        PRED_047,
        GT_047
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    results
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V3 EVALUATION COMPLETE")
print("=" * 70)

print("\nSummary:")

for result in results:

    print(
        f"\n{result['sequence']}"
    )

    print(
        f"  Precision: "
        f"{result['precision'] * 100:.2f}%"
    )

    print(
        f"  Recall: "
        f"{result['recall'] * 100:.2f}%"
    )

    print(
        f"  Mean center error: "
        f"{result['mean_center_error']:.2f} px"
    )

    print(
        f"  Median center error: "
        f"{result['median_center_error']:.2f} px"
    )

print("\nSummary saved:")
print(SUMMARY_FILE)

print("\n" + "=" * 70)