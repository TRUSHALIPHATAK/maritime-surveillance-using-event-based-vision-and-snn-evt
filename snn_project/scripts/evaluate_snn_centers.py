import numpy as np
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PRED_045 = (
    BASE_DIR
    / "../results/temporal_snn_inference/ship_045_predicted_heatmaps.npy"
).resolve()

PRED_047 = (
    BASE_DIR
    / "../results/temporal_snn_inference/ship_047_predicted_heatmaps.npy"
).resolve()

GT_045 = (
    BASE_DIR
    / "../data/ship_045_event_gt_labels.csv"
).resolve()

GT_047 = (
    BASE_DIR
    / "../data/ship_047_event_gt_labels.csv"
).resolve()

RESULT_DIR = (
    BASE_DIR
    / "../results/temporal_snn_inference"
).resolve()

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

# Minimum heatmap activation to consider as a detection
THRESHOLD = 0.30

# Minimum distance between detected peaks
MIN_DISTANCE = 5

# Maximum number of ships to detect
MAX_PEAKS = 5


# ============================================================
# LOAD DATA
# ============================================================

pred_045 = np.load(PRED_045)
pred_047 = np.load(PRED_047)

gt_045 = pd.read_csv(GT_045)
gt_047 = pd.read_csv(GT_047)

print("045 prediction:", pred_045.shape)
print("047 prediction:", pred_047.shape)

print("045 GT:", gt_045.shape)
print("047 GT:", gt_047.shape)


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

    # Check every pixel
    for y in range(height):
        for x in range(width):

            score = heatmap[y, x]

            # Ignore weak activations
            if score < threshold:
                continue

            # Define local neighborhood
            y0 = max(0, y - min_distance)
            y1 = min(height, y + min_distance + 1)

            x0 = max(0, x - min_distance)
            x1 = min(width, x + min_distance + 1)

            neighborhood = heatmap[
                y0:y1,
                x0:x1
            ]

            # Keep only local maxima
            if score >= neighborhood.max():

                candidates.append(
                    (
                        float(score),
                        int(x),
                        int(y)
                    )
                )

    # Strongest peaks first
    candidates.sort(
        key=lambda p: p[0],
        reverse=True
    )

    # Non-maximum suppression
    selected = []

    for score, x, y in candidates:

        too_close = False

        for _, px, py in selected:

            distance = (
                (x - px) ** 2
                +
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
# MATCH PREDICTIONS TO GT
# ============================================================

def match_centers(
    predictions,
    ground_truth,
    max_distance=15
):

    matches = []

    used_gt = set()

    for score, px, py in predictions:

        best_index = None
        best_distance = float("inf")

        for i, (gx, gy) in enumerate(
            ground_truth
        ):

            if i in used_gt:
                continue

            distance = (
                (px - gx) ** 2
                +
                (py - gy) ** 2
            ) ** 0.5

            if distance < best_distance:

                best_distance = distance
                best_index = i

        if (
            best_index is not None
            and best_distance <= max_distance
        ):

            gx, gy = ground_truth[best_index]

            matches.append(
                (
                    score,
                    px,
                    py,
                    gx,
                    gy,
                    best_distance
                )
            )

            used_gt.add(best_index)

    return matches


# ============================================================
# EVALUATION
# ============================================================

def evaluate_sequence(
    prediction,
    gt,
    name
):

    print()
    print("=" * 70)
    print("Evaluating:", name)
    print("=" * 70)

    all_results = []

    total_gt = 0
    total_predictions = 0
    total_matches = 0

    for bin_id in range(
        prediction.shape[0]
    ):

        heatmap = prediction[bin_id]

        gt_centers = get_gt_centers(
            gt,
            bin_id
        )

        predictions = detect_peaks(
            heatmap
        )

        matches = match_centers(
            predictions,
            gt_centers
        )

        total_gt += len(gt_centers)
        total_predictions += len(predictions)
        total_matches += len(matches)

        # Store predictions
        for score, x, y in predictions:

            all_results.append(
                {
                    "bin_id": bin_id,
                    "prediction_x": x,
                    "prediction_y": y,
                    "score": score
                }
            )

        # Print selected bins
        if bin_id in [
            0,
            5,
            10,
            20,
            30,
            40,
            49
        ]:

            print()
            print(
                f"Bin {bin_id:02d}"
            )

            print(
                "GT centers:",
                [
                    (
                        round(x, 2),
                        round(y, 2)
                    )
                    for x, y in gt_centers
                ]
            )

            print(
                "Predicted:",
                [
                    (
                        round(x, 2),
                        round(y, 2),
                        round(score, 3)
                    )
                    for score, x, y in predictions
                ]
            )

            print(
                "Matches:",
                len(matches)
            )

            if matches:

                print(
                    "Distances:",
                    [
                        round(m[5], 2)
                        for m in matches
                    ]
                )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        total_matches / total_predictions
        if total_predictions > 0
        else 0
    )

    recall = (
        total_matches / total_gt
        if total_gt > 0
        else 0
    )

    mean_error = 0

    if total_matches > 0:

        distances = []

        for bin_id in range(
            prediction.shape[0]
        ):

            predictions = detect_peaks(
                prediction[bin_id]
            )

            gt_centers = get_gt_centers(
                gt,
                bin_id
            )

            matches = match_centers(
                predictions,
                gt_centers
            )

            distances.extend(
                [m[5] for m in matches]
            )

        mean_error = np.mean(
            distances
        )

    print()
    print("-" * 70)

    print(
        "Total GT ships:",
        total_gt
    )

    print(
        "Total predictions:",
        total_predictions
    )

    print(
        "Matched detections:",
        total_matches
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"Mean center error: {mean_error:.2f} pixels"
    )

    # --------------------------------------------------------
    # Save detections
    # --------------------------------------------------------

    output = pd.DataFrame(
        all_results
    )

    output_path = (
        RESULT_DIR
        / f"{name}_detected_centers.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        "Saved:",
        output_path
    )

    return {
        "sequence": name,
        "gt": total_gt,
        "predictions": total_predictions,
        "matches": total_matches,
        "precision": precision,
        "recall": recall,
        "mean_error": mean_error
    }


# ============================================================
# RUN
# ============================================================

results = []

results.append(
    evaluate_sequence(
        pred_045,
        gt_045,
        "ship_045"
    )
)

results.append(
    evaluate_sequence(
        pred_047,
        gt_047,
        "ship_047"
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)

summary_path = (
    RESULT_DIR
    / "center_detection_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print()
print("=" * 70)
print("CENTER DETECTION EVALUATION COMPLETE")
print("=" * 70)

print(summary.to_string(index=False))

print()
print(
    "Summary saved:",
    summary_path
)