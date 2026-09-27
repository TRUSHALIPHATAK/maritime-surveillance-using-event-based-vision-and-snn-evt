import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50

MIN_DISTANCE = 6

MAX_PEAKS = 5

MATCH_DISTANCE = 8.0


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v2"
)

DATA_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "data"
)


# ============================================================
# FILES
# ============================================================

PRED_045 = os.path.join(
    RESULT_DIR,
    "ship_045_predicted_heatmaps_v2.npy"
)

PRED_047 = os.path.join(
    RESULT_DIR,
    "ship_047_predicted_heatmaps_v2.npy"
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


            if score >= neighborhood.max():

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
# GT CENTERS
# ============================================================

def get_gt_centers(
    gt_dataframe,
    bin_id
):

    rows = gt_dataframe[
        gt_dataframe["bin_id"] == bin_id
    ]


    centers = []


    for _, row in rows.iterrows():

        if int(row["visible"]) != 1:
            continue


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
# EVALUATE ONE SEQUENCE
# ============================================================

def evaluate_sequence(
    prediction_file,
    gt_file,
    sequence_name
):

    print()
    print("=" * 70)
    print(sequence_name)
    print("=" * 70)


    predictions = np.load(
        prediction_file
    )


    gt = pd.read_csv(
        gt_file
    )


    print(
        "Prediction shape:",
        predictions.shape
    )


    all_predictions = 0
    all_gt = 0
    all_matches = 0

    errors = []

    rows = []


    for bin_id in range(
        predictions.shape[0]
    ):

        heatmap = predictions[
            bin_id
        ]


        detected = detect_peaks(
            heatmap
        )


        gt_centers = get_gt_centers(
            gt,
            bin_id
        )


        all_predictions += len(
            detected
        )

        all_gt += len(
            gt_centers
        )


        # ----------------------------------------------------
        # GREEDY MATCHING
        # ----------------------------------------------------

        matched_gt = set()


        for score, px, py in detected:

            best_distance = float(
                "inf"
            )

            best_index = None


            for index, (
                gx,
                gy
            ) in enumerate(gt_centers):

                if index in matched_gt:
                    continue


                distance = (
                    (px - gx) ** 2 +
                    (py - gy) ** 2
                ) ** 0.5


                if distance < best_distance:

                    best_distance = distance

                    best_index = index


            if (
                best_index is not None
                and best_distance <= MATCH_DISTANCE
            ):

                matched_gt.add(
                    best_index
                )

                all_matches += 1

                errors.append(
                    best_distance
                )


        rows.append(
            {
                "bin_id": bin_id,
                "ground_truth": len(gt_centers),
                "predictions": len(detected),
                "matches": len(matched_gt)
            }
        )


    # ========================================================
    # METRICS
    # ========================================================

    if all_predictions > 0:

        precision = (
            all_matches /
            all_predictions
        )

    else:

        precision = 0.0


    if all_gt > 0:

        recall = (
            all_matches /
            all_gt
        )

    else:

        recall = 0.0


    if len(errors) > 0:

        mean_error = float(
            np.mean(errors)
        )

        median_error = float(
            np.median(errors)
        )

    else:

        mean_error = 0.0

        median_error = 0.0


    print()
    print("Ground-truth centers:", all_gt)
    print("Predicted centers:", all_predictions)
    print("Matched centers:", all_matches)

    print()
    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"Mean error:   {mean_error:.2f} pixels"
    )

    print(
        f"Median error: {median_error:.2f} pixels"
    )


    # ========================================================
    # SAVE BIN RESULTS
    # ========================================================

    output_csv = os.path.join(
        RESULT_DIR,
        f"{sequence_name}_evaluation.csv"
    )


    pd.DataFrame(
        rows
    ).to_csv(
        output_csv,
        index=False
    )


    return {
        "sequence": sequence_name,
        "ground_truth": all_gt,
        "predictions": all_predictions,
        "matches": all_matches,
        "precision": precision,
        "recall": recall,
        "mean_error": mean_error,
        "median_error": median_error
    }


# ============================================================
# RUN EVALUATION
# ============================================================

result_045 = evaluate_sequence(
    PRED_045,
    GT_045,
    "ship_045"
)


result_047 = evaluate_sequence(
    PRED_047,
    GT_047,
    "ship_047"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        result_045,
        result_047
    ]
)


summary_path = os.path.join(
    RESULT_DIR,
    "v2_center_detection_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()
print(summary.to_string(index=False))

print()
print(
    "Summary saved:"
)

print(
    os.path.abspath(
        summary_path
    )
)