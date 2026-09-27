import os
import numpy as np
import pandas as pd


# ============================================================
# V3 PEAK SCORE ANALYSIS
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V3 - PEAK SCORE ANALYSIS")
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
    "v3_peak_score_analysis.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

THRESHOLD = 0.30
MIN_DISTANCE = 6
MAX_PEAKS = 10
MATCH_DISTANCE = 8.0


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
# ANALYZE ONE SEQUENCE
# ============================================================

def analyze_sequence(
    sequence_name,
    prediction_file,
    gt_file
):

    print("\n" + "=" * 70)
    print(
        f"ANALYZING {sequence_name}"
    )
    print("=" * 70)

    heatmaps = np.load(
        prediction_file
    )

    gt_by_bin = load_gt_centers(
        gt_file
    )

    results = []

    for bin_id in range(
        heatmaps.shape[0]
    ):

        heatmap = heatmaps[
            bin_id
        ]

        gt_centers = gt_by_bin.get(
            bin_id,
            []
        )

        peaks = detect_peaks(
            heatmap
        )

        used_gt = set()

        for peak_rank, (
            score,
            px,
            py
        ) in enumerate(peaks, start=1):

            best_distance = None
            best_gt_index = None

            for gt_index, (
                gx,
                gy
            ) in enumerate(gt_centers):

                if gt_index in used_gt:
                    continue

                distance = np.sqrt(
                    (px - gx) ** 2 +
                    (py - gy) ** 2
                )

                if distance <= MATCH_DISTANCE:

                    if (
                        best_distance is None
                        or distance < best_distance
                    ):

                        best_distance = distance
                        best_gt_index = gt_index

            if best_gt_index is not None:

                used_gt.add(
                    best_gt_index
                )

                matched = True

            else:

                matched = False

            results.append(
                {
                    "sequence": sequence_name,
                    "bin_id": bin_id,
                    "peak_rank": peak_rank,
                    "x": px,
                    "y": py,
                    "score": score,
                    "matched": matched,
                    "center_error": (
                        best_distance
                        if matched
                        else np.nan
                    )
                }
            )

    return results


# ============================================================
# RUN
# ============================================================

all_results = []

all_results.extend(
    analyze_sequence(
        "ship_045",
        PRED_045,
        GT_045
    )
)

all_results.extend(
    analyze_sequence(
        "ship_047",
        PRED_047,
        GT_047
    )
)


df = pd.DataFrame(
    all_results
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PEAK SCORE SUMMARY")
print("=" * 70)


for sequence in [
    "ship_045",
    "ship_047"
]:

    seq = df[
        df["sequence"] == sequence
    ]

    matched = seq[
        seq["matched"] == True
    ]

    false_positive = seq[
        seq["matched"] == False
    ]

    print(
        f"\n{sequence}"
    )

    print(
        f"Total peaks: "
        f"{len(seq)}"
    )

    print(
        f"Matched peaks: "
        f"{len(matched)}"
    )

    print(
        f"False-positive peaks: "
        f"{len(false_positive)}"
    )

    if len(matched) > 0:

        print(
            f"Matched score mean: "
            f"{matched['score'].mean():.4f}"
        )

        print(
            f"Matched score median: "
            f"{matched['score'].median():.4f}"
        )

        print(
            f"Matched score min: "
            f"{matched['score'].min():.4f}"
        )

    if len(false_positive) > 0:

        print(
            f"False-positive score mean: "
            f"{false_positive['score'].mean():.4f}"
        )

        print(
            f"False-positive score median: "
            f"{false_positive['score'].median():.4f}"
        )

        print(
            f"False-positive score max: "
            f"{false_positive['score'].max():.4f}"
        )


# ============================================================
# SCORE THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

for threshold in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
]:

    print(
        f"\nThreshold >= {threshold:.2f}"
    )

    for sequence in [
        "ship_045",
        "ship_047"
    ]:

        seq = df[
            (df["sequence"] == sequence) &
            (df["score"] >= threshold)
        ]

        matched = seq[
            seq["matched"] == True
        ]

        print(
            f"  {sequence}: "
            f"{len(seq)} peaks, "
            f"{len(matched)} matched"
        )


# ============================================================
# TOP PEAKS
# ============================================================

print("\n" + "=" * 70)
print("TOP PEAKS")
print("=" * 70)

for sequence in [
    "ship_045",
    "ship_047"
]:

    print(
        f"\n{sequence}"
    )

    seq = df[
        df["sequence"] == sequence
    ].sort_values(
        "score",
        ascending=False
    )

    print(
        seq[
            [
                "bin_id",
                "peak_rank",
                "x",
                "y",
                "score",
                "matched",
                "center_error"
            ]
        ].head(20).to_string(
            index=False
        )
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved:"
)

print(
    OUTPUT_FILE
)