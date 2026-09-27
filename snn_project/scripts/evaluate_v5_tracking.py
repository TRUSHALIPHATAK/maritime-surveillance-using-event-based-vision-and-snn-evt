import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    SNN_PROJECT_DIR
)


TRACKING_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5"
)


GT_045 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_045_tracking_ground_truth.csv"
)

GT_047 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_047_event_gt_labels.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

MATCH_DISTANCE = 8.0


# ============================================================
# LOAD GT
# ============================================================

def load_gt_045():

    df = pd.read_csv(
        GT_045
    )

    rows = []

    for _, row in df.iterrows():

        if int(row["visible"]) != 1:
            continue

        rows.append(
            {
                "bin_id": int(
                    row["bin_id"]
                ),

                "target_id": int(
                    row["target_id"]
                ),

                "x": float(
                    row["center_x"]
                ),

                "y": float(
                    row["center_y"]
                )
            }
        )

    return pd.DataFrame(rows)


def load_gt_047():

    df = pd.read_csv(
        GT_047
    )

    rows = []

    for _, row in df.iterrows():

        if int(row["visible"]) != 1:
            continue

        x = (
            float(row["x_128"])
            +
            float(row["w_128"]) / 2.0
        )

        y = (
            float(row["y_128"])
            +
            float(row["h_128"]) / 2.0
        )

        rows.append(
            {
                "bin_id": int(
                    row["bin_id"]
                ),

                "target_id": int(
                    row["target_id"]
                ),

                "x": x,
                "y": y
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EVALUATE TRACKS
# ============================================================

def evaluate_sequence(
    sequence_name,
    gt
):

    track_file = os.path.join(
        TRACKING_DIR,
        f"{sequence_name}_tracks.csv"
    )

    tracks = pd.read_csv(
        track_file
    )


    print()
    print("=" * 70)
    print(
        f"V5 TRACK EVALUATION: {sequence_name}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Track-level results
    # --------------------------------------------------------

    track_results = []


    for track_id in sorted(
        tracks["track_id"].unique()
    ):

        track = tracks[
            tracks["track_id"]
            ==
            track_id
        ]


        matched_distances = []

        matched_targets = []


        for _, detection in track.iterrows():

            bin_id = int(
                detection["bin_id"]
            )

            x = float(
                detection["x"]
            )

            y = float(
                detection["y"]
            )


            gt_bin = gt[
                gt["bin_id"]
                ==
                bin_id
            ]


            if len(gt_bin) == 0:

                continue


            distances = []

            for _, target in gt_bin.iterrows():

                d = np.sqrt(
                    (
                        x
                        -
                        target["x"]
                    ) ** 2
                    +
                    (
                        y
                        -
                        target["y"]
                    ) ** 2
                )

                distances.append(
                    (
                        d,
                        int(
                            target["target_id"]
                        )
                    )
                )


            distances.sort(
                key=lambda item: item[0]
            )


            best_distance, best_target = (
                distances[0]
            )


            matched_distances.append(
                best_distance
            )

            matched_targets.append(
                best_target
            )


        if len(
            matched_distances
        ) == 0:

            mean_error = np.nan
            median_error = np.nan
            match_ratio = 0.0

        else:

            matched_distances = np.array(
                matched_distances
            )

            mean_error = float(
                matched_distances.mean()
            )

            median_error = float(
                np.median(
                    matched_distances
                )
            )

            match_ratio = float(
                np.mean(
                    matched_distances
                    <=
                    MATCH_DISTANCE
                )
            )


        # Most frequently matched target
        if matched_targets:

            values, counts = np.unique(
                matched_targets,
                return_counts=True
            )

            dominant_target = int(
                values[
                    np.argmax(counts)
                ]
            )

            dominant_count = int(
                counts.max()
            )

        else:

            dominant_target = -1
            dominant_count = 0


        track_results.append(
            {
                "track_id": track_id,

                "track_length":
                    len(track),

                "mean_score":
                    float(
                        track["score"].mean()
                    ),

                "mean_error":
                    mean_error,

                "median_error":
                    median_error,

                "match_ratio":
                    match_ratio,

                "dominant_target":
                    dominant_target,

                "dominant_target_hits":
                    dominant_count
            }
        )


    results = pd.DataFrame(
        track_results
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        results.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Best target association
    # --------------------------------------------------------

    print()
    print(
        "Likely track-to-GT associations:"
    )


    for target_id in sorted(
        gt["target_id"].unique()
    ):

        target_results = results[
            results["dominant_target"]
            ==
            target_id
        ]


        if len(target_results) == 0:

            print(
                f"GT Target {target_id}: "
                f"NO TRACK"
            )

            continue


        best = target_results.sort_values(
            [
                "match_ratio",
                "track_length"
            ],
            ascending=False
        ).iloc[0]


        print(
            f"GT Target {target_id}: "
            f"Track {int(best['track_id'])} | "
            f"length={int(best['track_length'])} | "
            f"match={best['match_ratio']:.3f} | "
            f"error={best['mean_error']:.2f}px"
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        TRACKING_DIR,
        f"{sequence_name}_evaluation.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )


    print()
    print(
        "Saved:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    gt_045 = load_gt_045()

    gt_047 = load_gt_047()


    evaluate_sequence(
        "ship_045",
        gt_045
    )


    evaluate_sequence(
        "ship_047",
        gt_047
    )