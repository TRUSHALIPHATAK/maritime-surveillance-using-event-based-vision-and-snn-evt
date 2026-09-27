import os
import numpy as np
import pandas as pd


# ============================================================
# FINAL VERIFICATION METRICS
# ============================================================
#
# Evaluates the final V5 filtered tracking results for:
#   - Ship 045
#   - Ship 047
#
# Metrics:
#   - Precision
#   - Recall
#   - F1 score
#   - Mean localization error
#   - Median localization error
#   - Maximum localization error
#   - Percentage within 2 / 4 / 6 pixels
#   - Number of tracks
#   - Longest track
#   - Mean track length
#   - Track coverage
#   - ID switches
#   - Total events
#   - Events per temporal bin
#   - Total SNN spikes
#   - Spikes per temporal bin
#
# Matching criterion:
#   Prediction is considered a match when its center is
#   within 6 pixels of a ground-truth center.
#
# ============================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "data"
)

RESULTS_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results"
)


# ============================================================
# DATASET PATHS
# ============================================================

DATASETS = {

    "045": {

        "gt": os.path.join(
            DATA_DIR,
            "ship_045_tracking_ground_truth.csv"
        ),

        "tracks": os.path.join(
            RESULTS_DIR,
            "temporal_tracking_v5",
            "ship_045_tracks_filtered.csv"
        ),

        "events": os.path.join(
            SNN_PROJECT_DIR,
            "..",
            "data_samples",
            "output",
            "ship_045_full.txt"
        ),

        "spikes": os.path.join(
            DATA_DIR,
            "ship_045_full_snn_spikes.npy"
        ),
    },

    "047": {

        "gt": os.path.join(
            DATA_DIR,
            "ship_047_event_gt_labels.csv"
        ),

        "tracks": os.path.join(
            RESULTS_DIR,
            "temporal_tracking_v5",
            "ship_047_tracks_filtered.csv"
        ),

        "events": os.path.join(
            SNN_PROJECT_DIR,
            "..",
            "data_samples",
            "output",
            "ship_047_full.txt"
        ),

        "spikes": os.path.join(
            DATA_DIR,
            "ship_047_full_snn_spikes.npy"
        ),
    }
}


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, candidates):

    columns = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(dataset, path):

    df = pd.read_csv(path)

    print()
    print("GT columns:", list(df.columns))

    # ========================================================
    # SHIP 045
    #
    # Columns:
    # bin_id
    # frame_number
    # target_id
    # visible
    # x
    # y
    # width
    # height
    # center_x
    # center_y
    # ========================================================

    if dataset == "045":

        bin_col = find_column(
            df,
            ["bin_id", "bin"]
        )

        target_col = find_column(
            df,
            ["target_id", "target"]
        )

        x_col = find_column(
            df,
            ["center_x", "cx"]
        )

        y_col = find_column(
            df,
            ["center_y", "cy"]
        )

        visible_col = find_column(
            df,
            ["visible", "visibility"]
        )

        if None in [
            bin_col,
            target_col,
            x_col,
            y_col
        ]:

            raise ValueError(
                "Could not identify Ship 045 GT columns."
            )

        gt = pd.DataFrame({

            "bin_id":
                df[bin_col].astype(int),

            "target_id":
                df[target_col].astype(int),

            "x":
                df[x_col].astype(float),

            "y":
                df[y_col].astype(float),
        })

        if visible_col is not None:

            gt["visible"] = (
                df[visible_col].astype(bool)
            )

        else:

            gt["visible"] = True

        return gt


    # ========================================================
    # SHIP 047
    #
    # Actual columns:
    #
    # bin_id
    # frame_number
    # target_id
    # visible
    # x_original
    # y_original
    # w_original
    # h_original
    # x_128
    # y_128
    # w_128
    # h_128
    #
    # Calculate center:
    #
    # center_x = x_128 + w_128 / 2
    # center_y = y_128 + h_128 / 2
    # ========================================================

    if dataset == "047":

        bin_col = find_column(
            df,
            ["bin_id", "bin"]
        )

        target_col = find_column(
            df,
            ["target_id", "target"]
        )

        visible_col = find_column(
            df,
            ["visible", "visibility"]
        )

        x_col = find_column(
            df,
            ["x_128"]
        )

        y_col = find_column(
            df,
            ["y_128"]
        )

        width_col = find_column(
            df,
            ["w_128"]
        )

        height_col = find_column(
            df,
            ["h_128"]
        )

        if None in [
            bin_col,
            target_col,
            x_col,
            y_col,
            width_col,
            height_col
        ]:

            raise ValueError(
                "Could not identify Ship 047 GT bbox columns."
            )

        gt = pd.DataFrame({

            "bin_id":
                df[bin_col].astype(int),

            "target_id":
                df[target_col].astype(int),

            "x":
                df[x_col].astype(float)
                +
                df[width_col].astype(float) / 2.0,

            "y":
                df[y_col].astype(float)
                +
                df[height_col].astype(float) / 2.0,
        })

        if visible_col is not None:

            gt["visible"] = (
                df[visible_col].astype(bool)
            )

        else:

            gt["visible"] = True

        return gt


    raise ValueError(
        f"Unsupported dataset: {dataset}"
    )


# ============================================================
# LOAD TRACKS
# ============================================================

def load_tracks(path):

    df = pd.read_csv(path)

    print(
        "Track columns:",
        list(df.columns)
    )

    bin_col = find_column(
        df,
        ["bin_id", "bin"]
    )

    track_col = find_column(
        df,
        ["track_id", "track", "id"]
    )

    x_col = find_column(
        df,
        ["x", "center_x", "cx"]
    )

    y_col = find_column(
        df,
        ["y", "center_y", "cy"]
    )

    score_col = find_column(
        df,
        ["score", "confidence", "confidence_score"]
    )

    if None in [
        bin_col,
        track_col,
        x_col,
        y_col
    ]:

        raise ValueError(
            "Could not identify tracking columns."
        )

    tracks = pd.DataFrame({

        "bin_id":
            df[bin_col].astype(int),

        "track_id":
            df[track_col].astype(int),

        "x":
            df[x_col].astype(float),

        "y":
            df[y_col].astype(float)
    })

    if score_col is not None:

        tracks["score"] = (
            df[score_col].astype(float)
        )

    else:

        tracks["score"] = 1.0

    return tracks


# ============================================================
# MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def match_predictions(
    gt,
    tracks,
    distance_threshold=6.0
):

    matches = []

    all_bins = sorted(
        set(gt["bin_id"].unique())
        |
        set(tracks["bin_id"].unique())
    )

    for bin_id in all_bins:

        gt_bin = gt[
            (gt["bin_id"] == bin_id)
            &
            (gt["visible"])
        ]

        pred_bin = tracks[
            tracks["bin_id"] == bin_id
        ]

        if len(gt_bin) == 0:
            continue

        if len(pred_bin) == 0:
            continue

        gt_used = set()
        pred_used = set()

        candidates = []

        # ----------------------------------------------------
        # Calculate every GT/prediction distance
        # ----------------------------------------------------

        for gt_index, gt_row in gt_bin.iterrows():

            for pred_index, pred_row in pred_bin.iterrows():

                dx = (
                    pred_row["x"]
                    -
                    gt_row["x"]
                )

                dy = (
                    pred_row["y"]
                    -
                    gt_row["y"]
                )

                distance = np.sqrt(
                    dx ** 2
                    +
                    dy ** 2
                )

                candidates.append(
                    (
                        distance,
                        gt_index,
                        pred_index
                    )
                )

        # ----------------------------------------------------
        # Nearest one-to-one matching
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: item[0]
        )

        for (
            distance,
            gt_index,
            pred_index
        ) in candidates:

            if gt_index in gt_used:
                continue

            if pred_index in pred_used:
                continue

            if distance > distance_threshold:
                continue

            gt_row = gt_bin.loc[
                gt_index
            ]

            pred_row = pred_bin.loc[
                pred_index
            ]

            matches.append({

                "bin_id":
                    bin_id,

                "target_id":
                    int(gt_row["target_id"]),

                "track_id":
                    int(pred_row["track_id"]),

                "distance":
                    float(distance),

                "gt_x":
                    float(gt_row["x"]),

                "gt_y":
                    float(gt_row["y"]),

                "pred_x":
                    float(pred_row["x"]),

                "pred_y":
                    float(pred_row["y"])
            })

            gt_used.add(gt_index)
            pred_used.add(pred_index)

    return pd.DataFrame(matches)


# ============================================================
# ID SWITCHES
# ============================================================

def calculate_id_switches(matches):

    if len(matches) == 0:
        return 0

    switches = 0

    for target_id in sorted(
        matches["target_id"].unique()
    ):

        target_matches = matches[
            matches["target_id"] == target_id
        ].sort_values(
            "bin_id"
        )

        previous_track = None

        for _, row in target_matches.iterrows():

            current_track = int(
                row["track_id"]
            )

            if (
                previous_track is not None
                and
                current_track != previous_track
            ):

                switches += 1

            previous_track = current_track

    return switches


# ============================================================
# TRACK COVERAGE
# ============================================================

def calculate_track_coverage(
    gt,
    matches
):

    visible_gt = gt[
        gt["visible"]
    ]

    if len(visible_gt) == 0:
        return 0.0

    return (
        len(matches)
        /
        len(visible_gt)
    )


# ============================================================
# EVENT METRICS
# ============================================================

def calculate_event_metrics(
    event_file
):

    if not os.path.exists(event_file):

        return {
            "events": None,
            "bins": None,
            "events_per_bin": None
        }

    with open(
        event_file,
        "r",
        encoding="utf-8"
    ) as file:

        event_count = sum(
            1
            for line in file
            if line.strip()
        )

    number_of_bins = 50

    return {

        "events":
            event_count,

        "bins":
            number_of_bins,

        "events_per_bin":
            event_count
            /
            number_of_bins
    }


# ============================================================
# SNN SPIKE METRICS
# ============================================================

def calculate_spike_metrics(
    spike_file
):

    if not os.path.exists(spike_file):

        return {
            "spikes": None,
            "spike_shape": None,
            "spikes_per_bin": None
        }

    spikes = np.load(
        spike_file,
        mmap_mode="r"
    )

    total_spikes = int(
        np.sum(spikes)
    )

    if 50 in spikes.shape:

        number_of_bins = 50

    else:

        number_of_bins = None

    if number_of_bins is not None:

        spikes_per_bin = (
            total_spikes
            /
            number_of_bins
        )

    else:

        spikes_per_bin = None

    return {

        "spikes":
            total_spikes,

        "spike_shape":
            spikes.shape,

        "spikes_per_bin":
            spikes_per_bin
    }


# ============================================================
# MAIN
# ============================================================

all_results = []


print()
print("=" * 80)
print("FINAL SNN DETECTION + TRACKING VERIFICATION")
print("=" * 80)


for dataset, paths in DATASETS.items():

    print()
    print("=" * 80)
    print(
        f"DATASET: SHIP {dataset}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for name, path in paths.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt = load_ground_truth(
        dataset,
        paths["gt"]
    )

    # --------------------------------------------------------
    # Load tracks
    # --------------------------------------------------------

    tracks = load_tracks(
        paths["tracks"]
    )

    visible_gt = gt[
        gt["visible"]
    ]

    print()
    print(
        f"GT visible points : "
        f"{len(visible_gt)}"
    )

    print(
        f"Predicted points  : "
        f"{len(tracks)}"
    )

    # --------------------------------------------------------
    # Match predictions
    # --------------------------------------------------------

    matches = match_predictions(
        gt,
        tracks,
        distance_threshold=6.0
    )

    # --------------------------------------------------------
    # Detection metrics
    # --------------------------------------------------------

    visible_gt_count = len(
        visible_gt
    )

    prediction_count = len(
        tracks
    )

    true_positive = len(
        matches
    )

    false_positive = (
        prediction_count
        -
        true_positive
    )

    false_negative = (
        visible_gt_count
        -
        true_positive
    )

    precision = (

        true_positive
        /
        prediction_count

        if prediction_count > 0
        else 0.0
    )

    recall = (

        true_positive
        /
        visible_gt_count

        if visible_gt_count > 0
        else 0.0
    )

    if (
        precision + recall
        >
        0
    ):

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0.0

    # --------------------------------------------------------
    # Localization metrics
    # --------------------------------------------------------

    if len(matches) > 0:

        mean_error = (
            matches["distance"].mean()
        )

        median_error = (
            matches["distance"].median()
        )

        maximum_error = (
            matches["distance"].max()
        )

        within_2px = (
            matches["distance"]
            <=
            2
        ).mean()

        within_4px = (
            matches["distance"]
            <=
            4
        ).mean()

        within_6px = (
            matches["distance"]
            <=
            6
        ).mean()

    else:

        mean_error = 0.0
        median_error = 0.0
        maximum_error = 0.0

        within_2px = 0.0
        within_4px = 0.0
        within_6px = 0.0

    # --------------------------------------------------------
    # Track metrics
    # --------------------------------------------------------

    track_lengths = (
        tracks
        .groupby("track_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    if len(track_lengths) > 0:

        number_of_tracks = (
            len(track_lengths)
        )

        longest_track = int(
            track_lengths.iloc[0]
        )

        mean_track_length = (
            track_lengths.mean()
        )

    else:

        number_of_tracks = 0
        longest_track = 0
        mean_track_length = 0.0

    track_coverage = (
        calculate_track_coverage(
            gt,
            matches
        )
    )

    id_switches = (
        calculate_id_switches(
            matches
        )
    )

    # --------------------------------------------------------
    # Event metrics
    # --------------------------------------------------------

    event_metrics = (
        calculate_event_metrics(
            paths["events"]
        )
    )

    # --------------------------------------------------------
    # SNN metrics
    # --------------------------------------------------------

    spike_metrics = (
        calculate_spike_metrics(
            paths["spikes"]
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("DETECTION")
    print("-" * 50)

    print(
        f"True Positives       : "
        f"{true_positive}"
    )

    print(
        f"False Positives      : "
        f"{false_positive}"
    )

    print(
        f"False Negatives      : "
        f"{false_negative}"
    )

    print(
        f"Precision            : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall               : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score             : "
        f"{f1:.4f}"
    )

    print()
    print("LOCALIZATION")
    print("-" * 50)

    print(
        f"Mean center error    : "
        f"{mean_error:.3f} px"
    )

    print(
        f"Median center error  : "
        f"{median_error:.3f} px"
    )

    print(
        f"Maximum error        : "
        f"{maximum_error:.3f} px"
    )

    print(
        f"Within 2 px          : "
        f"{within_2px * 100:.2f}%"
    )

    print(
        f"Within 4 px          : "
        f"{within_4px * 100:.2f}%"
    )

    print(
        f"Within 6 px          : "
        f"{within_6px * 100:.2f}%"
    )

    print()
    print("TRACKING")
    print("-" * 50)

    print(
        f"Number of tracks     : "
        f"{number_of_tracks}"
    )

    print(
        f"Longest track        : "
        f"{longest_track} bins"
    )

    print(
        f"Mean track length    : "
        f"{mean_track_length:.2f} bins"
    )

    print(
        f"Track coverage       : "
        f"{track_coverage * 100:.2f}%"
    )

    print(
        f"ID switches          : "
        f"{id_switches}"
    )

    print()
    print("EVENT PROCESSING")
    print("-" * 50)

    print(
        f"Total events         : "
        f"{event_metrics['events']}"
    )

    print(
        f"Temporal bins        : "
        f"{event_metrics['bins']}"
    )

    print(
        f"Events / bin         : "
        f"{event_metrics['events_per_bin']:.2f}"
    )

    print()
    print("SNN")
    print("-" * 50)

    print(
        f"SNN spike count      : "
        f"{spike_metrics['spikes']}"
    )

    print(
        f"SNN tensor shape     : "
        f"{spike_metrics['spike_shape']}"
    )

    if (
        spike_metrics["spikes_per_bin"]
        is not None
    ):

        print(
            f"Spikes / bin         : "
            f"{spike_metrics['spikes_per_bin']:.2f}"
        )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.append({

        "Dataset":
            dataset,

        "TP":
            true_positive,

        "FP":
            false_positive,

        "FN":
            false_negative,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Mean_Error_px":
            mean_error,

        "Median_Error_px":
            median_error,

        "Max_Error_px":
            maximum_error,

        "Within_2px_%":
            within_2px * 100,

        "Within_4px_%":
            within_4px * 100,

        "Within_6px_%":
            within_6px * 100,

        "Tracks":
            number_of_tracks,

        "Longest_Track_bins":
            longest_track,

        "Mean_Track_bins":
            mean_track_length,

        "Track_Coverage_%":
            track_coverage * 100,

        "ID_Switches":
            id_switches,

        "Events":
            event_metrics["events"],

        "Events_per_bin":
            event_metrics["events_per_bin"],

        "SNN_Spikes":
            spike_metrics["spikes"],

        "Spikes_per_bin":
            spike_metrics["spikes_per_bin"]
    })


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    all_results
)

print()
print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "final_verification_metrics.csv"
)

summary.to_csv(
    output_file,
    index=False
)

print()
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    output_file
)