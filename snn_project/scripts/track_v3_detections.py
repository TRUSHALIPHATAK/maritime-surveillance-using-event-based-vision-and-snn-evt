import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3"
)

TRACK_RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v3"
)

os.makedirs(TRACK_RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.72

MIN_DISTANCE = 6

MAX_PEAKS = 10

# Maximum allowed movement between consecutive temporal bins
MAX_MOVE = 12.0

# Number of consecutive bins a track can miss
MAX_MISSED = 2

# Ignore very short tracks
MIN_TRACK_LENGTH = 5


# ============================================================
# INPUT FILES
# ============================================================

PRED_045 = os.path.join(
    RESULT_DIR,
    "ship_045_predicted_heatmaps_v3.npy"
)

PRED_047 = os.path.join(
    RESULT_DIR,
    "ship_047_predicted_heatmaps_v3.npy"
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
    """
    Detect local maxima in a heatmap using pure NumPy/Python.
    No scipy required.
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

            if score >= neighborhood.max():

                candidates.append(
                    (
                        score,
                        x,
                        y
                    )
                )

    # Highest confidence first
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
# TRACK STRUCTURE
# ============================================================

def create_track(track_id, detection, bin_id):

    score, x, y = detection

    return {
        "track_id": track_id,

        "detections": [
            {
                "bin_id": bin_id,
                "x": float(x),
                "y": float(y),
                "score": float(score)
            }
        ],

        "last_x": float(x),
        "last_y": float(y),

        "last_bin": bin_id,

        "missed": 0
    }


# ============================================================
# TEMPORAL TRACKING
# ============================================================

def track_sequence(prediction_file, sequence_name):

    print()
    print("=" * 70)
    print(f"TEMPORAL TRACKING: {sequence_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load prediction heatmaps
    # --------------------------------------------------------

    heatmaps = np.load(prediction_file)

    print("Heatmap shape:", heatmaps.shape)

    num_bins = heatmaps.shape[0]

    active_tracks = []

    finished_tracks = []

    next_track_id = 1

    total_detections = 0

    # --------------------------------------------------------
    # Process each temporal bin
    # --------------------------------------------------------

    for bin_id in range(num_bins):

        heatmap = heatmaps[bin_id]

        detections = detect_peaks(
            heatmap,
            threshold=THRESHOLD,
            min_distance=MIN_DISTANCE,
            max_peaks=MAX_PEAKS
        )

        total_detections += len(detections)

        # ----------------------------------------------------
        # Convert detection format
        # ----------------------------------------------------

        unmatched_detections = list(
            range(len(detections))
        )

        unmatched_tracks = list(
            range(len(active_tracks))
        )

        matches = []

        # ----------------------------------------------------
        # Calculate candidate distances
        # ----------------------------------------------------

        distance_candidates = []

        for ti in unmatched_tracks:

            track = active_tracks[ti]

            for di in unmatched_detections:

                score, x, y = detections[di]

                distance = np.sqrt(
                    (x - track["last_x"]) ** 2 +
                    (y - track["last_y"]) ** 2
                )

                if distance <= MAX_MOVE:

                    distance_candidates.append(
                        (
                            distance,
                            ti,
                            di
                        )
                    )

        # Closest pairs first
        distance_candidates.sort(
            key=lambda item: item[0]
        )

        used_tracks = set()
        used_detections = set()

        for distance, ti, di in distance_candidates:

            if ti in used_tracks:
                continue

            if di in used_detections:
                continue

            matches.append(
                (
                    ti,
                    di,
                    distance
                )
            )

            used_tracks.add(ti)
            used_detections.add(di)

        # ----------------------------------------------------
        # Update matched tracks
        # ----------------------------------------------------

        for ti, di, distance in matches:

            track = active_tracks[ti]

            score, x, y = detections[di]

            track["detections"].append(
                {
                    "bin_id": bin_id,
                    "x": float(x),
                    "y": float(y),
                    "score": float(score)
                }
            )

            track["last_x"] = float(x)
            track["last_y"] = float(y)
            track["last_bin"] = bin_id
            track["missed"] = 0

        # ----------------------------------------------------
        # Handle unmatched tracks
        # ----------------------------------------------------

        tracks_to_remove = []

        for ti in range(len(active_tracks)):

            if ti in used_tracks:
                continue

            track = active_tracks[ti]

            track["missed"] += 1

            if track["missed"] > MAX_MISSED:

                tracks_to_remove.append(ti)

        # Move expired tracks to finished list
        for ti in reversed(tracks_to_remove):

            finished_tracks.append(
                active_tracks.pop(ti)
            )

        # ----------------------------------------------------
        # Create new tracks for unmatched detections
        # ----------------------------------------------------

        for di, detection in enumerate(detections):

            if di in used_detections:
                continue

            new_track = create_track(
                next_track_id,
                detection,
                bin_id
            )

            active_tracks.append(new_track)

            next_track_id += 1

        print(
            f"Bin {bin_id:02d}: "
            f"detections={len(detections):2d} | "
            f"matches={len(matches):2d} | "
            f"active_tracks={len(active_tracks):2d}"
        )

    # --------------------------------------------------------
    # Finish remaining tracks
    # --------------------------------------------------------

    finished_tracks.extend(active_tracks)

    # --------------------------------------------------------
    # Filter short tracks
    # --------------------------------------------------------

    valid_tracks = []

    for track in finished_tracks:

        if len(track["detections"]) >= MIN_TRACK_LENGTH:

            valid_tracks.append(track)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TRACKING SUMMARY")
    print("-" * 70)

    print("Total detections:", total_detections)

    print("Tracks created:", next_track_id - 1)

    print("Valid tracks:", len(valid_tracks))

    if len(valid_tracks) > 0:

        lengths = [
            len(track["detections"])
            for track in valid_tracks
        ]

        print(
            "Track length:",
            f"min={min(lengths)},",
            f"max={max(lengths)},",
            f"mean={np.mean(lengths):.2f}"
        )

    # --------------------------------------------------------
    # Save tracking CSV
    # --------------------------------------------------------

    rows = []

    for track in valid_tracks:

        for detection in track["detections"]:

            rows.append(
                {
                    "track_id": track["track_id"],
                    "bin_id": detection["bin_id"],
                    "x": detection["x"],
                    "y": detection["y"],
                    "score": detection["score"]
                }
            )

    tracking_df = pd.DataFrame(rows)

    csv_path = os.path.join(
        TRACK_RESULT_DIR,
        f"{sequence_name}_tracks.csv"
    )

    tracking_df.to_csv(
        csv_path,
        index=False
    )

    print()
    print("Saved tracking CSV:")
    print(csv_path)

    return valid_tracks, tracking_df


# ============================================================
# TRACKING VISUALIZATION
# ============================================================

def plot_tracks(
    tracking_df,
    sequence_name
):

    plt.figure(figsize=(10, 7))

    if tracking_df.empty:

        plt.title(
            f"{sequence_name} - No valid tracks"
        )

        plt.xlabel("X position")
        plt.ylabel("Y position")

        plt.gca().invert_yaxis()

        output_path = os.path.join(
            TRACK_RESULT_DIR,
            f"{sequence_name}_tracking.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        return

    track_ids = sorted(
        tracking_df["track_id"].unique()
    )

    for track_id in track_ids:

        track = tracking_df[
            tracking_df["track_id"] == track_id
        ].sort_values("bin_id")

        plt.plot(
            track["x"],
            track["y"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=f"Track {track_id}"
        )

    plt.xlabel("X position (128 × 128 event grid)")
    plt.ylabel("Y position (128 × 128 event grid)")

    plt.title(
        f"{sequence_name} - SNN Temporal Ship Tracking"
    )

    plt.xlim(0, 128)
    plt.ylim(128, 0)

    if len(track_ids) <= 15:

        plt.legend(
            fontsize=7,
            loc="best"
        )

    plt.grid(True, alpha=0.3)

    output_path = os.path.join(
        TRACK_RESULT_DIR,
        f"{sequence_name}_tracking.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved tracking plot:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    tracks_045, df_045 = track_sequence(
        PRED_045,
        "ship_045"
    )

    plot_tracks(
        df_045,
        "ship_045"
    )

    tracks_047, df_047 = track_sequence(
        PRED_047,
        "ship_047"
    )

    plot_tracks(
        df_047,
        "ship_047"
    )

    print()
    print("=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)

    print()
    print("Results directory:")
    print(TRACK_RESULT_DIR)