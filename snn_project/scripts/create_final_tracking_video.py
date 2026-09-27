import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SNN_PROJECT_DIR)

TRACKING_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5"
)

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "final_tracking_videos"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Sequence-specific paths
# ------------------------------------------------------------

SEQUENCES = {

    "ship_045": {
        "track_file": os.path.join(
            TRACKING_DIR,
            "ship_045_tracks_filtered.csv"
        ),

        "image_dir": os.path.join(
            PROJECT_ROOT,
            "data_samples",
            "interp",
            "ship_045_full"
        ),

        "info_file": os.path.join(
            PROJECT_ROOT,
            "data_samples",
            "interp",
            "ship_045_full",
            "info.txt"
        ),

        "time_bin_file": os.path.join(
            PROJECT_ROOT,
            "ship_045_event_dataset",
            "4_time_bins",
            "ship_045_time_bins.csv"
        ),
    },

    "ship_047": {
        "track_file": os.path.join(
            TRACKING_DIR,
            "ship_047_tracks_filtered.csv"
        ),

        "image_dir": os.path.join(
            PROJECT_ROOT,
            "data_samples",
            "interp",
            "ship_047_full"
        ),

        "info_file": os.path.join(
            PROJECT_ROOT,
            "data_samples",
            "interp",
            "ship_047_full",
            "info.txt"
        ),

        "time_bin_file": os.path.join(
            SNN_PROJECT_DIR,
            "ship_047_event_dataset",
            "4_time_bins",
            "ship_047_time_bins.csv"
        ),
    }
}


# ============================================================
# VIDEO SETTINGS
# ============================================================

FPS = 10

IMAGE_WIDTH = 1345
IMAGE_HEIGHT = 451

EVENT_WIDTH = 128
EVENT_HEIGHT = 128

TRAIL_LENGTH = 15

POINT_RADIUS = 7
LINE_THICKNESS = 3

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# LOAD ORIGINAL IMAGE TIMESTAMPS
# ============================================================

def load_image_timestamps(info_file):

    frame_numbers = []
    timestamps = []

    with open(
        info_file,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            filename = os.path.basename(
                parts[0]
            )

            timestamp = int(
                parts[1]
            )

            stem = os.path.splitext(
                filename
            )[0]

            frame_number = int(stem)

            frame_numbers.append(
                frame_number
            )

            timestamps.append(
                timestamp
            )

    return (
        np.array(frame_numbers),
        np.array(timestamps)
    )


# ============================================================
# LOAD EVENT BIN TIMESTAMPS
# ============================================================

def load_event_bin_times(time_bin_file):

    df = pd.read_csv(
        time_bin_file
    )

    print()
    print(
        "Time-bin columns:",
        list(df.columns)
    )

    # Try to identify the time column.
    possible_columns = [
        "start_time_us",
        "start_timestamp_us",
        "start_time",
        "start",
        "timestamp"
    ]

    start_column = None

    for col in possible_columns:

        if col in df.columns:

            start_column = col
            break

    if start_column is None:

        raise ValueError(
            "Could not identify event-bin start timestamp column."
        )

    return df[start_column].astype(
        np.int64
    ).to_numpy()


# ============================================================
# MAP EVENT BIN → ORIGINAL IMAGE
# ============================================================

def build_bin_to_frame_mapping(
    image_timestamps,
    bin_timestamps
):

    mapping = []

    for bin_time in bin_timestamps:

        differences = np.abs(
            image_timestamps -
            bin_time
        )

        closest_index = int(
            np.argmin(
                differences
            )
        )

        mapping.append(
            closest_index
        )

    return np.array(
        mapping,
        dtype=np.int32
    )


# ============================================================
# EVENT COORDINATE → IMAGE COORDINATE
# ============================================================

def event_to_image(
    x,
    y
):

    image_x = (
        x *
        IMAGE_WIDTH /
        EVENT_WIDTH
    )

    image_y = (
        y *
        IMAGE_HEIGHT /
        EVENT_HEIGHT
    )

    return (
        int(round(image_x)),
        int(round(image_y))
    )


# ============================================================
# DRAW TRACKING
# ============================================================

def draw_tracks(
    frame,
    tracks,
    current_bin
):

    visible_tracks = tracks[
        tracks["bin_id"] <= current_bin
    ]

    for track_id in sorted(
        visible_tracks["track_id"].unique()
    ):

        track = visible_tracks[
            visible_tracks["track_id"]
            ==
            track_id
        ]

        # Only use recent trajectory points
        track = track[
            track["bin_id"]
            >=
            current_bin - TRAIL_LENGTH
        ]

        if len(track) == 0:
            continue

        points = []

        for _, row in track.iterrows():

            x = float(
                row["x"]
            )

            y = float(
                row["y"]
            )

            px, py = event_to_image(
                x,
                y
            )

            points.append(
                (px, py)
            )

        # ----------------------------------------------------
        # Draw trajectory
        # ----------------------------------------------------

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 255, 255),
                LINE_THICKNESS
            )

        # ----------------------------------------------------
        # Current position
        # ----------------------------------------------------

        current_row = track.iloc[-1]

        px, py = event_to_image(
            float(current_row["x"]),
            float(current_row["y"])
        )

        cv2.circle(
            frame,
            (px, py),
            POINT_RADIUS,
            (0, 255, 0),
            -1
        )

        # ----------------------------------------------------
        # Track label
        # ----------------------------------------------------

        label = (
            f"Track {int(track_id)}"
        )

        cv2.putText(
            frame,
            label,
            (
                px + 10,
                py - 10
            ),
            FONT,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if "score" in current_row:

            score = float(
                current_row["score"]
            )

            score_text = (
                f"{score:.2f}"
            )

            cv2.putText(
                frame,
                score_text,
                (
                    px + 10,
                    py + 15
                ),
                FONT,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(
    sequence_name,
    config
):

    print()
    print("=" * 80)
    print(
        f"CREATING FINAL VIDEO: {sequence_name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load tracks
    # --------------------------------------------------------

    tracks = pd.read_csv(
        config["track_file"]
    )

    print(
        f"Loaded {len(tracks)} tracking detections"
    )

    print(
        "Tracks:",
        sorted(
            tracks["track_id"]
            .unique()
        )
    )

    # --------------------------------------------------------
    # Load image timestamps
    # --------------------------------------------------------

    frame_numbers, image_timestamps = (
        load_image_timestamps(
            config["info_file"]
        )
    )

    print(
        f"Original frames: {len(frame_numbers)}"
    )

    # --------------------------------------------------------
    # Load event-bin timestamps
    # --------------------------------------------------------

    bin_timestamps = load_event_bin_times(
        config["time_bin_file"]
    )

    print(
        f"Event bins: {len(bin_timestamps)}"
    )

    # --------------------------------------------------------
    # Build correct temporal mapping
    # --------------------------------------------------------

    bin_to_frame = build_bin_to_frame_mapping(
        image_timestamps,
        bin_timestamps
    )

    print()
    print(
        "Bin → frame mapping:"
    )

    for bin_id in [
        0,
        1,
        5,
        10,
        20,
        30,
        40,
        49
    ]:

        if bin_id >= len(
            bin_to_frame
        ):
            continue

        frame_index = bin_to_frame[
            bin_id
        ]

        print(
            f"Bin {bin_id:02d}"
            f" → frame "
            f"{frame_numbers[frame_index]}"
            f" "
            f"(timestamp "
            f"{bin_timestamps[bin_id]}"
            f" us)"
        )

    # --------------------------------------------------------
    # Video output
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{sequence_name}_final_tracking.mp4"
    )

    first_image = cv2.imread(
        os.path.join(
            config["image_dir"],
            "000001.jpg"
        )
    )

    if first_image is None:

        raise FileNotFoundError(
            "Could not load first image."
        )

    height, width = first_image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_file,
        fourcc,
        FPS,
        (width, height)
    )

    # --------------------------------------------------------
    # Create one video frame per event bin
    # --------------------------------------------------------

    for bin_id in range(
        len(bin_timestamps)
    ):

        frame_index = bin_to_frame[
            bin_id
        ]

        frame_number = frame_numbers[
            frame_index
        ]

        image_path = os.path.join(
            config["image_dir"],
            f"{frame_number:06d}.jpg"
        )

        frame = cv2.imread(
            image_path
        )

        if frame is None:

            print(
                f"WARNING: Could not read "
                f"{image_path}"
            )

            continue

        # ----------------------------------------------------
        # Draw tracking
        # ----------------------------------------------------

        draw_tracks(
            frame,
            tracks,
            bin_id
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 55),
            (0, 0, 0),
            -1
        )

        title = (
            f"{sequence_name.upper()}  |  "
            f"Event Bin: {bin_id + 1}/"
            f"{len(bin_timestamps)}"
        )

        cv2.putText(
            frame,
            title,
            (20, 35),
            FONT,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Bottom information
        # ----------------------------------------------------

        timestamp_text = (
            f"Frame: {frame_number:06d}   "
            f"Time: {bin_timestamps[bin_id]} us"
        )

        cv2.putText(
            frame,
            timestamp_text,
            (20, height - 20),
            FONT,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        writer.write(
            frame
        )

    writer.release()

    print()
    print(
        "VIDEO SAVED:"
    )

    print(
        output_file
    )

    print(
        "Size:",
        os.path.getsize(
            output_file
        ) / (1024 * 1024),
        "MB"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for sequence_name, config in SEQUENCES.items():

        create_video(
            sequence_name,
            config
        )

    print()
    print("=" * 80)
    print(
        "FINAL TRACKING VIDEOS COMPLETE"
    )
    print("=" * 80)