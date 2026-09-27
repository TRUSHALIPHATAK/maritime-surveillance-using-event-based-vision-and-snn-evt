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
    "temporal_tracking_v3"
)

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "tracking_videos"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET PATHS
# ============================================================

IMAGE_DIR_045 = os.path.join(
    PROJECT_ROOT,
    "data_samples",
    "interp",
    "ship_045_full"
)

IMAGE_DIR_047 = os.path.join(
    PROJECT_ROOT,
    "data_samples",
    "interp",
    "ship_047_full"
)


TIME_BIN_FILE_045 = os.path.join(
    PROJECT_ROOT,
    "ship_045_event_dataset",
    "4_time_bins",
    "ship_045_time_bins.csv"
)

TIME_BIN_FILE_047 = os.path.join(
    SNN_PROJECT_DIR,
    "ship_047_event_dataset",
    "4_time_bins",
    "ship_047_time_bins.csv"
)


INFO_FILE_045 = os.path.join(
    PROJECT_ROOT,
    "data_samples",
    "interp",
    "ship_045_full",
    "info.txt"
)

INFO_FILE_047 = os.path.join(
    PROJECT_ROOT,
    "data_samples",
    "interp",
    "ship_047_full",
    "info.txt"
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 10

# Number of previous tracking positions displayed
TRAIL_LENGTH = 15

# Current detection circle
POINT_RADIUS = 8

# Trail thickness
LINE_THICKNESS = 3

# Text size
FONT_SCALE = 0.65

# How many image frames after the last detection
# the track remains visible.
#
# This is only for visualization.
# It does NOT create new detections.
MAX_VISUAL_GAP_BINS = 2


# ============================================================
# TRACK COLORS
# ============================================================

TRACK_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
    (128, 255, 0),
    (255, 0, 128),
    (128, 128, 255),
    (0, 255, 128),
]


# ============================================================
# IMAGE FILES
# ============================================================

def get_image_files(image_dir):

    files = []

    for filename in os.listdir(image_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            files.append(filename)

    # Important:
    # filenames are zero-padded, so normal sorting works.
    files.sort()

    return [
        os.path.join(image_dir, filename)
        for filename in files
    ]


# ============================================================
# READ ORIGINAL IMAGE TIMESTAMPS
# ============================================================

def load_image_timestamps(info_file):

    """
    Reads info.txt.

    Supports entries such as:

        000001.jpg 0

    or:

        data_samples/interp/ship_045_full/000001.jpg 0

    Returns:

        frame_indices
        timestamps_us
    """

    if not os.path.exists(info_file):

        raise FileNotFoundError(
            f"info.txt not found:\n{info_file}"
        )

    frame_indices = []
    timestamps_us = []

    with open(info_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            # ------------------------------------------------
            # The first field may contain the complete path.
            # We only need the actual filename.
            # ------------------------------------------------

            filename = os.path.basename(parts[0])

            timestamp = int(parts[1])

            # ------------------------------------------------
            # Extract frame number
            #
            # Example:
            # 000001.jpg -> 1
            # ------------------------------------------------

            filename_without_extension = os.path.splitext(
                filename
            )[0]

            frame_number = int(
                filename_without_extension
            )

            frame_indices.append(
                frame_number - 1
            )

            timestamps_us.append(
                timestamp
            )

    frame_indices = np.array(
        frame_indices,
        dtype=np.int64
    )

    timestamps_us = np.array(
        timestamps_us,
        dtype=np.int64
    )

    if len(timestamps_us) == 0:

        raise ValueError(
            f"No valid timestamp entries found in:\n"
            f"{info_file}"
        )

    print(
        f"Loaded {len(timestamps_us)} image timestamps"
    )

    print(
        f"Image timestamp range: "
        f"{timestamps_us[0]} -> "
        f"{timestamps_us[-1]} us"
    )

    return (
        frame_indices,
        timestamps_us
    )

# ============================================================
# READ EVENT BIN TIMESTAMPS
# ============================================================

def load_event_bin_timestamps(csv_file):

    """
    Reads 4_time_bins CSV.

    Expected columns include:

        bin_id
        start_time
        end_time

    or equivalent timestamp columns.

    The function tries to detect the correct columns
    automatically.
    """

    if not os.path.exists(csv_file):

        raise FileNotFoundError(
            f"Time-bin file not found:\n{csv_file}"
        )

    df = pd.read_csv(csv_file)

    print(
        f"Loaded time-bin file: {os.path.basename(csv_file)}"
    )

    print(
        "Columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Find bin ID column
    # --------------------------------------------------------

    bin_candidates = [
        "bin_id",
        "bin",
        "bin_index",
        "id"
    ]

    bin_col = None

    for col in bin_candidates:

        if col in df.columns:

            bin_col = col
            break

    if bin_col is None:

        # Fall back to row number
        df["bin_id"] = np.arange(len(df))

        bin_col = "bin_id"


    # --------------------------------------------------------
    # Find start timestamp
    # --------------------------------------------------------

    start_candidates = [
        "start_time",
        "start_timestamp",
        "start_us",
        "start",
        "timestamp_start"
    ]

    start_col = None

    for col in start_candidates:

        if col in df.columns:

            start_col = col
            break


    # --------------------------------------------------------
    # Find end timestamp
    # --------------------------------------------------------

    end_candidates = [
        "end_time",
        "end_timestamp",
        "end_us",
        "end",
        "timestamp_end"
    ]

    end_col = None

    for col in end_candidates:

        if col in df.columns:

            end_col = col
            break


    # --------------------------------------------------------
    # If columns have different names, identify numeric
    # timestamp-like columns.
    # --------------------------------------------------------

    if start_col is None or end_col is None:

        numeric_cols = []

        for col in df.columns:

            if col == bin_col:
                continue

            try:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

                if values.notna().all():

                    numeric_cols.append(col)

            except Exception:
                pass

        if len(numeric_cols) >= 2:

            if start_col is None:
                start_col = numeric_cols[0]

            if end_col is None:
                end_col = numeric_cols[1]


    if start_col is None or end_col is None:

        raise ValueError(
            "Could not identify start/end timestamp "
            f"columns in:\n{csv_file}\n\n"
            f"Columns found: {list(df.columns)}"
        )


    # --------------------------------------------------------
    # Extract timestamps
    # --------------------------------------------------------

    bin_ids = pd.to_numeric(
        df[bin_col]
    ).astype(int).to_numpy()

    start_times = pd.to_numeric(
        df[start_col]
    ).to_numpy(dtype=np.int64)

    end_times = pd.to_numeric(
        df[end_col]
    ).to_numpy(dtype=np.int64)


    # Use center timestamp of each event bin.
    # This represents the temporal location of the bin.
    center_times = (
        start_times.astype(np.float64)
        + end_times.astype(np.float64)
    ) / 2.0

    center_times = center_times.astype(
        np.int64
    )


    print(
        f"Event bins: {len(bin_ids)}"
    )

    print(
        f"Event timestamp range: "
        f"{start_times[0]} -> {end_times[-1]} us"
    )

    print(
        f"Bin duration approximately: "
        f"{np.mean(end_times - start_times):.1f} us"
    )

    return (
        bin_ids,
        start_times,
        end_times,
        center_times
    )


# ============================================================
# BIN → ACTUAL IMAGE FRAME
# ============================================================

def create_bin_to_frame_mapping(
    image_timestamps,
    bin_center_times
):

    """
    Maps every event bin to the original image frame
    whose timestamp is closest to the event-bin center.

    This is the important correction.

    We DO NOT assume:

        bin 0 -> frame 0
        bin 1 -> frame 6
        ...

    Instead:

        event-bin timestamp
              ↓
        nearest image timestamp
              ↓
        actual image frame
    """

    mapping = {}

    for bin_index, event_time in enumerate(
        bin_center_times
    ):

        # Find nearest image timestamp
        differences = np.abs(
            image_timestamps.astype(np.int64)
            - int(event_time)
        )

        frame_index = int(
            np.argmin(differences)
        )

        mapping[bin_index] = frame_index


    return mapping


# ============================================================
# PRINT BIN / FRAME ALIGNMENT
# ============================================================

def print_alignment_table(
    bin_to_frame,
    image_timestamps,
    bin_center_times
):

    print()
    print("-" * 70)
    print("EVENT BIN → IMAGE FRAME ALIGNMENT")
    print("-" * 70)

    selected_bins = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        49
    ]

    max_bin = len(bin_center_times) - 1

    for bin_id in selected_bins:

        if bin_id > max_bin:
            continue

        frame_index = bin_to_frame[bin_id]

        event_time = bin_center_times[bin_id]

        image_time = image_timestamps[frame_index]

        error = abs(
            int(event_time)
            - int(image_time)
        )

        print(
            f"Bin {bin_id:02d}  →  "
            f"Frame {frame_index + 1:03d}  | "
            f"event={event_time:9d} us | "
            f"image={image_time:9d} us | "
            f"error={error:6d} us"
        )

    print("-" * 70)
    print()


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def event_to_image_coordinates(
    x,
    y,
    width,
    height
):

    """
    Tracking coordinates are in 128x128 space.

    Convert them to original image coordinates.

    128x128:
        x = 0 ... 127
        y = 0 ... 127

    Original:
        x = 0 ... width-1
        y = 0 ... height-1
    """

    original_x = int(
        round(
            float(x)
            * (width - 1)
            / 127.0
        )
    )

    original_y = int(
        round(
            float(y)
            * (height - 1)
            / 127.0
        )
    )

    original_x = max(
        0,
        min(
            original_x,
            width - 1
        )
    )

    original_y = max(
        0,
        min(
            original_y,
            height - 1
        )
    )

    return original_x, original_y


# ============================================================
# DRAW TRACK
# ============================================================

def draw_track(
    frame,
    track,
    current_bin,
    width,
    height,
    color
):

    """
    Draw the trajectory of one track.

    Important:
    We draw only detections that have already happened.

    We do not create fake future detections.
    """

    track = track.sort_values(
        "bin_id"
    ).copy()


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = track[
        track["bin_id"] <= current_bin
    ].tail(TRAIL_LENGTH)


    if len(history) == 0:

        return


    # --------------------------------------------------------
    # Convert all points to image coordinates
    # --------------------------------------------------------

    points = []

    for _, row in history.iterrows():

        x = float(row["x"])
        y = float(row["y"])

        px, py = event_to_image_coordinates(
            x,
            y,
            width,
            height
        )

        points.append(
            (px, py)
        )


    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------

    for i in range(
        1,
        len(points)
    ):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            color,
            LINE_THICKNESS,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Find most recent detection
    # --------------------------------------------------------

    latest = history.iloc[-1]

    latest_bin = int(
        latest["bin_id"]
    )


    # --------------------------------------------------------
    # Determine whether this track is still active
    # --------------------------------------------------------

    bin_gap = (
        current_bin
        - latest_bin
    )


    # If the detection is too old, don't show current marker.
    #
    # We still show the historical trail.
    if bin_gap > MAX_VISUAL_GAP_BINS:

        return


    # --------------------------------------------------------
    # Current position
    # --------------------------------------------------------

    x = float(latest["x"])
    y = float(latest["y"])

    px, py = event_to_image_coordinates(
        x,
        y,
        width,
        height
    )


    # --------------------------------------------------------
    # Detection score
    # --------------------------------------------------------

    score = float(
        latest["score"]
    )


    # --------------------------------------------------------
    # Current position marker
    # --------------------------------------------------------

    cv2.circle(
        frame,
        (px, py),
        POINT_RADIUS,
        color,
        -1
    )

    cv2.circle(
        frame,
        (px, py),
        POINT_RADIUS + 4,
        color,
        2
    )


    # --------------------------------------------------------
    # Track label
    # --------------------------------------------------------

    label = (
        f"Track {int(latest['track_id'])} "
        f"{score:.2f}"
    )


    cv2.putText(
        frame,
        label,
        (px + 12, py - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(
    sequence_name,
    image_dir,
    info_file,
    time_bin_file
):

    print()
    print("=" * 70)
    print(
        f"CREATING TRACKING VIDEO: {sequence_name}"
    )
    print("=" * 70)


    # ========================================================
    # LOAD IMAGES
    # ========================================================

    image_files = get_image_files(
        image_dir
    )

    if len(image_files) == 0:

        raise FileNotFoundError(
            f"No images found:\n{image_dir}"
        )

    print(
        "Images:",
        len(image_files)
    )


    # ========================================================
    # LOAD TRACKING CSV
    # ========================================================

    csv_path = os.path.join(
        TRACKING_DIR,
        f"{sequence_name}_tracks.csv"
    )

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"Tracking CSV not found:\n{csv_path}"
        )

    tracking_df = pd.read_csv(
        csv_path
    )

    print(
        "Tracking rows:",
        len(tracking_df)
    )

    print(
        "Tracking columns:",
        list(tracking_df.columns)
    )


    # ========================================================
    # LOAD IMAGE TIMESTAMPS
    # ========================================================

    (
        frame_indices,
        image_timestamps
    ) = load_image_timestamps(
        info_file
    )


    if len(image_timestamps) != len(
        image_files
    ):

        raise ValueError(
            "Number of timestamps does not match "
            f"number of images.\n"
            f"Images: {len(image_files)}\n"
            f"Timestamps: {len(image_timestamps)}"
        )


    # ========================================================
    # LOAD EVENT BIN TIMESTAMPS
    # ========================================================

    (
        bin_ids,
        bin_start_times,
        bin_end_times,
        bin_center_times
    ) = load_event_bin_timestamps(
        time_bin_file
    )


    # ========================================================
    # CREATE ACTUAL BIN → FRAME MAPPING
    # ========================================================

    bin_to_frame = create_bin_to_frame_mapping(
        image_timestamps,
        bin_center_times
    )


    print_alignment_table(
        bin_to_frame,
        image_timestamps,
        bin_center_times
    )


    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    first_frame = cv2.imread(
        image_files[0]
    )

    if first_frame is None:

        raise RuntimeError(
            f"Could not read:\n"
            f"{image_files[0]}"
        )

    height, width = first_frame.shape[:2]

    print(
        "Frame size:",
        f"{width} x {height}"
    )


    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{sequence_name}_snn_tracking_fixed.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        FPS,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not open video writer."
        )


    # ========================================================
    # TRACK IDs
    # ========================================================

    if len(tracking_df) > 0:

        track_ids = sorted(
            tracking_df[
                "track_id"
            ].unique()
        )

    else:

        track_ids = []


    print(
        "Tracks:",
        track_ids
    )


    # ========================================================
    # PROCESS EVERY ORIGINAL IMAGE
    # ========================================================

    for frame_index, image_path in enumerate(
        image_files
    ):

        frame = cv2.imread(
            image_path
        )

        if frame is None:

            print(
                "Skipping unreadable frame:",
                image_path
            )

            continue


        # ----------------------------------------------------
        # Determine which event bin is associated with
        # this original image frame.
        #
        # We use the nearest event-bin timestamp.
        # ----------------------------------------------------

        image_time = image_timestamps[
            frame_index
        ]

        time_difference = np.abs(
            bin_center_times.astype(np.int64)
            - int(image_time)
        )

        bin_id = int(
            np.argmin(
                time_difference
            )
        )


        # ----------------------------------------------------
        # Draw all tracks
        # ----------------------------------------------------

        for track_id in track_ids:

            track = tracking_df[
                tracking_df["track_id"]
                == track_id
            ]

            color = TRACK_COLORS[
                (
                    int(track_id) - 1
                )
                % len(TRACK_COLORS)
            ]


            draw_track(
                frame,
                track,
                bin_id,
                width,
                height,
                color
            )


        # ====================================================
        # INFORMATION OVERLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (470, 85),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"SNN Ship Tracking - {sequence_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            (
                f"Frame: "
                f"{frame_index + 1}/"
                f"{len(image_files)}   "
                f"Event Bin: {bin_id:02d}"
            ),
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Timestamp: {image_time} us",
            (20, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(
            frame
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            (frame_index + 1) % 50 == 0
            or frame_index
            == len(image_files) - 1
        ):

            print(
                f"Processed "
                f"{frame_index + 1}/"
                f"{len(image_files)} frames"
            )


    # ========================================================
    # FINISH
    # ========================================================

    writer.release()

    print()
    print(
        "Video saved:"
    )

    print(
        output_path
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # SHIP 045
    # --------------------------------------------------------

    create_video(
        sequence_name="ship_045",
        image_dir=IMAGE_DIR_045,
        info_file=INFO_FILE_045,
        time_bin_file=TIME_BIN_FILE_045
    )


    # --------------------------------------------------------
    # SHIP 047
    # --------------------------------------------------------

    create_video(
        sequence_name="ship_047",
        image_dir=IMAGE_DIR_047,
        info_file=INFO_FILE_047,
        time_bin_file=TIME_BIN_FILE_047
    )


    print()
    print("=" * 70)
    print(
        "ALL FIXED TRACKING VIDEOS CREATED"
    )
    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )