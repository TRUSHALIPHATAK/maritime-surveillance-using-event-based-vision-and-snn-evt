import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "evt_visualization_v4"
)

SEQUENCES = {
    "045": {
        "voxel": os.path.join(
            BASE_DIR,
            "..",
            "ship_045_event_dataset",
            "3_voxel_grid",
            "ship_045_voxel_grid.npy"
        ),
        "tracks": os.path.join(
            BASE_DIR,
            "results",
            "mot_evt_v4",
            "ship_045_evt_tracks_v4.csv"
        )
    },

    "047": {
        "voxel": os.path.join(
            BASE_DIR,
            "..",
            "snn_project",
            "ship_047_event_dataset",
            "3_voxel_grid",
            "ship_047_voxel_grid.npy"
        ),
        "tracks": os.path.join(
            BASE_DIR,
            "results",
            "mot_evt_v4",
            "ship_047_evt_tracks_v4.csv"
        )
    }
}


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

SCALE = 4

FPS = 2

UPSAMPLED_SIZE = 128 * SCALE

# Event display threshold.
# Increase this if frames look too dense.
EVENT_THRESHOLD = 0.0


# ============================================================
# EVENT FRAME CREATION
# ============================================================

def create_event_frame(event_group):
    """
    Convert a grouped event tensor:

        [2, 128, 128]

    into an event-only RGB image.

    Channel 0 = negative polarity
    Channel 1 = positive polarity
    """

    negative = event_group[0]
    positive = event_group[1]

    # Normalize each polarity independently
    if np.max(negative) > 0:
        negative = negative / np.max(negative)

    if np.max(positive) > 0:
        positive = positive / np.max(positive)

    frame = np.zeros(
        (128, 128, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Positive events
    # Displayed as bright red
    # --------------------------------------------------------

    positive_mask = positive > EVENT_THRESHOLD

    frame[positive_mask, 2] = (
        positive[positive_mask] * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Negative events
    # Displayed as bright blue
    # --------------------------------------------------------

    negative_mask = negative > EVENT_THRESHOLD

    frame[negative_mask, 0] = (
        negative[negative_mask] * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # If both polarities occur at same location,
    # combine them.
    # --------------------------------------------------------

    both_mask = positive_mask & negative_mask

    frame[both_mask] = np.array(
        [255, 0, 255],
        dtype=np.uint8
    )

    return frame


# ============================================================
# TRACK DRAWING
# ============================================================

def draw_tracks(frame, tracks, group_id):
    """
    Draw V4 tracks on top of the event-only frame.
    """

    group_tracks = tracks[
        tracks["group_id"] == group_id
    ]

    for _, track in group_tracks.iterrows():

        x = int(round(track["x"]))
        y = int(round(track["y"]))

        track_id = int(track["track_id"])

        confidence = float(
            track["confidence"]
        )

        status = str(
            track["status"]
        )

        # ----------------------------------------------------
        # Scale coordinates
        # ----------------------------------------------------

        sx = x * SCALE
        sy = y * SCALE

        # ----------------------------------------------------
        # Draw center
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (sx, sy),
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (sx, sy),
            9,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # Track label
        # ----------------------------------------------------

        label = (
            f"ID {track_id} "
            f"{confidence:.2f}"
        )

        if status.upper() == "INITIAL":
            label += " INITIAL"

        label_x = max(
            5,
            sx + 8
        )

        label_y = max(
            18,
            sy - 8
        )

        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return frame


# ============================================================
# HEADER
# ============================================================

def draw_header(frame, sequence, group_id):

    text = (
        f"EVS | EVT-MOT V4 | "
        f"Sequence {sequence} | "
        f"Group {group_id:02d}/09"
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 32),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def process_sequence(sequence):

    print()
    print("=" * 70)
    print(f"EVS VISUALIZATION — SEQUENCE {sequence}")
    print("=" * 70)

    voxel_path = SEQUENCES[sequence]["voxel"]
    tracks_path = SEQUENCES[sequence]["tracks"]

    print()
    print("Voxel:")
    print(voxel_path)

    print()
    print("Tracks:")
    print(tracks_path)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(voxel_path):
        raise FileNotFoundError(
            f"Voxel file not found:\n{voxel_path}"
        )

    if not os.path.exists(tracks_path):
        raise FileNotFoundError(
            f"Track file not found:\n{tracks_path}"
        )

    # --------------------------------------------------------
    # Load voxel
    # --------------------------------------------------------

    voxel = np.load(voxel_path)

    print()
    print("Voxel shape:", voxel.shape)

    if voxel.shape != (50, 2, 128, 128):
        raise ValueError(
            f"Unexpected voxel shape: {voxel.shape}"
        )

    # --------------------------------------------------------
    # Load V4 tracks
    # --------------------------------------------------------

    tracks = pd.read_csv(tracks_path)

    print(
        "Track rows:",
        len(tracks)
    )

    print(
        "Track columns:",
        list(tracks.columns)
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    sequence_output = os.path.join(
        OUTPUT_DIR,
        f"ship_{sequence}"
    )

    frames_dir = os.path.join(
        sequence_output,
        "frames"
    )

    os.makedirs(
        frames_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    video_path = os.path.join(
        sequence_output,
        f"ship_{sequence}_evt_v4_tracking.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        video_path,
        fourcc,
        FPS,
        (
            UPSAMPLED_SIZE,
            UPSAMPLED_SIZE + 32
        )
    )

    # --------------------------------------------------------
    # 50 bins -> 10 temporal groups
    # Each group = 5 bins
    # --------------------------------------------------------

    for group_id in range(10):

        start_bin = group_id * 5
        end_bin = start_bin + 5

        # ----------------------------------------------------
        # Combine five temporal bins
        # ----------------------------------------------------

        event_group = voxel[
            start_bin:end_bin
        ].sum(axis=0)

        # ----------------------------------------------------
        # Create pure event frame
        # ----------------------------------------------------

        frame = create_event_frame(
            event_group
        )

        # ----------------------------------------------------
        # Upscale event frame
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            (
                UPSAMPLED_SIZE,
                UPSAMPLED_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Draw V4 tracks
        # ----------------------------------------------------

        frame = draw_tracks(
            frame,
            tracks,
            group_id
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        frame = draw_header(
            frame,
            sequence,
            group_id
        )

        # ----------------------------------------------------
        # Save PNG
        # ----------------------------------------------------

        frame_path = os.path.join(
            frames_dir,
            f"group_{group_id:02d}.png"
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        # ----------------------------------------------------
        # Add to video
        # ----------------------------------------------------

        writer.write(frame)

        print(
            f"Group {group_id:02d}: "
            f"bins {start_bin:02d}-{end_bin - 1:02d} "
            f"-> {frame_path}"
        )

    writer.release()

    print()
    print("Frames saved:")
    print(frames_dir)

    print()
    print("Video saved:")
    print(video_path)

    print()
    print("DONE")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for sequence in ["045", "047"]:
        process_sequence(sequence)

    print()
    print("=" * 70)
    print("EVS VISUALIZATION COMPLETE")
    print("=" * 70)