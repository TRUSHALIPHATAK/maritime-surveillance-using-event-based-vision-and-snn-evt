import os
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SNN_PROJECT_DIR)

# ------------------------------------------------------------
# Ship 045
# ------------------------------------------------------------

VOXEL_045 = os.path.join(
    PROJECT_ROOT,
    "ship_045_event_dataset",
    "3_voxel_grid",
    "ship_045_voxel_grid.npy"
)

HEATMAP_045 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_045_predicted_heatmaps_v3.npy"
)

TRACK_045 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5",
    "ship_045_tracks_filtered.csv"
)

# ------------------------------------------------------------
# Ship 047
# ------------------------------------------------------------

VOXEL_047 = os.path.join(
    SNN_PROJECT_DIR,
    "ship_047_event_dataset",
    "3_voxel_grid",
    "ship_047_voxel_grid.npy"
)

HEATMAP_047 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_047_predicted_heatmaps_v3.npy"
)

TRACK_047 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5",
    "ship_047_tracks_filtered.csv"
)

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "snn_tracking_videos"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FPS = 10

FRAME_SIZE = 512

# Event visualization
EVENT_SCALE = 4

# Heatmap threshold only for visualization
HEATMAP_THRESHOLD = 0.50

# Detection marker
POINT_RADIUS = 7

# Text
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# LOAD TRACKING DATA
# ============================================================

def load_tracks(track_file):

    data = np.genfromtxt(
        track_file,
        delimiter=",",
        names=True
    )

    tracks = {}

    for row in data:

        track_id = int(row["track_id"])
        bin_id = int(row["bin_id"])

        x = float(row["x"])
        y = float(row["y"])

        score = float(row["score"])

        if track_id not in tracks:
            tracks[track_id] = {}

        tracks[track_id][bin_id] = {
            "x": x,
            "y": y,
            "score": score
        }

    return tracks


# ============================================================
# EVENT FRAME
# ============================================================

def create_event_image(voxel):

    # voxel shape:
    # (2, 128, 128)

    positive = voxel[0]
    negative = voxel[1]

    height, width = positive.shape

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # Normalize positive events
    if positive.max() > 0:
        pos = positive / positive.max()
    else:
        pos = positive

    # Normalize negative events
    if negative.max() > 0:
        neg = negative / negative.max()
    else:
        neg = negative

    # Positive = green
    image[:, :, 1] = np.clip(
        pos * 255,
        0,
        255
    ).astype(np.uint8)

    # Negative = red
    image[:, :, 2] = np.clip(
        neg * 255,
        0,
        255
    ).astype(np.uint8)

    image = cv2.resize(
        image,
        (FRAME_SIZE, FRAME_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    return image


# ============================================================
# HEATMAP IMAGE
# ============================================================

def create_heatmap_image(heatmap):

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    heatmap_uint8 = (
        heatmap * 255
    ).astype(np.uint8)

    # Apply OpenCV heatmap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.resize(
        heatmap_color,
        (FRAME_SIZE, FRAME_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    return heatmap_color


# ============================================================
# DRAW TRACK IDS ON HEATMAP
# ============================================================

def draw_tracks(
    heatmap_image,
    tracks,
    bin_id
):

    output = heatmap_image.copy()

    for track_id, track_data in tracks.items():

        if bin_id not in track_data:
            continue

        detection = track_data[bin_id]

        x = detection["x"]
        y = detection["y"]
        score = detection["score"]

        # 128 -> 512
        px = int(x * FRAME_SIZE / 128)
        py = int(y * FRAME_SIZE / 128)

        # Draw detection point
        cv2.circle(
            output,
            (px, py),
            POINT_RADIUS,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            output,
            (px, py),
            POINT_RADIUS + 2,
            (0, 0, 0),
            2
        )

        # Track label
        label = f"T{track_id}"

        cv2.putText(
            output,
            label,
            (px + 12, py - 8),
            FONT,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Score
        score_text = f"{score:.2f}"

        cv2.putText(
            output,
            score_text,
            (px + 12, py + 14),
            FONT,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(
    name,
    voxel_file,
    heatmap_file,
    track_file
):

    print()
    print("=" * 80)
    print(f"CREATING SNN + TRACKING VIDEO: {name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    voxel = np.load(voxel_file)
    heatmaps = np.load(heatmap_file)

    tracks = load_tracks(track_file)

    print(f"Voxel grid       : {voxel.shape}")
    print(f"SNN heatmaps     : {heatmaps.shape}")
    print(f"Number of tracks : {len(tracks)}")

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_snn_tracking.mp4"
    )

    # Two panels side by side
    video_width = FRAME_SIZE * 2
    video_height = FRAME_SIZE + 70

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_file,
        fourcc,
        FPS,
        (video_width, video_height)
    )

    # --------------------------------------------------------
    # Process 50 temporal bins
    # --------------------------------------------------------

    num_bins = min(
        len(voxel),
        len(heatmaps)
    )

    for bin_id in range(num_bins):

        # ----------------------------------------------------
        # EVENT INPUT
        # ----------------------------------------------------

        event_image = create_event_image(
            voxel[bin_id]
        )

        # ----------------------------------------------------
        # SNN HEATMAP
        # ----------------------------------------------------

        heatmap = heatmaps[bin_id]

        heatmap_image = create_heatmap_image(
            heatmap
        )

        # ----------------------------------------------------
        # TRACKING OVERLAY
        # ----------------------------------------------------

        heatmap_image = draw_tracks(
            heatmap_image,
            tracks,
            bin_id
        )

        # ----------------------------------------------------
        # PANEL TITLES
        # ----------------------------------------------------

        cv2.rectangle(
            event_image,
            (0, 0),
            (FRAME_SIZE, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            event_image,
            "EVENT INPUT",
            (20, 38),
            FONT,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.rectangle(
            heatmap_image,
            (0, 0),
            (FRAME_SIZE, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            heatmap_image,
            "SNN + TRACKING",
            (20, 38),
            FONT,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # COMBINE PANELS
        # ----------------------------------------------------

        combined = np.hstack(
            [
                event_image,
                heatmap_image
            ]
        )

        # ----------------------------------------------------
        # INFORMATION BAR
        # ----------------------------------------------------

        info_bar = np.zeros(
            (70, video_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            info_bar,
            f"Temporal Bin: {bin_id + 1}/{num_bins}",
            (20, 30),
            FONT,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        max_value = float(
            heatmap.max()
        )

        cv2.putText(
            info_bar,
            f"SNN Max Response: {max_value:.3f}",
            (450, 30),
            FONT,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            info_bar,
            f"Active Tracks: {sum(bin_id in t for t in tracks.values())}",
            (950, 30),
            FONT,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # FINAL FRAME
        # ----------------------------------------------------

        final_frame = np.vstack(
            [
                combined,
                info_bar
            ]
        )

        writer.write(final_frame)

    writer.release()

    print()
    print("VIDEO SAVED:")
    print(output_file)

    file_size = os.path.getsize(
        output_file
    ) / (1024 * 1024)

    print(
        f"Size: {file_size:.3f} MB"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_video(
        "ship_045",
        VOXEL_045,
        HEATMAP_045,
        TRACK_045
    )

    create_video(
        "ship_047",
        VOXEL_047,
        HEATMAP_047,
        TRACK_047
    )

    print()
    print("=" * 80)
    print("SNN + TRACKING VISUALIZATION COMPLETE")
    print("=" * 80)