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

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "final_event_tracking_videos"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SEQUENCES = {

    "ship_045": {

        "voxel_file": os.path.join(
            PROJECT_ROOT,
            "ship_045_event_dataset",
            "3_voxel_grid",
            "ship_045_voxel_grid.npy"
        ),

        "heatmap_file": os.path.join(
            SNN_PROJECT_DIR,
            "results",
            "temporal_snn_center_v3",
            "ship_045_predicted_heatmaps_v3.npy"
        ),

        "track_file": os.path.join(
            SNN_PROJECT_DIR,
            "results",
            "temporal_tracking_v5",
            "ship_045_tracks_filtered.csv"
        ),

        "output_file": os.path.join(
            OUTPUT_DIR,
            "ship_045_event_snn_tracking.mp4"
        )
    },

    "ship_047": {

        "voxel_file": os.path.join(
            SNN_PROJECT_DIR,
            "ship_047_event_dataset",
            "3_voxel_grid",
            "ship_047_voxel_grid.npy"
        ),

        "heatmap_file": os.path.join(
            SNN_PROJECT_DIR,
            "results",
            "temporal_snn_center_v3",
            "ship_047_predicted_heatmaps_v3.npy"
        ),

        "track_file": os.path.join(
            SNN_PROJECT_DIR,
            "results",
            "temporal_tracking_v5",
            "ship_047_tracks_filtered.csv"
        ),

        "output_file": os.path.join(
            OUTPUT_DIR,
            "ship_047_event_snn_tracking.mp4"
        )
    }
}


# ============================================================
# SETTINGS
# ============================================================

GRID_SIZE = 128

FPS = 10

TRAIL_LENGTH = 15

POINT_RADIUS = 5

LINE_THICKNESS = 2

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# CREATE EVENT IMAGE
# ============================================================

def create_event_image(
    voxel,
    scale=255
):

    """
    Convert a 2-channel event voxel [polarity, y, x]
    into a visible event image.

    Channel 0 = negative polarity
    Channel 1 = positive polarity
    """

    negative = voxel[0]

    positive = voxel[1]

    # Normalize independently.
    neg_max = negative.max()

    pos_max = positive.max()

    if neg_max > 0:

        negative = (
            negative / neg_max
        )

    if pos_max > 0:

        positive = (
            positive / pos_max
        )

    image = np.zeros(
        (
            GRID_SIZE,
            GRID_SIZE,
            3
        ),
        dtype=np.uint8
    )

    # Positive events → green
    image[:, :, 1] = (
        positive * scale
    ).astype(np.uint8)

    # Negative events → red
    image[:, :, 2] = (
        negative * scale
    ).astype(np.uint8)

    return image


# ============================================================
# CREATE HEATMAP IMAGE
# ============================================================

def create_heatmap_image(
    heatmap
):

    """
    Convert SNN predicted heatmap to
    an 8-bit visualization.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    image = (
        heatmap * 255
    ).astype(
        np.uint8
    )

    # Apply OpenCV colormap only for visualization.
    colored = cv2.applyColorMap(
        image,
        cv2.COLORMAP_JET
    )

    return colored


# ============================================================
# DRAW TRACKS
# ============================================================

def draw_tracks(
    image,
    tracks,
    current_bin
):

    if len(tracks) == 0:

        return image


    # Only tracks up to current bin.
    past_tracks = tracks[
        tracks["bin_id"] <= current_bin
    ]


    for track_id in sorted(
        past_tracks["track_id"].unique()
    ):

        track = past_tracks[
            past_tracks["track_id"]
            ==
            track_id
        ].copy()


        # Only recent trajectory.
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

            px = int(
                round(x)
            )

            py = int(
                round(y)
            )

            points.append(
                (
                    px,
                    py
                )
            )


        # ----------------------------------------------------
        # Draw trajectory
        # ----------------------------------------------------

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                image,
                points[i - 1],
                points[i],
                (255, 255, 255),
                LINE_THICKNESS
            )


        # ----------------------------------------------------
        # Current position
        # ----------------------------------------------------

        current = track.iloc[-1]

        px = int(
            round(
                float(
                    current["x"]
                )
            )
        )

        py = int(
            round(
                float(
                    current["y"]
                )
            )
        )


        cv2.circle(
            image,
            (
                px,
                py
            ),
            POINT_RADIUS,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Track label
        # ----------------------------------------------------

        label = (
            f"T{int(track_id)}"
        )


        cv2.putText(
            image,
            label,
            (
                px + 6,
                py - 6
            ),
            FONT,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


    return image


# ============================================================
# ADD TITLE
# ============================================================

def add_title(
    image,
    title
):

    cv2.rectangle(
        image,
        (0, 0),
        (
            image.shape[1],
            25
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        title,
        (5, 17),
        FONT,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


# ============================================================
# ADD INFORMATION
# ============================================================

def add_information(
    image,
    text,
    y
):

    cv2.putText(
        image,
        text,
        (5, y),
        FONT,
        0.38,
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
        f"CREATING EVENT-BASED SNN VIDEO: "
        f"{sequence_name}"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Load voxel grid
    # --------------------------------------------------------

    voxels = np.load(
        config["voxel_file"]
    )

    print(
        "Voxel grid:",
        voxels.shape
    )


    # --------------------------------------------------------
    # Load SNN heatmaps
    # --------------------------------------------------------

    heatmaps = np.load(
        config["heatmap_file"]
    )

    print(
        "SNN heatmaps:",
        heatmaps.shape
    )


    # --------------------------------------------------------
    # Load filtered tracks
    # --------------------------------------------------------

    tracks = pd.read_csv(
        config["track_file"]
    )

    print(
        "Tracking detections:",
        len(tracks)
    )

    print(
        "Tracks:",
        sorted(
            tracks["track_id"]
            .unique()
        )
    )


    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if voxels.shape[0] != 50:

        raise ValueError(
            "Expected 50 temporal bins."
        )


    if heatmaps.shape[0] != 50:

        raise ValueError(
            "Expected 50 SNN heatmaps."
        )


    # --------------------------------------------------------
    # Create video
    # --------------------------------------------------------

    #
    # Layout:
    #
    # +----------------+----------------+
    # | Event Input    | SNN Heatmap     |
    # |                |                |
    # +----------------+----------------+
    # | SNN Tracking                    |
    # |                                 |
    # +---------------------------------+
    #

    panel_width = GRID_SIZE

    panel_height = GRID_SIZE


    video_width = (
        panel_width * 2
    )

    video_height = (
        panel_height * 2
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        config["output_file"],
        fourcc,
        FPS,
        (
            video_width,
            video_height
        )
    )


    # --------------------------------------------------------
    # Process all temporal bins
    # --------------------------------------------------------

    for bin_id in range(50):


        # ====================================================
        # PANEL 1
        # EVENT INPUT
        # ====================================================

        event_image = create_event_image(
            voxels[bin_id]
        )

        event_image = add_title(
            event_image,
            "EVENT INPUT"
        )


        add_information(
            event_image,
            f"Bin {bin_id + 1}/50",
            123
        )


        # ====================================================
        # PANEL 2
        # SNN HEATMAP
        # ====================================================

        heatmap_image = create_heatmap_image(
            heatmaps[bin_id]
        )

        heatmap_image = add_title(
            heatmap_image,
            "SNN PREDICTED HEATMAP"
        )


        max_score = float(
            heatmaps[bin_id].max()
        )


        add_information(
            heatmap_image,
            f"Max: {max_score:.3f}",
            123
        )


        # ====================================================
        # PANEL 3
        # TRACKING
        # ====================================================

        tracking_image = event_image.copy()


        tracking_image = draw_tracks(
            tracking_image,
            tracks,
            bin_id
        )


        tracking_image = add_title(
            tracking_image,
            "SNN + TRACKING"
        )


        # Show active tracks.

        active_tracks = tracks[
            tracks["bin_id"] <= bin_id
        ]


        active_ids = []


        if len(active_tracks) > 0:

            active_ids = sorted(
                active_tracks[
                    "track_id"
                ].unique()
            )


        active_text = (
            "Tracks: "
            +
            (
                ", ".join(
                    [
                        f"T{int(x)}"
                        for x in active_ids
                    ]
                )
                if active_ids
                else "None"
            )
        )


        add_information(
            tracking_image,
            active_text,
            123
        )


        # ====================================================
        # PANEL 4
        # PIPELINE INFORMATION
        # ====================================================

        info = np.zeros(
            (
                GRID_SIZE,
                GRID_SIZE,
                3
            ),
            dtype=np.uint8
        )


        cv2.rectangle(
            info,
            (0, 0),
            (
                GRID_SIZE,
                GRID_SIZE
            ),
            (15, 15, 15),
            -1
        )


        cv2.putText(
            info,
            "SNN PIPELINE",
            (8, 18),
            FONT,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        pipeline_lines = [

            "Event Stream",

            "     ↓",

            "Voxel Grid",

            "     ↓",

            "SNN",

            "     ↓",

            "Heatmap",

            "     ↓",

            "Peak Detection",

            "     ↓",

            "V5 Tracker",

            "     ↓",

            "Trajectory"
        ]


        y = 35


        for line in pipeline_lines:

            cv2.putText(
                info,
                line,
                (8, y),
                FONT,
                0.34,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            y += 9


        # ====================================================
        # COMBINE PANELS
        # ====================================================

        top = np.hstack(
            [
                event_image,
                heatmap_image
            ]
        )


        bottom = np.hstack(
            [
                tracking_image,
                info
            ]
        )


        final_frame = np.vstack(
            [
                top,
                bottom
            ]
        )


        # ====================================================
        # GLOBAL HEADER
        # ====================================================

        cv2.putText(
            final_frame,
            f"{sequence_name.upper()} | "
            f"EVENT-BASED SNN OBJECT TRACKING",
            (
                6,
                246
            ),
            FONT,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        writer.write(
            final_frame
        )


    writer.release()


    print()
    print(
        "VIDEO SAVED:"
    )

    print(
        config["output_file"]
    )

    print(
        "Size:",
        round(
            os.path.getsize(
                config["output_file"]
            ) / (
                1024 * 1024
            ),
            3
        ),
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
        "EVENT-BASED SNN VISUALIZATION COMPLETE"
    )
    print("=" * 80)