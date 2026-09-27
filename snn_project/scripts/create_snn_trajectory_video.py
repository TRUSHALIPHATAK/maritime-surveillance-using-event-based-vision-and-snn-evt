import os
import cv2
import numpy as np

# ============================================================
# SNN + TRAJECTORY VIDEO
# Uses the already-generated filtered V5 tracking results.
# No training or model changes are performed here.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

TRACK_FILES = {
    "ship_045": os.path.join(
        SNN_PROJECT_DIR,
        "results",
        "temporal_tracking_v5",
        "ship_045_tracks_filtered.csv",
    ),
    "ship_047": os.path.join(
        SNN_PROJECT_DIR,
        "results",
        "temporal_tracking_v5",
        "ship_047_tracks_filtered.csv",
    ),
}

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "snn_trajectory_videos",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 10
SIZE = 700
PAD = 60
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors
COLORS = [
    (0, 255, 0),
    (0, 220, 255),
    (255, 140, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def load_tracks(csv_path):
    """Load filtered V5 tracks into {track_id: {bin_id: data}}."""

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Tracking file not found:\n{csv_path}"
        )

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if data.size == 0:
        raise RuntimeError(
            f"No tracking rows found in:\n{csv_path}"
        )

    if data.ndim == 0:
        data = np.array([data])

    required = {
        "track_id",
        "bin_id",
        "x",
        "y",
        "score",
    }

    columns = set(data.dtype.names or [])
    missing = required - columns

    if missing:
        raise RuntimeError(
            f"Missing columns {sorted(missing)} in:\n"
            f"{csv_path}\n"
            f"Found columns: {sorted(columns)}"
        )

    tracks = {}

    for row in data:
        track_id = int(row["track_id"])
        bin_id = int(row["bin_id"])

        tracks.setdefault(track_id, {})

        tracks[track_id][bin_id] = (
            float(row["x"]),
            float(row["y"]),
            float(row["score"]),
        )

    return tracks


def xy_to_pixel(x, y):
    """Convert 128x128 SNN/event coordinates to video coordinates."""

    plot_size = SIZE - 2 * PAD

    px = int(
        PAD + (float(x) / 127.0) * plot_size
    )

    py = int(
        PAD + (float(y) / 127.0) * plot_size
    )

    return px, py


def make_frame(name, tracks, bin_id, total_bins):
    """Draw trajectories accumulated up to the current bin."""

    frame = np.zeros(
        (SIZE, SIZE, 3),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"{name} | SNN-DERIVED OBJECT TRAJECTORY",
        (20, 35),
        FONT,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Plot boundary
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (PAD, PAD),
        (SIZE - PAD, SIZE - PAD),
        (100, 100, 100),
        1,
    )

    # --------------------------------------------------------
    # Coordinate grid
    # --------------------------------------------------------

    for value in [32, 64, 96]:

        px, _ = xy_to_pixel(value, 0)
        _, py = xy_to_pixel(0, value)

        cv2.line(
            frame,
            (px, PAD),
            (px, SIZE - PAD),
            (45, 45, 45),
            1,
        )

        cv2.line(
            frame,
            (PAD, py),
            (SIZE - PAD, py),
            (45, 45, 45),
            1,
        )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "X (128x128 event/SNN space)",
        (PAD + 5, SIZE - 18),
        FONT,
        0.48,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Y",
        (20, PAD + 10),
        FONT,
        0.55,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    active = 0

    # --------------------------------------------------------
    # Draw each track
    # --------------------------------------------------------

    for idx, track_id in enumerate(sorted(tracks)):

        track = tracks[track_id]

        visible_bins = sorted(
            b for b in track
            if b <= bin_id
        )

        if not visible_bins:
            continue

        active += 1

        color = COLORS[
            idx % len(COLORS)
        ]

        points = [
            xy_to_pixel(
                track[b][0],
                track[b][1],
            )
            for b in visible_bins
        ]

        # ----------------------------------------------------
        # Accumulated trajectory
        # ----------------------------------------------------

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                color,
                4,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Starting point
        # ----------------------------------------------------

        sx, sy = points[0]

        cv2.circle(
            frame,
            (sx, sy),
            5,
            color,
            -1,
        )

        # ----------------------------------------------------
        # Current position
        # ----------------------------------------------------

        px, py = points[-1]

        cv2.circle(
            frame,
            (px, py),
            10,
            color,
            -1,
        )

        cv2.circle(
            frame,
            (px, py),
            13,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Track ID
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"T{track_id}",
            (px + 15, py - 10),
            FONT,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Information bar
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Temporal Bin: {bin_id + 1}/{total_bins}",
        (20, SIZE - 45),
        FONT,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Active Tracks: {active}",
        (300, SIZE - 45),
        FONT,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def create_video(name, csv_path):

    print()
    print("=" * 70)
    print(f"CREATING SNN + TRAJECTORY VIDEO: {name}")
    print("=" * 70)

    tracks = load_tracks(csv_path)

    print(
        f"Track IDs: {sorted(tracks)}"
    )

    max_bin = max(
        max(track.keys())
        for track in tracks.values()
    )

    total_bins = max_bin + 1

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_snn_trajectory.mp4",
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (SIZE, SIZE),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video:\n{output_path}"
        )

    for bin_id in range(total_bins):

        frame = make_frame(
            name,
            tracks,
            bin_id,
            total_bins,
        )

        writer.write(frame)

    writer.release()

    size_mb = (
        os.path.getsize(output_path)
        / (1024 * 1024)
    )

    print("VIDEO SAVED:")
    print(output_path)
    print(
        f"Size: {size_mb:.3f} MB"
    )


def main():

    for name, csv_path in TRACK_FILES.items():
        create_video(
            name,
            csv_path,
        )

    print()
    print("=" * 70)
    print("SNN + TRAJECTORY VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()