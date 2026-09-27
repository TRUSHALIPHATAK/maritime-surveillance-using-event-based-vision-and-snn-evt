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

PRED_045 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_045_predicted_heatmaps_v3.npy"
)

PRED_047 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_047_predicted_heatmaps_v3.npy"
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

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "snn_detection_diagnostic"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.72
MIN_DISTANCE = 6
MAX_PEAKS = 10

SELECTED_BINS = [
    0,
    5,
    10,
    20,
    30,
    40,
    49
]


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

            if score >= neighborhood.max():

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

            distance = (
                (x - px) ** 2
                +
                (y - py) ** 2
            ) ** 0.5

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

def load_gt_045():

    df = pd.read_csv(
        GT_045
    )

    gt = {}

    for bin_id in sorted(
        df["bin_id"].unique()
    ):

        rows = df[
            df["bin_id"] == bin_id
        ]

        centers = []

        for _, row in rows.iterrows():

            if int(row["visible"]) != 1:
                continue

            centers.append(
                (
                    float(row["center_x"]),
                    float(row["center_y"])
                )
            )

        gt[int(bin_id)] = centers

    return gt


def load_gt_047():

    df = pd.read_csv(
        GT_047
    )

    gt = {}

    for bin_id in sorted(
        df["bin_id"].unique()
    ):

        rows = df[
            df["bin_id"] == bin_id
        ]

        centers = []

        for _, row in rows.iterrows():

            x = float(row["x_128"])
            y = float(row["y_128"])

            w = float(row["w_128"])
            h = float(row["h_128"])

            centers.append(
                (
                    x + w / 2,
                    y + h / 2
                )
            )

        gt[int(bin_id)] = centers

    return gt


# ============================================================
# GET IMAGE
# ============================================================

def get_images(image_dir):

    files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    files.sort()

    return [
        os.path.join(
            image_dir,
            f
        )
        for f in files
    ]


# ============================================================
# CREATE BIN → FRAME MAPPING
# ============================================================

def bin_to_frame(
    bin_id,
    num_bins,
    num_frames
):

    return int(
        round(
            bin_id
            *
            (num_frames - 1)
            /
            (num_bins - 1)
        )
    )


# ============================================================
# DRAW DIAGNOSTIC
# ============================================================

def create_diagnostic(
    sequence_name,
    prediction_file,
    image_dir,
    gt
):

    print()
    print("=" * 70)
    print(
        f"DIAGNOSTIC: {sequence_name}"
    )
    print("=" * 70)


    predictions = np.load(
        prediction_file
    )

    print(
        "Prediction shape:",
        predictions.shape
    )


    images = get_images(
        image_dir
    )

    print(
        "Images:",
        len(images)
    )


    sequence_output = os.path.join(
        OUTPUT_DIR,
        sequence_name
    )

    os.makedirs(
        sequence_output,
        exist_ok=True
    )


    for bin_id in SELECTED_BINS:

        if bin_id >= len(predictions):
            continue


        # ----------------------------------------------------
        # Prediction heatmap
        # ----------------------------------------------------

        heatmap = predictions[
            bin_id
        ]


        peaks = detect_peaks(
            heatmap
        )


        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        frame_index = bin_to_frame(
            bin_id,
            len(predictions),
            len(images)
        )

        frame = cv2.imread(
            images[frame_index]
        )


        if frame is None:
            continue


        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # Draw GT
        # BLUE = ground truth
        # ----------------------------------------------------

        for gx, gy in gt.get(
            bin_id,
            []
        ):

            px = int(
                round(
                    gx
                    *
                    (width - 1)
                    /
                    127
                )
            )

            py = int(
                round(
                    gy
                    *
                    (height - 1)
                    /
                    127
                )
            )


            cv2.circle(
                frame,
                (px, py),
                12,
                (255, 0, 0),
                3
            )


            cv2.putText(
                frame,
                "GT",
                (
                    px + 15,
                    py - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )


        # ----------------------------------------------------
        # Draw ALL SNN peaks
        # GREEN = SNN
        # ----------------------------------------------------

        for i, (
            score,
            x,
            y
        ) in enumerate(peaks):

            px = int(
                round(
                    x
                    *
                    (width - 1)
                    /
                    127
                )
            )

            py = int(
                round(
                    y
                    *
                    (height - 1)
                    /
                    127
                )
            )


            cv2.circle(
                frame,
                (px, py),
                9,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"SNN{i+1} {score:.2f}",
                (
                    px + 12,
                    py + 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (600, 90),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"SNN Detection Diagnostic - {sequence_name}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"Bin: {bin_id}   "
                f"Frame: {frame_index + 1}/"
                f"{len(images)}   "
                f"GT: {len(gt.get(bin_id, []))}   "
                f"SNN: {len(peaks)}"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


        output_file = os.path.join(
            sequence_output,
            f"bin_{bin_id:02d}.png"
        )


        cv2.imwrite(
            output_file,
            frame
        )


        print()
        print(
            f"Bin {bin_id:02d}"
        )

        print(
            "GT:",
            [
                (
                    round(x, 2),
                    round(y, 2)
                )
                for x, y in gt.get(
                    bin_id,
                    []
                )
            ]
        )

        print(
            "SNN:",
            [
                (
                    round(x, 2),
                    int(px),
                    int(py),
                    round(score, 3)
                )
                for score, px, py in peaks
                for x, y in [(px, py)]
            ]
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    gt_045 = load_gt_045()

    gt_047 = load_gt_047()


    create_diagnostic(
        "ship_045",
        PRED_045,
        IMAGE_DIR_045,
        gt_045
    )


    create_diagnostic(
        "ship_047",
        PRED_047,
        IMAGE_DIR_047,
        gt_047
    )


    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print()
    print(
        OUTPUT_DIR
    )