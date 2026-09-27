import numpy as np
import pandas as pd
import cv2
import os


# ============================================================
# PATHS
# ============================================================

# Current directory:
# DVS-Voltmeter-main/snn_project/scripts

VOXEL_FILE = (
    "../ship_047_event_dataset/"
    "3_voxel_grid/"
    "ship_047_voxel_grid.npy"
)

LABEL_FILE = (
    "../data/"
    "ship_047_event_gt_labels.csv"
)

OUTPUT_DIR = (
    "../results/"
    "event_gt_alignment_047"
)


# Create output directory
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD VOXEL GRID
# ============================================================

print("Loading voxel grid...")

voxel = np.load(
    VOXEL_FILE
)

print(
    "Voxel shape:",
    voxel.shape
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth labels...")

labels = pd.read_csv(
    LABEL_FILE
)

print(
    "Number of labels:",
    len(labels)
)


# ============================================================
# SELECT BINS TO VISUALIZE
# ============================================================

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


# ============================================================
# PROCESS EACH BIN
# ============================================================

for bin_id in selected_bins:

    print(
        f"\nProcessing bin {bin_id}..."
    )

    # --------------------------------------------------------
    # GET EVENT DATA
    # --------------------------------------------------------

    event_bin = voxel[
        bin_id
    ]

    # Shape:
    # (2, 128, 128)

    off_events = event_bin[0]
    on_events = event_bin[1]


    # --------------------------------------------------------
    # CREATE EVENT IMAGE
    # --------------------------------------------------------

    event_image = np.zeros(
        (
            128,
            128,
            3
        ),
        dtype=np.uint8
    )


    # ON events
    event_image[
        on_events > 0
    ] = [
        255,
        255,
        255
    ]


    # OFF events
    event_image[
        off_events > 0
    ] = [
        150,
        150,
        150
    ]


    # --------------------------------------------------------
    # GET GROUND-TRUTH LABELS FOR THIS BIN
    # --------------------------------------------------------

    bin_labels = labels[
        labels["bin_id"] == bin_id
    ]


    # --------------------------------------------------------
    # DRAW BOUNDING BOX
    # --------------------------------------------------------

    for _, row in bin_labels.iterrows():

        target_id = int(
            row["target_id"]
        )

        visible = int(
            row["visible"]
        )


        # Skip invisible targets
        if visible == 0:
            continue


        # Bounding box coordinates
        x = int(
            row["x_128"]
        )

        y = int(
            row["y_128"]
        )

        w = int(
            row["w_128"]
        )

        h = int(
            row["h_128"]
        )


        # Make very small boxes visible
        w = max(
            w,
            2
        )

        h = max(
            h,
            2
        )


        # ----------------------------------------------------
        # DRAW RECTANGLE
        # ----------------------------------------------------

        cv2.rectangle(
            event_image,
            (
                x,
                y
            ),
            (
                x + w,
                y + h
            ),
            (
                255,
                255,
                255
            ),
            1
        )


        # ----------------------------------------------------
        # DRAW TARGET ID
        # ----------------------------------------------------

        cv2.putText(
            event_image,
            f"T{target_id}",
            (
                x,
                max(
                    10,
                    y - 3
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (
                255,
                255,
                255
            ),
            1
        )


    # --------------------------------------------------------
    # GET FRAME NUMBER
    # --------------------------------------------------------

    frame_number = int(
        bin_labels[
            "frame_number"
        ].iloc[0]
    )


    # --------------------------------------------------------
    # ADD BIN INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        event_image,
        f"Bin {bin_id} | Frame {frame_number}",
        (
            5,
            15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (
            255,
            255,
            255
        ),
        1
    )


    # --------------------------------------------------------
    # ENLARGE IMAGE
    # --------------------------------------------------------

    visualization = cv2.resize(
        event_image,
        (
            640,
            640
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"bin_{bin_id:02d}_alignment.png"
    )


    cv2.imwrite(
        output_path,
        visualization
    )


    print(
        "Saved:",
        output_path
    )


# ============================================================
# COMPLETION
# ============================================================

print(
    "\nAlignment visualization completed!"
)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)