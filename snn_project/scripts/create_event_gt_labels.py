import numpy as np
import pandas as pd
import os


# ------------------------------------------------
# PATHS
# ------------------------------------------------

TIME_BIN_FILE = (
    "../ship_047_event_dataset/"
    "4_time_bins/"
    "ship_047_time_bins.csv"
)


GT_FOLDER = (
    "../../../VISO/SOT/sot/ship/047/gt"
)


OUTPUT_FOLDER = (
    "../data"
)


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ------------------------------------------------
# ORIGINAL IMAGE SIZE
# ------------------------------------------------

ORIGINAL_WIDTH = 1345
ORIGINAL_HEIGHT = 451


# ------------------------------------------------
# EVENT / SNN SIZE
# ------------------------------------------------

EVENT_WIDTH = 128
EVENT_HEIGHT = 128


# ------------------------------------------------
# FRAME TIMING
# ------------------------------------------------

FRAME_INTERVAL_US = 33333


# ------------------------------------------------
# LOAD TIME BINS
# ------------------------------------------------

print("Loading event time bins...")

time_bins = pd.read_csv(
    TIME_BIN_FILE
)

print(
    "Number of event bins:",
    len(time_bins)
)


# ------------------------------------------------
# LOAD GROUND TRUTH
# ------------------------------------------------

def load_ground_truth(file_path):

    data = []

    with open(
        file_path,
        "r"
    ) as file:

        for line in file:

            values = line.strip().split(",")

            if len(values) != 4:
                continue

            x, y, w, h = map(
                float,
                values
            )

            data.append(
                [x, y, w, h]
            )

    return data


print("\nLoading ground truth...")


gt_file = os.path.join(
    GT_FOLDER,
    "1_1_300.txt"
)


ground_truth = load_ground_truth(
    gt_file
)


print(
    "Target 1:",
    len(ground_truth),
    "entries"
)


# ------------------------------------------------
# CREATE EVENT-ALIGNED LABELS
# ------------------------------------------------

print(
    "\nCreating event-aligned "
    "ground truth labels..."
)


labels = []


for _, row in time_bins.iterrows():

    bin_id = int(
        row["bin_id"]
    )

    start_time = float(
        row["start_time_us"]
    )

    end_time = float(
        row["end_time_us"]
    )


    # --------------------------------------------
    # FIND MIDDLE OF EVENT BIN
    # --------------------------------------------

    center_time = (
        start_time +
        end_time
    ) / 2


    # --------------------------------------------
    # MAP TIME TO ORIGINAL FRAME
    # --------------------------------------------

    frame_index = int(
        center_time //
        FRAME_INTERVAL_US
    )


    # Convert zero-based index to frame number

    frame_number = (
        frame_index + 1
    )


    # Ensure valid range

    frame_number = max(
        1,
        min(
            frame_number,
            300
        )
    )


    # --------------------------------------------
    # GET GROUND-TRUTH BOX
    # --------------------------------------------

    x, y, w, h = (
        ground_truth[
            frame_number - 1
        ]
    )


    # --------------------------------------------
    # CHECK VISIBILITY
    # --------------------------------------------

    visible = not (
        x == 0 and
        y == 0 and
        w == 0 and
        h == 0
    )


    # --------------------------------------------
    # SCALE BOUNDING BOX
    # --------------------------------------------

    if visible:

        x_128 = (
            x *
            EVENT_WIDTH /
            ORIGINAL_WIDTH
        )

        y_128 = (
            y *
            EVENT_HEIGHT /
            ORIGINAL_HEIGHT
        )

        w_128 = (
            w *
            EVENT_WIDTH /
            ORIGINAL_WIDTH
        )

        h_128 = (
            h *
            EVENT_HEIGHT /
            ORIGINAL_HEIGHT
        )

    else:

        x_128 = 0
        y_128 = 0
        w_128 = 0
        h_128 = 0


    # --------------------------------------------
    # SAVE LABEL
    # --------------------------------------------

    labels.append({

        "bin_id":
        bin_id,

        "frame_number":
        frame_number,

        "target_id":
        1,

        "visible":
        int(visible),

        # Original coordinates

        "x_original":
        x,

        "y_original":
        y,

        "w_original":
        w,

        "h_original":
        h,

        # Event-space coordinates

        "x_128":
        x_128,

        "y_128":
        y_128,

        "w_128":
        w_128,

        "h_128":
        h_128

    })


# ------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------

labels_df = pd.DataFrame(
    labels
)


# ------------------------------------------------
# SAVE
# ------------------------------------------------

output_file = os.path.join(

    OUTPUT_FOLDER,

    "ship_047_event_gt_labels.csv"

)


labels_df.to_csv(

    output_file,

    index=False

)


print(
    "\nGround truth alignment completed!"
)


print(
    "Output saved to:"
)


print(
    output_file
)


# ------------------------------------------------
# STATISTICS
# ------------------------------------------------

print(
    "\n----------------------------"
)

print(
    "GROUND TRUTH STATISTICS"
)

print(
    "----------------------------"
)

print(
    "Total label entries:",
    len(labels_df)
)

print(
    "Total event bins:",
    labels_df["bin_id"].nunique()
)

print(
    "Target 1 visible:",
    labels_df[
        labels_df["target_id"] == 1
    ]["visible"].sum()
)

print(
    "----------------------------"
)


# ------------------------------------------------
# SHOW SAMPLE
# ------------------------------------------------

print(
    "\nFirst 10 labels:"
)

print(
    labels_df.head(10)
)