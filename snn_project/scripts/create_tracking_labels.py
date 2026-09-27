import pandas as pd
import os


# ------------------------------------------------
# PATHS
# ------------------------------------------------

INPUT_FILE = (
    "snn_project/data/"
    "ship_045_event_gt_labels.csv"
)


OUTPUT_DIR = (
    "snn_project/data"
)


OUTPUT_FILE = os.path.join(

    OUTPUT_DIR,

    "ship_045_tracking_ground_truth.csv"

)


# ------------------------------------------------
# LOAD LABELS
# ------------------------------------------------

print("Loading event ground truth labels...")


labels = pd.read_csv(
    INPUT_FILE
)


print(
    "Total labels:",
    len(labels)
)


# ------------------------------------------------
# CREATE TRACKING LABELS
# ------------------------------------------------

tracking_data = []


for _, row in labels.iterrows():


    bin_id = int(
        row["bin_id"]
    )


    frame_number = int(
        row["frame_number"]
    )


    target_id = int(
        row["target_id"]
    )


    visible = int(
        row["visible"]
    )


    x = float(
        row["x_128"]
    )


    y = float(
        row["y_128"]
    )


    w = float(
        row["w_128"]
    )


    h = float(
        row["h_128"]
    )


    # --------------------------------------------
    # CALCULATE CENTER
    # --------------------------------------------

    if visible == 1:


        center_x = (
            x + w / 2
        )


        center_y = (
            y + h / 2
        )


    else:


        center_x = None

        center_y = None


    # --------------------------------------------
    # SAVE
    # --------------------------------------------

    tracking_data.append({

        "bin_id":
        bin_id,

        "frame_number":
        frame_number,

        "target_id":
        target_id,

        "visible":
        visible,

        "x":
        x,

        "y":
        y,

        "width":
        w,

        "height":
        h,

        "center_x":
        center_x,

        "center_y":
        center_y

    })


# ------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------

tracking_df = pd.DataFrame(
    tracking_data
)


# ------------------------------------------------
# SAVE
# ------------------------------------------------

tracking_df.to_csv(

    OUTPUT_FILE,

    index=False

)


print(
    "\nTracking ground truth created successfully!"
)


print(
    "Saved to:"
)


print(
    OUTPUT_FILE
)


# ------------------------------------------------
# STATISTICS
# ------------------------------------------------

print(
    "\n----------------------------"
)


print(
    "TRACKING DATASET STATISTICS"
)


print(
    "----------------------------"
)


print(
    "Total entries:",
    len(tracking_df)
)


print(
    "Number of time bins:",
    tracking_df["bin_id"].nunique()
)


print(
    "Number of targets:",
    tracking_df["target_id"].nunique()
)


print(
    "\nVisibility by target:"
)


for target_id in sorted(
    tracking_df["target_id"].unique()
):


    target_data = tracking_df[

        tracking_df[
            "target_id"
        ] == target_id

    ]


    visible_count = target_data[
        "visible"
    ].sum()


    print(

        f"Target {target_id}: "
        f"{visible_count} / "
        f"{len(target_data)} visible"

    )


print(
    "----------------------------"
)


# ------------------------------------------------
# SHOW SAMPLE
# ------------------------------------------------

print(
    "\nFirst 10 entries:"
)


print(

    tracking_df.head(10)

)