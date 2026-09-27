import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# PATHS
# ============================================================

VOXEL_045 = "../../ship_045_event_dataset/3_voxel_grid/ship_045_voxel_grid.npy"
GT_045 = "../data/ship_045_event_gt_labels.csv"

VOXEL_047 = "../ship_047_event_dataset/3_voxel_grid/ship_047_voxel_grid.npy"
GT_047 = "../data/ship_047_event_gt_labels.csv"

OUTPUT_045 = "../results/event_representation_check_045"
OUTPUT_047 = "../results/event_representation_check_047"

os.makedirs(OUTPUT_045, exist_ok=True)
os.makedirs(OUTPUT_047, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

voxel_045 = np.load(VOXEL_045)
voxel_047 = np.load(VOXEL_047)

gt_045 = pd.read_csv(GT_045)
gt_047 = pd.read_csv(GT_047)

print("045 voxel:", voxel_045.shape)
print("047 voxel:", voxel_047.shape)

print("045 GT:", gt_045.shape)
print("047 GT:", gt_047.shape)


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def visualize_sequence(
    voxel,
    gt,
    output_dir,
    sequence_name,
    selected_bins
):

    for bin_id in selected_bins:

        # ----------------------------------------------------
        # Event data
        # ----------------------------------------------------

        positive = voxel[
            bin_id,
            0
        ]

        negative = voxel[
            bin_id,
            1
        ]

        # Combined event activity
        event_image = positive + negative

        # Normalize only for visualization
        if event_image.max() > 0:

            display_image = (
                event_image /
                event_image.max()
            )

        else:

            display_image = event_image


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        rows = gt[
            (gt["bin_id"] == bin_id) &
            (gt["visible"] == 1)
        ]

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.imshow(
            display_image,
            cmap="gray",
            origin="upper"
        )

        # ----------------------------------------------------
        # Draw every visible target
        # ----------------------------------------------------

        for _, row in rows.iterrows():

            x = row["x_128"]
            y = row["y_128"]

            w = row["w_128"]
            h = row["h_128"]

            target_id = int(
                row["target_id"]
            )

            rectangle = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(
                rectangle
            )

            ax.text(
                x,
                max(0, y - 2),
                f"T{target_id}",
                color="red",
                fontsize=10,
                fontweight="bold"
            )


        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        ax.set_title(
            f"{sequence_name} | "
            f"Event Bin {bin_id}"
        )

        ax.set_xlabel(
            "X (128×128 event space)"
        )

        ax.set_ylabel(
            "Y (128×128 event space)"
        )

        ax.set_xlim(
            0,
            128
        )

        ax.set_ylim(
            128,
            0
        )

        ax.grid(
            False
        )

        plt.tight_layout()


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = os.path.join(
            output_dir,
            f"{sequence_name}_bin_{bin_id:02d}.png"
        )

        plt.savefig(
            output_file,
            dpi=150
        )

        plt.close()

        print(
            "Saved:",
            output_file
        )


# ============================================================
# SELECT BINS
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
# RUN 045
# ============================================================

print("\nGenerating 045 visualizations...")

visualize_sequence(
    voxel_045,
    gt_045,
    OUTPUT_045,
    "ship_045",
    selected_bins
)


# ============================================================
# RUN 047
# ============================================================

print("\nGenerating 047 visualizations...")

visualize_sequence(
    voxel_047,
    gt_047,
    OUTPUT_047,
    "ship_047",
    selected_bins
)


print("\n========================================")
print("EVENT REPRESENTATION CHECK COMPLETE")
print("========================================")

print("\n045 results:")
print(OUTPUT_045)

print("\n047 results:")
print(OUTPUT_047)