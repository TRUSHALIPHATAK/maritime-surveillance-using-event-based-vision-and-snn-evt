import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VOXEL_045 = (
    BASE_DIR
    / "../../ship_045_event_dataset/3_voxel_grid/ship_045_voxel_grid.npy"
).resolve()

GT_045 = (
    BASE_DIR
    / "../data/ship_045_event_gt_labels.csv"
).resolve()

VOXEL_047 = (
    BASE_DIR
    / "../ship_047_event_dataset/3_voxel_grid/ship_047_voxel_grid.npy"
).resolve()

GT_047 = (
    BASE_DIR
    / "../data/ship_047_event_gt_labels.csv"
).resolve()


DATA_DIR = (
    BASE_DIR / "../data"
).resolve()

RESULTS_DIR = (
    BASE_DIR / "../results"
).resolve()


OUTPUT_045 = (
    DATA_DIR / "ship_045_center_heatmaps.npy"
)

OUTPUT_047 = (
    DATA_DIR / "ship_047_center_heatmaps.npy"
)

VIS_045 = (
    RESULTS_DIR / "center_heatmaps_045"
)

VIS_047 = (
    RESULTS_DIR / "center_heatmaps_047"
)


# ============================================================
# SETTINGS
# ============================================================

HEIGHT = 128
WIDTH = 128

SIGMA = 1.5

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
# CREATE GAUSSIAN
# ============================================================

def create_gaussian_heatmap(cx, cy):

    y_grid, x_grid = np.mgrid[
        0:HEIGHT,
        0:WIDTH
    ]

    heatmap = np.exp(
        -(
            (x_grid - cx) ** 2
            +
            (y_grid - cy) ** 2
        )
        /
        (
            2 * SIGMA ** 2
        )
    )

    return heatmap.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE SEQUENCE
# ============================================================

def process_sequence(
    voxel_file,
    gt_file,
    output_file,
    visualization_dir,
    sequence_name
):

    print()
    print("=" * 60)
    print("Processing:", sequence_name)
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not voxel_file.exists():
        raise FileNotFoundError(
            f"Voxel file not found:\n{voxel_file}"
        )

    if not gt_file.exists():
        raise FileNotFoundError(
            f"GT file not found:\n{gt_file}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    voxel = np.load(
        voxel_file
    )

    gt = pd.read_csv(
        gt_file
    )

    print(
        "Voxel shape:",
        voxel.shape
    )

    print(
        "GT shape:",
        gt.shape
    )

    # --------------------------------------------------------
    # Create visualization directory
    # --------------------------------------------------------

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Store all 50 heatmaps
    # --------------------------------------------------------

    heatmaps = []

    # ========================================================
    # PROCESS EVERY TEMPORAL BIN
    # ========================================================

    for bin_id in range(
        voxel.shape[0]
    ):

        # Start with empty heatmap
        heatmap = np.zeros(
            (HEIGHT, WIDTH),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Get ALL visible targets in this bin
        # ----------------------------------------------------

        rows = gt[
            (gt["bin_id"] == bin_id)
            &
            (gt["visible"] == 1)
        ]

        # ----------------------------------------------------
        # Add one Gaussian for EACH visible ship
        # ----------------------------------------------------

        for _, row in rows.iterrows():

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

            # Bounding-box center
            cx = x + (
                w / 2.0
            )

            cy = y + (
                h / 2.0
            )

            # Create ship center
            ship_heatmap = (
                create_gaussian_heatmap(
                    cx,
                    cy
                )
            )

            # Keep all ships
            heatmap = np.maximum(
                heatmap,
                ship_heatmap
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Append INSIDE the bin loop
        # ----------------------------------------------------

        heatmaps.append(
            heatmap
        )

    # ========================================================
    # Convert list → array
    # ========================================================

    heatmaps = np.stack(
        heatmaps,
        axis=0
    )

    print(
        "Heatmap shape:",
        heatmaps.shape
    )

    # --------------------------------------------------------
    # Verify expected shape
    # --------------------------------------------------------

    expected_shape = (
        voxel.shape[0],
        HEIGHT,
        WIDTH
    )

    if heatmaps.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected heatmap shape: "
            f"{heatmaps.shape}. "
            f"Expected: {expected_shape}"
        )

    # ========================================================
    # SAVE NUMPY FILE
    # ========================================================

    np.save(
        output_file,
        heatmaps
    )

    print(
        "Saved:",
        output_file
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    print()
    print("Creating visualizations...")

    for bin_id in SELECTED_BINS:

        if bin_id >= voxel.shape[0]:
            continue

        # ----------------------------------------------------
        # Event image
        # ----------------------------------------------------

        event_image = (
            voxel[bin_id, 0]
            +
            voxel[bin_id, 1]
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.imshow(
            event_image,
            cmap="gray",
            origin="upper"
        )

        ax.imshow(
            heatmaps[bin_id],
            cmap="hot",
            alpha=0.65,
            origin="upper"
        )

        ax.set_title(
            f"{sequence_name} | "
            f"Bin {bin_id} | "
            f"Multi-Ship Center Heatmap"
        )

        ax.set_xlabel(
            "X"
        )

        ax.set_ylabel(
            "Y"
        )

        ax.set_xlim(
            0,
            128
        )

        ax.set_ylim(
            128,
            0
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Windows-safe output path
        # ----------------------------------------------------

        output_image = (
            visualization_dir
            /
            f"{sequence_name}_bin_{bin_id:02d}.png"
        )

        output_image = (
            output_image.resolve()
        )

        plt.savefig(
            str(output_image),
            dpi=150,
            format="png"
        )

        plt.close(
            fig
        )

        print(
            "Saved:",
            output_image
        )


# ============================================================
# RUN 045
# ============================================================

process_sequence(
    VOXEL_045,
    GT_045,
    OUTPUT_045,
    VIS_045,
    "ship_045"
)


# ============================================================
# RUN 047
# ============================================================

process_sequence(
    VOXEL_047,
    GT_047,
    OUTPUT_047,
    VIS_047,
    "ship_047"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

check_045 = np.load(
    OUTPUT_045
)

check_047 = np.load(
    OUTPUT_047
)

print(
    "045 heatmaps:",
    check_045.shape
)

print(
    "047 heatmaps:",
    check_047.shape
)

print()
print(
    "Expected:"
)

print(
    "045:",
    "(50, 128, 128)"
)

print(
    "047:",
    "(50, 128, 128)"
)

print()
print(
    "CENTER HEATMAP GENERATION COMPLETE"
)