import os
import sys
import torch
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from evt_project.models.evt_model import EVTModel
from evt_project.data.combined_evt_dataset import (
    CombinedEVTSequenceDataset
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "evt_project",
    "results",
    "best_evt_model.pth"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EVT SHIP LOCATION DIAGNOSTIC")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Checkpoint:",
    CHECKPOINT
)


# ============================================================
# MODEL
# ============================================================

model = EVTModel(
    input_channels=2,
    image_size=128,
    patch_size=32,
    temporal_groups=10,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout=0.1
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# ============================================================
# DATASET
# ============================================================

dataset = CombinedEVTSequenceDataset()

print(
    "Sequences:",
    len(dataset)
)


# ============================================================
# HELPER
# ============================================================

def get_gt_centers(sequence_name, group_id):

    """
    Extract unique GT ship centers belonging to
    this temporal group.

    Returns:
        [(x, y, target_id), ...]
    """

    if sequence_name == "045":

        csv_path = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "data",
            "ship_045_event_gt_labels.csv"
        )

    elif sequence_name == "047":

        csv_path = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "data",
            "ship_047_event_gt_labels.csv"
        )

    else:
        raise ValueError(
            f"Unknown sequence: {sequence_name}"
        )

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True
    )

    start_bin = group_id * 5
    end_bin = start_bin + 5

    centers = []

    for row in data:

        bin_id = int(row["bin_id"])

        if not (
            start_bin <= bin_id < end_bin
        ):
            continue

        visible = int(row["visible"])

        if visible == 0:
            continue

        x = int(round(row["x_128"]))
        y = int(round(row["y_128"]))
        target_id = int(row["target_id"])

        item = (
            x,
            y,
            target_id
        )

        if item not in centers:
            centers.append(item)

    return centers


# ============================================================
# DIAGNOSTIC
# ============================================================

overall_gt_values = []
overall_max_values = []

with torch.no_grad():

    for sequence_idx in range(len(dataset)):

        events, heatmaps, sequence_name = dataset[
            sequence_idx
        ]

        events_input = (
            events
            .unsqueeze(0)
            .to(DEVICE)
        )

        predictions = model(
            events_input
        )

        # [1,10,1,128,128]
        predictions = (
            predictions
            .squeeze(0)
            .squeeze(1)
            .cpu()
            .numpy()
        )

        print()
        print("=" * 70)
        print(
            f"SEQUENCE {sequence_name}"
        )
        print("=" * 70)

        sequence_gt_values = []
        sequence_max_values = []

        for group_id in range(10):

            pred = predictions[group_id]

            gt_centers = get_gt_centers(
                sequence_name,
                group_id
            )

            print()
            print(
                f"Group {group_id}"
            )

            print(
                "-" * 50
            )

            if len(gt_centers) == 0:

                print(
                    "No visible GT ships."
                )

                continue

            # ------------------------------------------------
            # Prediction statistics
            # ------------------------------------------------

            pred_min = pred.min()
            pred_max = pred.max()
            pred_mean = pred.mean()

            max_y, max_x = np.unravel_index(
                np.argmax(pred),
                pred.shape
            )

            print(
                f"Prediction min : {pred_min:.4f}"
            )

            print(
                f"Prediction mean: {pred_mean:.4f}"
            )

            print(
                f"Prediction max : {pred_max:.4f}"
            )

            print(
                f"Global maximum : "
                f"({max_x}, {max_y})"
            )

            # ------------------------------------------------
            # GT center values
            # ------------------------------------------------

            print()
            print("GT ship centers:")

            for x, y, target_id in gt_centers:

                value = float(
                    pred[y, x]
                )

                sequence_gt_values.append(
                    value
                )

                overall_gt_values.append(
                    value
                )

                print(
                    f"  Target {target_id}: "
                    f"({x:3d}, {y:3d}) "
                    f"-> prediction = {value:.4f}"
                )

            # ------------------------------------------------
            # Maximum prediction near each GT center
            # ------------------------------------------------

            print()
            print(
                "Local response around GT centers:"
            )

            radius = 3

            for x, y, target_id in gt_centers:

                x1 = max(
                    0,
                    x - radius
                )

                x2 = min(
                    128,
                    x + radius + 1
                )

                y1 = max(
                    0,
                    y - radius
                )

                y2 = min(
                    128,
                    y + radius + 1
                )

                local_region = pred[
                    y1:y2,
                    x1:x2
                ]

                local_max = float(
                    local_region.max()
                )

                local_y, local_x = np.unravel_index(
                    np.argmax(local_region),
                    local_region.shape
                )

                local_x += x1
                local_y += y1

                print(
                    f"  Target {target_id}: "
                    f"local max = {local_max:.4f} "
                    f"at ({local_x},{local_y})"
                )

                sequence_max_values.append(
                    local_max
                )

                overall_max_values.append(
                    local_max
                )

        # ----------------------------------------------------
        # Sequence summary
        # ----------------------------------------------------

        if sequence_gt_values:

            print()
            print(
                "-" * 70
            )

            print(
                f"{sequence_name} GT-center response:"
            )

            print(
                f"Mean GT-center value : "
                f"{np.mean(sequence_gt_values):.4f}"
            )

            print(
                f"Max GT-center value  : "
                f"{np.max(sequence_gt_values):.4f}"
            )

            print(
                f"Min GT-center value  : "
                f"{np.min(sequence_gt_values):.4f}"
            )

            print(
                f"Mean local max       : "
                f"{np.mean(sequence_max_values):.4f}"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FINAL EVT LOCATION DIAGNOSTIC")
print("=" * 70)

if overall_gt_values:

    print(
        f"Mean prediction at GT centers : "
        f"{np.mean(overall_gt_values):.4f}"
    )

    print(
        f"Maximum prediction at GT area  : "
        f"{np.max(overall_gt_values):.4f}"
    )

    print(
        f"Minimum prediction at GT area  : "
        f"{np.min(overall_gt_values):.4f}"
    )

    print(
        f"Mean local peak near GT        : "
        f"{np.mean(overall_max_values):.4f}"
    )

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
If prediction values near GT centers are clearly
higher than the general prediction background:

    -> EVT HAS LEARNED SHIP LOCATIONS.

If GT-center values are similar to the background:

    -> EVT has NOT learned the ship locations well yet.

If GT-center values are high but detect_evt.py
finds few peaks:

    -> The MODEL is working,
       but the PEAK DETECTION method needs adjustment.

If the global maximum repeatedly appears near the
GT ship centers:

    -> The EVT model has learned meaningful spatial
       localization.
""")

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)