import os
import sys
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from evt_project.models.evt_model import EVTModel

from evt_project.data.combined_evt_dataset import (
    CombinedEVTSequenceDataset
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "evt_project",
    "results",
    "best_evt_model_v2.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "evt_project",
    "results",
    "inference_v2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EVT V2 INFERENCE + SHIP LOCATION DIAGNOSTIC")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODEL
# ============================================================

model = EVTModel(
    input_channels=2,
    image_size=128,

    # V2
    patch_size=8,

    temporal_groups=10,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout=0.1
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

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

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown")
    )

    print(
        "Checkpoint loss:",
        checkpoint.get("loss", "unknown")
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print(
    "Checkpoint loaded:",
    CHECKPOINT
)


# ============================================================
# DATASET
# ============================================================

dataset = CombinedEVTSequenceDataset()

print(
    "Sequences:",
    len(dataset)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_peak(heatmap):
    """
    Return the strongest location in a heatmap.

    Returns:
        x, y, confidence
    """

    y, x = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape
    )

    confidence = heatmap[y, x]

    return (
        int(x),
        int(y),
        float(confidence)
    )


def euclidean_distance(
    x1,
    y1,
    x2,
    y2
):

    return float(
        np.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )
    )


# ============================================================
# DIAGNOSTIC STORAGE
# ============================================================

all_results = []


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    for sequence_idx in range(
        len(dataset)
    ):

        # ----------------------------------------------------
        # Load sequence
        # ----------------------------------------------------

        events, gt_heatmaps, sequence_name = (
            dataset[sequence_idx]
        )

        # events:
        # [10, 2, 128, 128]

        events_input = (
            events
            .unsqueeze(0)
            .to(DEVICE)
        )


        # ----------------------------------------------------
        # Model forward pass
        # ----------------------------------------------------

        raw_predictions = model(
            events_input
        )

        # raw_predictions:
        # [1, 10, 1, 128, 128]


        # ----------------------------------------------------
        # Convert logits -> probability
        # ----------------------------------------------------

        predictions = torch.sigmoid(
            raw_predictions
        )


        # ----------------------------------------------------
        # Remove batch/channel dimensions
        # ----------------------------------------------------

        predictions = (
            predictions
            .squeeze(0)
            .squeeze(1)
            .cpu()
            .numpy()
        )

        gt_heatmaps = (
            gt_heatmaps
            .numpy()
        )


        # ====================================================
        # SEQUENCE INFORMATION
        # ====================================================

        print()
        print("-" * 70)
        print(
            f"Sequence: {sequence_name}"
        )

        print(
            "Input shape :",
            events.shape
        )

        print(
            "Output shape:",
            predictions.shape
        )

        print(
            "Prediction range:",
            f"{predictions.min():.4f}",
            "-",
            f"{predictions.max():.4f}"
        )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        fig, axes = plt.subplots(
            10,
            3,
            figsize=(12, 30)
        )


        sequence_distances = []


        # ====================================================
        # GROUP LOOP
        # ====================================================

        for group_id in range(10):

            # ------------------------------------------------
            # Event image
            # ------------------------------------------------

            event_image = (
                events[group_id]
                .numpy()
                .sum(axis=0)
            )


            # ------------------------------------------------
            # Ground truth heatmap
            # ------------------------------------------------

            gt = gt_heatmaps[
                group_id
            ]


            # ------------------------------------------------
            # EVT prediction
            # ------------------------------------------------

            pred = predictions[
                group_id
            ]


            # ------------------------------------------------
            # Find GT center
            # ------------------------------------------------

            gt_x, gt_y, gt_conf = get_peak(
                gt
            )


            # ------------------------------------------------
            # Find EVT predicted center
            # ------------------------------------------------

            pred_x, pred_y, pred_conf = get_peak(
                pred
            )


            # ------------------------------------------------
            # Calculate localization error
            # ------------------------------------------------

            distance = euclidean_distance(
                pred_x,
                pred_y,
                gt_x,
                gt_y
            )

            sequence_distances.append(
                distance
            )


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            all_results.append(
                {
                    "sequence": sequence_name,
                    "group": group_id,
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "gt_peak": gt_conf,
                    "pred_peak": pred_conf,
                    "center_error_px": distance
                }
            )


            # ------------------------------------------------
            # Print diagnostic
            # ------------------------------------------------

            print()
            print(
                f"Group {group_id:02d}"
            )

            print(
                f"  GT center   : "
                f"({gt_x}, {gt_y})"
            )

            print(
                f"  EVT center  : "
                f"({pred_x}, {pred_y})"
            )

            print(
                f"  EVT peak    : "
                f"{pred_conf:.4f}"
            )

            print(
                f"  Error       : "
                f"{distance:.2f} px"
            )


            # =================================================
            # PLOT 1: EVENTS
            # =================================================

            axes[group_id, 0].imshow(
                event_image,
                cmap="gray"
            )

            axes[group_id, 0].set_title(
                f"Group {group_id} - Events"
            )


            # =================================================
            # PLOT 2: GT HEATMAP
            # =================================================

            axes[group_id, 1].imshow(
                gt,
                cmap="hot",
                vmin=0,
                vmax=1
            )

            axes[group_id, 1].scatter(
                gt_x,
                gt_y,
                marker="x",
                s=80
            )

            axes[group_id, 1].set_title(
                f"GT Heatmap\n"
                f"Center ({gt_x},{gt_y})"
            )


            # =================================================
            # PLOT 3: EVT PREDICTION
            # =================================================

            axes[group_id, 2].imshow(
                pred,
                cmap="hot",
                vmin=0,
                vmax=1
            )

            axes[group_id, 2].scatter(
                gt_x,
                gt_y,
                marker="x",
                s=80
            )

            axes[group_id, 2].scatter(
                pred_x,
                pred_y,
                marker="o",
                facecolors="none",
                s=80
            )

            axes[group_id, 2].set_title(
                f"EVT Prediction\n"
                f"Error: {distance:.2f}px"
            )


            # ------------------------------------------------
            # Remove axes
            # ------------------------------------------------

            for col in range(3):

                axes[
                    group_id,
                    col
                ].axis("off")


        # ====================================================
        # SEQUENCE SUMMARY
        # ====================================================

        mean_error = np.mean(
            sequence_distances
        )

        median_error = np.median(
            sequence_distances
        )

        within_2 = np.mean(
            np.array(sequence_distances) <= 2
        ) * 100

        within_4 = np.mean(
            np.array(sequence_distances) <= 4
        ) * 100

        within_6 = np.mean(
            np.array(sequence_distances) <= 6
        ) * 100


        print()
        print(
            "=" * 70
        )

        print(
            f"SEQUENCE {sequence_name} SUMMARY"
        )

        print(
            f"Mean center error   : "
            f"{mean_error:.2f} px"
        )

        print(
            f"Median center error : "
            f"{median_error:.2f} px"
        )

        print(
            f"Within 2 px         : "
            f"{within_2:.2f}%"
        )

        print(
            f"Within 4 px         : "
            f"{within_4:.2f}%"
        )

        print(
            f"Within 6 px         : "
            f"{within_6:.2f}%"
        )

        print(
            "=" * 70
        )


        # ====================================================
        # SAVE VISUALIZATION
        # ====================================================

        plt.tight_layout()

        figure_file = os.path.join(
            OUTPUT_DIR,
            f"ship_{sequence_name}_evt_v2_diagnostic.png"
        )

        plt.savefig(
            figure_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Saved visualization:",
            figure_file
        )


        # ====================================================
        # SAVE RAW PREDICTIONS
        # ====================================================

        prediction_file = os.path.join(
            OUTPUT_DIR,
            f"ship_{sequence_name}_evt_heatmaps_v2.npy"
        )

        np.save(
            prediction_file,
            predictions
        )

        print(
            "Saved heatmaps:",
            prediction_file
        )


# ============================================================
# SAVE DIAGNOSTIC CSV
# ============================================================

csv_file = os.path.join(
    OUTPUT_DIR,
    "evt_v2_location_diagnostic.csv"
)


with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sequence",
            "group",
            "gt_x",
            "gt_y",
            "pred_x",
            "pred_y",
            "gt_peak",
            "pred_peak",
            "center_error_px"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


print()
print(
    "Saved diagnostic CSV:",
    csv_file
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

all_errors = np.array(
    [
        row["center_error_px"]
        for row in all_results
    ]
)


print()
print("=" * 70)
print("OVERALL EVT V2 LOCATION DIAGNOSTIC")
print("=" * 70)

print(
    f"Total groups       : {len(all_errors)}"
)

print(
    f"Mean error         : "
    f"{np.mean(all_errors):.2f} px"
)

print(
    f"Median error       : "
    f"{np.median(all_errors):.2f} px"
)

print(
    f"Maximum error      : "
    f"{np.max(all_errors):.2f} px"
)

print(
    f"Within 2 px        : "
    f"{np.mean(all_errors <= 2) * 100:.2f}%"
)

print(
    f"Within 4 px        : "
    f"{np.mean(all_errors <= 4) * 100:.2f}%"
)

print(
    f"Within 6 px        : "
    f"{np.mean(all_errors <= 6) * 100:.2f}%"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("EVT V2 DIAGNOSTIC COMPLETE")
print("=" * 70)