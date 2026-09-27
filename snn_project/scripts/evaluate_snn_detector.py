import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn


# ============================================================
# PATHS
# ============================================================

VOXEL_045 = "../../ship_045_event_dataset/3_voxel_grid/ship_045_voxel_grid.npy"
VOXEL_047 = "../ship_047_event_dataset/3_voxel_grid/ship_047_voxel_grid.npy"

GT_045 = "../data/ship_045_event_gt_labels.csv"
GT_047 = "../data/ship_047_event_gt_labels.csv"

MODEL_PATH = "../models/snn_ship_detector.pth"

OUTPUT_DIR = "../results/snn_detector/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL
# ============================================================

class SNNDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            2, 16,
            kernel_size=3,
            padding=1
        )

        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=snn.surrogate.fast_sigmoid()
        )

        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(
            16, 32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(
            beta=0.9,
            spike_grad=snn.surrogate.fast_sigmoid()
        )

        self.pool2 = nn.AvgPool2d(2)

        self.fc = nn.Linear(
            32 * 32 * 32,
            4
        )

    def forward(self, x):

        cur1 = self.conv1(x)

        mem1 = self.lif1.init_leaky()

        spk1, mem1 = self.lif1(
            cur1,
            mem1
        )

        x = self.pool1(spk1)

        cur2 = self.conv2(x)

        mem2 = self.lif2.init_leaky()

        spk2, mem2 = self.lif2(
            cur2,
            mem2
        )

        x = self.pool2(spk2)

        x = x.flatten(1)

        output = self.fc(x)

        return output


# ============================================================
# LOAD MODEL
# ============================================================

model = SNNDetector().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

voxel_045 = np.load(
    VOXEL_045
).astype(np.float32)

voxel_047 = np.load(
    VOXEL_047
).astype(np.float32)

gt_045 = pd.read_csv(GT_045)
gt_047 = pd.read_csv(GT_047)

print("\n045 voxel:", voxel_045.shape)
print("047 voxel:", voxel_047.shape)

print("045 GT:", gt_045.shape)
print("047 GT:", gt_047.shape)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_sequence(voxel, gt):

    X = voxel

    targets = []

    for bin_id in range(50):

        rows = gt[
            (gt["bin_id"] == bin_id) &
            (gt["target_id"] == 1)
        ]

        if len(rows) == 0:

            print(
                f"Warning: no target 1 for bin {bin_id}"
            )

            targets.append(
                [0, 0, 0, 0]
            )

            continue

        row = rows.iloc[0]

        # Coordinates are ALREADY in 128x128 space
        x = row["x_128"]
        y = row["y_128"]
        w = row["w_128"]
        h = row["h_128"]

        # Normalize to 0-1
        target = [
            x / 128.0,
            y / 128.0,
            w / 128.0,
            h / 128.0
        ]

        targets.append(target)

    return X, np.array(
        targets,
        dtype=np.float32
    )


X045, Y045 = prepare_sequence(
    voxel_045,
    gt_045
)

X047, Y047 = prepare_sequence(
    voxel_047,
    gt_047
)


# ============================================================
# COMBINE DATA
# ============================================================

X = np.concatenate(
    [X045, X047],
    axis=0
)

Y = np.concatenate(
    [Y045, Y047],
    axis=0
)

print("\nCombined X:", X.shape)
print("Combined Y:", Y.shape)


# ============================================================
# RECREATE SAME TRAIN/TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.arange(
    len(X)
)

np.random.shuffle(indices)

split = int(
    0.8 * len(indices)
)

train_indices = indices[:split]

test_indices = indices[split:]

X_test = X[test_indices]

Y_test = Y[test_indices]

print("\nTraining samples:", len(train_indices))
print("Testing samples:", len(test_indices))


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    # box format:
    # x, y, w, h

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corners

    xa1 = x1
    ya1 = y1

    xa2 = x1 + w1
    ya2 = y1 + h1

    xb1 = x2
    yb1 = y2

    xb2 = x2 + w2
    yb2 = y2 + h2

    # Intersection

    inter_x1 = max(
        xa1,
        xb1
    )

    inter_y1 = max(
        ya1,
        yb1
    )

    inter_x2 = min(
        xa2,
        xb2
    )

    inter_y2 = min(
        ya2,
        yb2
    )

    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_w *
        inter_h
    )

    # Areas

    area1 = max(
        0,
        w1
    ) * max(
        0,
        h1
    )

    area2 = max(
        0,
        w2
    ) * max(
        0,
        h2
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\nRunning evaluation...\n")

with torch.no_grad():

    for i in range(
        len(X_test)
    ):

        sample = torch.tensor(
            X_test[i],
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        prediction = model(
            sample
        )

        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )

        # Normalize prediction to 0-1
        prediction = np.clip(
            prediction,
            0,
            1
        )

        # Convert to 128x128 pixels

        pred_box = (
            prediction *
            128.0
        )

        gt_box = (
            Y_test[i] *
            128.0
        )

        # Ensure valid width/height

        pred_box[2] = max(
            1,
            pred_box[2]
        )

        pred_box[3] = max(
            1,
            pred_box[3]
        )

        gt_box[2] = max(
            1,
            gt_box[2]
        )

        gt_box[3] = max(
            1,
            gt_box[3]
        )

        # Calculate IoU

        iou = calculate_iou(
            pred_box,
            gt_box
        )

        results.append({

            "test_index":
                i,

            "original_index":
                int(test_indices[i]),

            "pred_x":
                pred_box[0],

            "pred_y":
                pred_box[1],

            "pred_w":
                pred_box[2],

            "pred_h":
                pred_box[3],

            "gt_x":
                gt_box[0],

            "gt_y":
                gt_box[1],

            "gt_w":
                gt_box[2],

            "gt_h":
                gt_box[3],

            "iou":
                iou
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "evaluation_metrics.csv"
)

results_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# CALCULATE METRICS
# ============================================================

ious = results_df[
    "iou"
].values

mean_iou = np.mean(
    ious
)

median_iou = np.median(
    ious
)

detection_25 = np.mean(
    ious >= 0.25
)

detection_50 = np.mean(
    ious >= 0.50
)

detection_75 = np.mean(
    ious >= 0.75
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("========================================")
print("SNN SHIP DETECTOR EVALUATION")
print("========================================")

print(
    f"Test samples   : {len(ious)}"
)

print(
    f"Mean IoU       : {mean_iou:.4f}"
)

print(
    f"Median IoU     : {median_iou:.4f}"
)

print(
    f"IoU >= 0.25    : "
    f"{detection_25 * 100:.2f}%"
)

print(
    f"IoU >= 0.50    : "
    f"{detection_50 * 100:.2f}%"
)

print(
    f"IoU >= 0.75    : "
    f"{detection_75 * 100:.2f}%"
)

print("\nMetrics saved to:")
print(metrics_file)


# ============================================================
# INDIVIDUAL IOU
# ============================================================

print("\nIndividual IoU values:")
print("----------------------------------------")

for _, row in results_df.iterrows():

    print(
        f"Test {int(row['test_index']):02d} "
        f"| Original {int(row['original_index']):03d} "
        f"| IoU = {row['iou']:.4f}"
    )


# ============================================================
# SUMMARY FILE
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "evaluation_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "SNN Ship Detector Evaluation\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Test samples: {len(ious)}\n"
    )

    f.write(
        f"Mean IoU: {mean_iou:.4f}\n"
    )

    f.write(
        f"Median IoU: {median_iou:.4f}\n"
    )

    f.write(
        f"IoU >= 0.25: "
        f"{detection_25 * 100:.2f}%\n"
    )

    f.write(
        f"IoU >= 0.50: "
        f"{detection_50 * 100:.2f}%\n"
    )

    f.write(
        f"IoU >= 0.75: "
        f"{detection_75 * 100:.2f}%\n"
    )


print("\nSummary saved to:")
print(summary_file)

print("\nEvaluation complete.")