import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# PATHS
# ============================================================

VOXEL_045 = (
    "../../ship_045_event_dataset/"
    "3_voxel_grid/"
    "ship_045_voxel_grid.npy"
)

GT_045 = (
    "../data/"
    "ship_045_event_gt_labels.csv"
)

VOXEL_047 = (
    "../ship_047_event_dataset/"
    "3_voxel_grid/"
    "ship_047_voxel_grid.npy"
)

GT_047 = (
    "../data/"
    "ship_047_event_gt_labels.csv"
)

MODEL_OUTPUT = (
    "../models/"
    "snn_ship_detector.pth"
)

RESULTS_DIR = (
    "../results/"
    "snn_detector"
)

os.makedirs(
    "../models",
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading 045 voxel grid...")

voxel_045 = np.load(
    VOXEL_045
).astype(
    np.float32
)

print(
    "045 shape:",
    voxel_045.shape
)


print("\nLoading 047 voxel grid...")

voxel_047 = np.load(
    VOXEL_047
).astype(
    np.float32
)

print(
    "047 shape:",
    voxel_047.shape
)


print("\nLoading ground truth...")

gt_045 = pd.read_csv(
    GT_045
)

gt_047 = pd.read_csv(
    GT_047
)

print(
    "045 GT entries:",
    len(gt_045)
)

print(
    "047 GT entries:",
    len(gt_047)
)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_sequence(
    voxel,
    gt
):

    inputs = []
    targets = []

    for bin_id in range(
        voxel.shape[0]
    ):

        # ----------------------------------------------------
        # Event data
        # ----------------------------------------------------

        event_bin = voxel[
            bin_id
        ]

        inputs.append(
            event_bin
        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        rows = gt[
            gt["bin_id"] == bin_id
        ]

        # We use target 1 for the first detector.
        target = rows[
            rows["target_id"] == 1
        ]

        if len(target) == 0:
            continue

        row = target.iloc[0]


        # x, y, width, height
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


        # Normalize coordinates
        # to 0-1 range.

        target_box = np.array(
            [
                x / 128.0,
                y / 128.0,
                w / 128.0,
                h / 128.0
            ],
            dtype=np.float32
        )

        targets.append(
            target_box
        )


    return (
        np.array(inputs),
        np.array(targets)
    )


print("\nPreparing 045...")

X_045, Y_045 = prepare_sequence(
    voxel_045,
    gt_045
)

print(
    "045 input:",
    X_045.shape
)

print(
    "045 targets:",
    Y_045.shape
)


print("\nPreparing 047...")

X_047, Y_047 = prepare_sequence(
    voxel_047,
    gt_047
)

print(
    "047 input:",
    X_047.shape
)

print(
    "047 targets:",
    Y_047.shape
)


# ============================================================
# COMBINE DATA
# ============================================================

X = np.concatenate(
    [
        X_045,
        X_047
    ],
    axis=0
)

Y = np.concatenate(
    [
        Y_045,
        Y_047
    ],
    axis=0
)


print("\nCombined dataset:")

print(
    "X:",
    X.shape
)

print(
    "Y:",
    Y.shape
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

num_samples = len(X)

indices = np.arange(
    num_samples
)

np.random.shuffle(
    indices
)

split = int(
    0.8 * num_samples
)

train_indices = indices[
    :split
]

test_indices = indices[
    split:
]


X_train = X[
    train_indices
]

Y_train = Y[
    train_indices
]

X_test = X[
    test_indices
]

Y_test = Y[
    test_indices
]


print("\nDataset split:")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32,
    device=DEVICE
)

Y_train = torch.tensor(
    Y_train,
    dtype=torch.float32,
    device=DEVICE
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32,
    device=DEVICE
)

Y_test = torch.tensor(
    Y_test,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# SNN DETECTOR
# ============================================================

class SNNDetector(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # SNN convolution layers
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            2,
            16,
            kernel_size=3,
            padding=1
        )

        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid()
        )


        self.pool1 = nn.AvgPool2d(
            2
        )


        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid()
        )


        self.pool2 = nn.AvgPool2d(
            2
        )


        # ----------------------------------------------------
        # Detection head
        #
        # 128x128
        # -> 64x64
        # -> 32x32
        #
        # 32 channels
        # ----------------------------------------------------

        self.fc = nn.Linear(
            32 * 32 * 32,
            4
        )


    def forward(
        self,
        x
    ):

        # x:
        # (batch, time, polarity, H, W)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        predictions = []


        # ----------------------------------------------------
        # Process temporal bins
        # ----------------------------------------------------

        for t in range(
            x.size(1)
        ):

            current = x[
                :,
                t
            ]


            # First SNN layer
            current = self.conv1(
                current
            )

            spike1, mem1 = self.lif1(
                current,
                mem1
            )

            spike1 = self.pool1(
                spike1
            )


            # Second SNN layer
            current = self.conv2(
                spike1
            )

            spike2, mem2 = self.lif2(
                current,
                mem2
            )

            spike2 = self.pool2(
                spike2
            )


            # Flatten
            features = spike2.flatten(
                start_dim=1
            )


            # Bounding box
            prediction = self.fc(
                features
            )


            predictions.append(
                prediction
            )


        return torch.stack(
            predictions,
            dim=1
        )


# ============================================================
# CREATE MODEL
# ============================================================

model = SNNDetector().to(
    DEVICE
)


print("\nModel created.")

print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 50

print("\nStarting training...")


for epoch in range(
    EPOCHS
):

    model.train()

    optimizer.zero_grad()


    # Add batch dimension.
    #
    # Current:
    # (samples, polarity, H, W)
    #
    # We process each sample as a
    # separate temporal sequence.

    epoch_loss = 0.0


    for i in range(
        len(X_train)
    ):

        # One event bin
        sample = X_train[
            i
        ]

        target = Y_train[
            i
        ]


        # Convert:
        #
        # (2,H,W)
        #
        # to
        #
        # (1,1,2,H,W)

        sample = sample.unsqueeze(
            0
        ).unsqueeze(
            0
        )


        prediction = model(
            sample
        )


        # Remove batch/time dimensions
        prediction = prediction[
            0,
            0
        ]


        loss = criterion(
            prediction,
            target
        )


        loss.backward()


        epoch_loss += (
            loss.item()
        )


    optimizer.step()


    epoch_loss /= len(
        X_train
    )


    if (
        (epoch + 1) % 5 == 0
        or epoch == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{EPOCHS} "
            f"Loss: "
            f"{epoch_loss:.6f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_OUTPUT
)

print(
    "\nModel saved to:"
)

print(
    MODEL_OUTPUT
)


# ============================================================
# TESTING
# ============================================================

print("\nTesting model...")

model.eval()

predictions = []

actuals = []


with torch.no_grad():

    for i in range(
        len(X_test)
    ):

        sample = X_test[
            i
        ]

        target = Y_test[
            i
        ]


        sample = sample.unsqueeze(
            0
        ).unsqueeze(
            0
        )


        prediction = model(
            sample
        )


        prediction = prediction[
            0,
            0
        ]


        predictions.append(
            prediction.cpu().numpy()
        )

        actuals.append(
            target.cpu().numpy()
        )


predictions = np.array(
    predictions
)

actuals = np.array(
    actuals
)


# ============================================================
# TEST LOSS
# ============================================================

test_loss = np.mean(
    np.abs(
        predictions
        -
        actuals
    )
)


print(
    "\nMean Absolute Error:",
    test_loss
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)


prediction_df = pd.DataFrame(
    predictions,
    columns=[
        "pred_x",
        "pred_y",
        "pred_w",
        "pred_h"
    ]
)


actual_df = pd.DataFrame(
    actuals,
    columns=[
        "gt_x",
        "gt_y",
        "gt_w",
        "gt_h"
    ]
)


results_df = pd.concat(
    [
        prediction_df,
        actual_df
    ],
    axis=1
)


results_df.to_csv(
    prediction_file,
    index=False
)


print(
    "\nPredictions saved to:"
)

print(
    prediction_file
)


print(
    "\nSNN DETECTOR TRAINING COMPLETED."
)