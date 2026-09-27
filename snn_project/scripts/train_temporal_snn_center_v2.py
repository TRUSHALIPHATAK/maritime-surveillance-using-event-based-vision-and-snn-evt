import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


# ============================================================
# TEMPORAL SNN CENTER DETECTOR V2
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V2")
print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# ROBUST PATH SETUP
# ============================================================

# Current file:
#
# DVS-Voltmeter-main/
# └── snn_project/
#     └── scripts/
#         └── train_temporal_snn_center_v2.py
#
# Therefore:
#
# SCRIPT_DIR      = snn_project/scripts
# SNN_PROJECT_DIR = snn_project
# PROJECT_ROOT    = DVS-Voltmeter-main

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    SNN_PROJECT_DIR
)


# ============================================================
# DATA PATHS
# ============================================================

VOXEL_045 = os.path.join(
    PROJECT_ROOT,
    "ship_045_event_dataset",
    "3_voxel_grid",
    "ship_045_voxel_grid.npy"
)

VOXEL_047 = os.path.join(
    SNN_PROJECT_DIR,
    "ship_047_event_dataset",
    "3_voxel_grid",
    "ship_047_voxel_grid.npy"
)

HEATMAP_045 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_045_center_heatmaps.npy"
)

HEATMAP_047 = os.path.join(
    SNN_PROJECT_DIR,
    "data",
    "ship_047_center_heatmaps.npy"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

MODEL_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v2"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "temporal_snn_center_v2.pth"
)

SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "training_summary.txt"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# VERIFY FILES BEFORE LOADING
# ============================================================

print()
print("=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

required_files = {
    "045 voxel": VOXEL_045,
    "047 voxel": VOXEL_047,
    "045 heatmap": HEATMAP_045,
    "047 heatmap": HEATMAP_047,
}

for name, path in required_files.items():

    print(
        f"{name:15s}: "
        f"{'FOUND' if os.path.exists(path) else 'NOT FOUND'}"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# SETTINGS
# ============================================================

NUM_BINS = 50

HEIGHT = 128
WIDTH = 128

INPUT_CHANNELS = 2

EPOCHS = 50

LEARNING_RATE = 0.001

BETA = 0.90

TARGET_WEIGHT = 15.0


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


voxel_045 = np.load(
    VOXEL_045
).astype(
    np.float32
)

voxel_047 = np.load(
    VOXEL_047
).astype(
    np.float32
)

heatmap_045 = np.load(
    HEATMAP_045
).astype(
    np.float32
)

heatmap_047 = np.load(
    HEATMAP_047
).astype(
    np.float32
)


print("045 voxel:", voxel_045.shape)
print("047 voxel:", voxel_047.shape)

print("045 heatmap:", heatmap_045.shape)
print("047 heatmap:", heatmap_047.shape)


# ============================================================
# CHECK DATA SHAPES
# ============================================================

expected_voxel_shape = (
    NUM_BINS,
    INPUT_CHANNELS,
    HEIGHT,
    WIDTH
)

expected_heatmap_shape = (
    NUM_BINS,
    HEIGHT,
    WIDTH
)


if voxel_045.shape != expected_voxel_shape:
    raise ValueError(
        f"045 voxel shape is {voxel_045.shape}, "
        f"expected {expected_voxel_shape}"
    )

if voxel_047.shape != expected_voxel_shape:
    raise ValueError(
        f"047 voxel shape is {voxel_047.shape}, "
        f"expected {expected_voxel_shape}"
    )

if heatmap_045.shape != expected_heatmap_shape:
    raise ValueError(
        f"045 heatmap shape is {heatmap_045.shape}, "
        f"expected {expected_heatmap_shape}"
    )

if heatmap_047.shape != expected_heatmap_shape:
    raise ValueError(
        f"047 heatmap shape is {heatmap_047.shape}, "
        f"expected {expected_heatmap_shape}"
    )


# ============================================================
# NORMALIZE EVENT VOXELS
# ============================================================

def normalize_voxel_grid(voxel):
    """
    Normalize every temporal bin independently
    using its maximum event value.
    """

    voxel = voxel.copy()

    for t in range(voxel.shape[0]):

        maximum = voxel[t].max()

        if maximum > 0:
            voxel[t] /= maximum

    return voxel


voxel_045 = normalize_voxel_grid(
    voxel_045
)

voxel_047 = normalize_voxel_grid(
    voxel_047
)


# ============================================================
# CREATE TRAINING DATA
# ============================================================

# 045:
# (50, 2, 128, 128)
#
# 047:
# (50, 2, 128, 128)
#
# Combined:
# (2, 50, 2, 128, 128)

X = np.stack(
    [
        voxel_045,
        voxel_047
    ],
    axis=0
)

Y = np.stack(
    [
        heatmap_045,
        heatmap_047
    ],
    axis=0
)


print()
print("=" * 70)
print("TRAINING DATA")
print("=" * 70)

print("X:", X.shape)
print("Y:", Y.shape)


X = torch.tensor(
    X,
    dtype=torch.float32,
    device=DEVICE
)

Y = torch.tensor(
    Y,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# MODEL
# ============================================================

class TemporalSNNCenterV2(nn.Module):

    def __init__(self):

        super().__init__()

        spike_grad = surrogate.fast_sigmoid()


        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            16
        )

        self.lif1 = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad
        )


        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            32
        )

        self.lif2 = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad
        )


        # ----------------------------------------------------
        # SPATIAL REDUCTION
        # ----------------------------------------------------

        self.pool = nn.AvgPool2d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # BLOCK 3
        # ----------------------------------------------------

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(
            32
        )

        self.lif3 = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad
        )


        # ----------------------------------------------------
        # HEATMAP HEAD
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1
        )


    def forward(self, x):

        num_timesteps = x.shape[1]

        # Initialize membrane states
        mem1 = self.lif1.init_leaky()

        mem2 = self.lif2.init_leaky()

        mem3 = self.lif3.init_leaky()

        outputs = []


        # ====================================================
        # TEMPORAL PROCESSING
        # ====================================================

        for t in range(num_timesteps):

            current = x[:, t]


            # ------------------------------------------------
            # BLOCK 1
            # ------------------------------------------------

            z1 = self.conv1(
                current
            )

            z1 = self.bn1(
                z1
            )

            spk1, mem1 = self.lif1(
                z1,
                mem1
            )


            # ------------------------------------------------
            # BLOCK 2
            # ------------------------------------------------

            z2 = self.conv2(
                spk1
            )

            z2 = self.bn2(
                z2
            )

            spk2, mem2 = self.lif2(
                z2,
                mem2
            )


            # ------------------------------------------------
            # POOL
            # ------------------------------------------------

            pooled = self.pool(
                spk2
            )


            # ------------------------------------------------
            # BLOCK 3
            # ------------------------------------------------

            z3 = self.conv3(
                pooled
            )

            z3 = self.bn3(
                z3
            )

            spk3, mem3 = self.lif3(
                z3,
                mem3
            )


            # ------------------------------------------------
            # HEATMAP
            # ------------------------------------------------

            heatmap = self.head(
                spk3
            )


            # 64 x 64 -> 128 x 128

            heatmap = F.interpolate(
                heatmap,
                size=(HEIGHT, WIDTH),
                mode="bilinear",
                align_corners=False
            )


            heatmap = torch.sigmoid(
                heatmap
            )


            outputs.append(
                heatmap
            )


        return torch.stack(
            outputs,
            dim=1
        )


# ============================================================
# CREATE MODEL
# ============================================================

model = TemporalSNNCenterV2().to(
    DEVICE
)


print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

def weighted_heatmap_loss(
    prediction,
    target
):

    # Give higher importance to ship-center pixels.
    weight = (
        1.0 +
        TARGET_WEIGHT * target
    )

    loss = (
        weight *
        (prediction - target) ** 2
    ).mean()

    return loss


criterion = weighted_heatmap_loss


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)


best_loss = float("inf")

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    optimizer.zero_grad()


    # Forward pass

    prediction = model(
        X
    )


    # Target:
    #
    # Y:
    # (2, 50, 128, 128)
    #
    # Convert to:
    # (2, 50, 1, 128, 128)

    target = Y.unsqueeze(
        2
    )


    loss = criterion(
        prediction,
        target
    )


    # Backpropagation

    loss.backward()


    # Prevent exploding gradients

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


    optimizer.step()


    current_loss = loss.item()

    history.append(
        current_loss
    )


    # Save best model

    if current_loss < best_loss:

        best_loss = current_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    # Print progress

    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"| Loss: {current_loss:.6f} "
            f"| Best: {best_loss:.6f}"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


# ============================================================
# FINAL PREDICTION
# ============================================================

model.eval()


with torch.no_grad():

    final_prediction = model(
        X
    )


print()
print("=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(
    "Prediction shape:",
    final_prediction.shape
)

print(
    "Prediction min:",
    final_prediction.min().item()
)

print(
    "Prediction max:",
    final_prediction.max().item()
)

print(
    "Prediction mean:",
    final_prediction.mean().item()
)


# ============================================================
# CONVERT PREDICTIONS TO NUMPY
# ============================================================

prediction_np = (
    final_prediction
    .squeeze(2)
    .cpu()
    .numpy()
)


print(
    "Saved prediction shape:",
    prediction_np.shape
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

PREDICTION_PATH = os.path.join(
    RESULT_DIR,
    "training_predictions.npy"
)

np.save(
    PREDICTION_PATH,
    prediction_np
)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w"
) as f:

    f.write(
        "Temporal SNN Center Detector V2\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    if torch.cuda.is_available():

        f.write(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}\n"
        )

    f.write(
        f"Input shape: {tuple(X.shape)}\n"
    )

    f.write(
        f"Target shape: {tuple(Y.shape)}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Beta: {BETA}\n"
    )

    f.write(
        f"Target weight: {TARGET_WEIGHT}\n"
    )

    f.write(
        f"Best loss: {best_loss:.8f}\n\n"
    )

    f.write(
        "Training history:\n"
    )

    for epoch, loss_value in enumerate(
        history,
        start=1
    ):

        f.write(
            f"Epoch {epoch}: "
            f"{loss_value:.8f}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Best model:")
print(
    os.path.abspath(
        MODEL_PATH
    )
)

print()
print("Training predictions:")
print(
    os.path.abspath(
        PREDICTION_PATH
    )
)

print()
print("Training summary:")
print(
    os.path.abspath(
        SUMMARY_PATH
    )
)

print()
print("=" * 70)