import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VOXEL_045 = (
    BASE_DIR
    / "../../ship_045_event_dataset/3_voxel_grid/ship_045_voxel_grid.npy"
).resolve()

HEATMAP_045 = (
    BASE_DIR
    / "../data/ship_045_center_heatmaps.npy"
).resolve()

VOXEL_047 = (
    BASE_DIR
    / "../ship_047_event_dataset/3_voxel_grid/ship_047_voxel_grid.npy"
).resolve()

HEATMAP_047 = (
    BASE_DIR
    / "../data/ship_047_center_heatmaps.npy"
).resolve()

MODEL_DIR = BASE_DIR / "../models"
RESULT_DIR = BASE_DIR / "../results/temporal_snn_center"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "temporal_snn_center.pth"
SUMMARY_PATH = RESULT_DIR / "training_summary.txt"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

voxel_045 = np.load(VOXEL_045).astype(np.float32)
heatmap_045 = np.load(HEATMAP_045).astype(np.float32)

voxel_047 = np.load(VOXEL_047).astype(np.float32)
heatmap_047 = np.load(HEATMAP_047).astype(np.float32)

print()
print("045 voxel:", voxel_045.shape)
print("045 heatmap:", heatmap_045.shape)
print("047 voxel:", voxel_047.shape)
print("047 heatmap:", heatmap_047.shape)


# ============================================================
# NORMALIZE EVENT VOXELS
# ============================================================

def normalize_voxel(x):
    """
    Normalize each temporal bin independently.
    """

    x = x.copy()

    for t in range(x.shape[0]):

        maximum = x[t].max()

        if maximum > 0:
            x[t] = x[t] / maximum

    return x


voxel_045 = normalize_voxel(voxel_045)
voxel_047 = normalize_voxel(voxel_047)


# ============================================================
# MODEL
# ============================================================

class TemporalSNNCenter(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(
            2,
            16,
            kernel_size=3,
            padding=1
        )

        self.lif1 = snn.Leaky(
            beta=0.90,
            learn_beta=True
        )

        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(
            beta=0.90,
            learn_beta=True
        )

        self.pool2 = nn.AvgPool2d(2)

        # Heatmap prediction
        self.head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        self.lif3 = snn.Leaky(
            beta=0.90,
            learn_beta=True
        )

    def forward(self, x):

        # x:
        # [T, B, 2, 128, 128]

        T = x.shape[0]

        # Membrane states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        outputs = []

        for t in range(T):

            current = x[t]

            # ------------------------------------------------
            # Layer 1
            # ------------------------------------------------

            current = self.conv1(current)

            spk1, mem1 = self.lif1(
                current,
                mem1
            )

            current = self.pool1(spk1)

            # ------------------------------------------------
            # Layer 2
            # ------------------------------------------------

            current = self.conv2(current)

            spk2, mem2 = self.lif2(
                current,
                mem2
            )

            current = self.pool2(spk2)

            # ------------------------------------------------
            # Heatmap head
            # ------------------------------------------------

            current = self.head(current)

            spk3, mem3 = self.lif3(
                current,
                mem3
            )

            # ------------------------------------------------
            # Upsample 32x32 -> 128x128
            # ------------------------------------------------

            heatmap = torch.nn.functional.interpolate(
                spk3,
                size=(128, 128),
                mode="bilinear",
                align_corners=False
            )

            outputs.append(heatmap)

        # [T, B, 1, 128, 128]
        return torch.stack(outputs)


# ============================================================
# CREATE MODEL
# ============================================================

model = TemporalSNNCenter().to(device)

print()
print(model)


# ============================================================
# LOSS
# ============================================================

def weighted_heatmap_loss(pred, target):

    """
    Give more importance to pixels around ship centers.

    Background occupies most of the image, so ordinary MSE
    can encourage the network to predict mostly zeros.
    """

    weight = 1.0 + 10.0 * target

    loss = weight * (pred - target) ** 2

    return loss.mean()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_sequence(
    model,
    voxel,
    heatmap,
    name,
    epochs=100,
    learning_rate=0.001
):

    print()
    print("=" * 60)
    print("Training:", name)
    print("=" * 60)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X = torch.from_numpy(voxel)

    Y = torch.from_numpy(heatmap)

    # X:
    # [50, 2, 128, 128]
    #
    # Add batch dimension:
    #
    # [50, 1, 2, 128, 128]

    X = X.unsqueeze(1)

    # Y:
    # [50, 128, 128]
    #
    # Add channel + batch:
    #
    # [50, 1, 1, 128, 128]

    Y = Y.unsqueeze(1).unsqueeze(1)

    X = X.to(device)
    Y = Y.to(device)

    print("Input:", tuple(X.shape))
    print("Target:", tuple(Y.shape))

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    losses = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        optimizer.zero_grad()

        prediction = model(X)

        # ----------------------------------------------------
        # IMPORTANT SHAPE CHECK
        # ----------------------------------------------------

        if prediction.shape != Y.shape:

            raise RuntimeError(
                f"Shape mismatch!\n"
                f"Prediction: {prediction.shape}\n"
                f"Target: {Y.shape}"
            )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = weighted_heatmap_loss(
            prediction,
            Y
        )

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.6f}"
            )

    return losses


# ============================================================
# TRAIN
# ============================================================

all_losses = {}

# Train on 045
losses_045 = train_sequence(
    model,
    voxel_045,
    heatmap_045,
    "ship_045",
    epochs=100,
    learning_rate=0.001
)

all_losses["ship_045"] = losses_045


# Train on 047
losses_047 = train_sequence(
    model,
    voxel_047,
    heatmap_047,
    "ship_047",
    epochs=100,
    learning_rate=0.001
)

all_losses["ship_047"] = losses_047


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w"
) as f:

    f.write("Temporal SNN Center Detector\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Device: {device}\n")

    if torch.cuda.is_available():
        f.write(
            f"GPU: {torch.cuda.get_device_name(0)}\n"
        )

    f.write("\n")

    f.write(
        f"045 voxel shape: {voxel_045.shape}\n"
    )

    f.write(
        f"045 heatmap shape: {heatmap_045.shape}\n"
    )

    f.write(
        f"047 voxel shape: {voxel_047.shape}\n"
    )

    f.write(
        f"047 heatmap shape: {heatmap_047.shape}\n"
    )

    f.write("\n")

    for name, losses in all_losses.items():

        f.write(f"{name}\n")
        f.write("-" * 40 + "\n")

        f.write(
            f"Initial loss: {losses[0]:.8f}\n"
        )

        f.write(
            f"Final loss: {losses[-1]:.8f}\n"
        )

        f.write(
            f"Minimum loss: {min(losses):.8f}\n"
        )

        f.write("\n")


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Model saved:", MODEL_PATH)
print("Summary saved:", SUMMARY_PATH)

print()
print("Final losses:")

for name, losses in all_losses.items():

    print(
        f"{name}: "
        f"{losses[0]:.6f} -> "
        f"{losses[-1]:.6f}"
    )