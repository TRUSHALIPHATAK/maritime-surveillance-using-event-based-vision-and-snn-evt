import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SNN_PROJECT_DIR)

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

MODEL_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "temporal_snn_center_v3.pth"
)

SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "training_summary.txt"
)

PREDICTION_PATH = os.path.join(
    RESULT_DIR,
    "training_predictions.npy"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NUM_BINS = 50
HEIGHT = 128
WIDTH = 128

EPOCHS = 60

LEARNING_RATE = 0.0005

BETA = 0.90

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0

GRAD_CLIP = 1.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V3")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    VOXEL_045,
    VOXEL_047,
    HEATMAP_045,
    HEATMAP_047
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

voxel_045 = np.load(VOXEL_045).astype(np.float32)

voxel_047 = np.load(VOXEL_047).astype(np.float32)

heatmap_045 = np.load(
    HEATMAP_045
).astype(np.float32)

heatmap_047 = np.load(
    HEATMAP_047
).astype(np.float32)


print("045 voxel:", voxel_045.shape)
print("047 voxel:", voxel_047.shape)

print("045 heatmap:", heatmap_045.shape)
print("047 heatmap:", heatmap_047.shape)


# ============================================================
# NORMALIZE EACH TEMPORAL BIN
# ============================================================

def normalize_voxel(voxel):

    output = np.zeros_like(voxel)

    for t in range(voxel.shape[0]):

        current = voxel[t]

        max_value = current.max()

        if max_value > 0:

            output[t] = current / max_value

    return output


voxel_045 = normalize_voxel(voxel_045)

voxel_047 = normalize_voxel(voxel_047)


# ============================================================
# CONVERT TO TORCH
# ============================================================

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


X = torch.tensor(
    X,
    dtype=torch.float32
)

Y = torch.tensor(
    Y,
    dtype=torch.float32
)


print("\nInput X:", X.shape)
print("Target Y:", Y.shape)


X = X.to(device)
Y = Y.to(device)


# ============================================================
# MODEL
# ============================================================

class TemporalSNNV3(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            2,
            16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.lif1 = snn.Leaky(
            beta=BETA,
            spike_grad=surrogate.fast_sigmoid()
        )


        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.lif2 = snn.Leaky(
            beta=BETA,
            spike_grad=surrogate.fast_sigmoid()
        )


        self.pool = nn.AvgPool2d(
            kernel_size=2
        )


        self.conv3 = nn.Conv2d(
            32,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(32)

        self.lif3 = snn.Leaky(
            beta=BETA,
            spike_grad=surrogate.fast_sigmoid()
        )


        self.dropout = nn.Dropout2d(
            p=0.15
        )


        self.head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        batch_size = x.shape[0]

        mem1 = self.lif1.init_leaky()

        mem2 = self.lif2.init_leaky()

        mem3 = self.lif3.init_leaky()

        outputs = []


        for t in range(x.shape[1]):

            xt = x[:, t]


            # ----------------------------------------------
            # Layer 1
            # ----------------------------------------------

            z = self.conv1(xt)

            z = self.bn1(z)

            spk1, mem1 = self.lif1(
                z,
                mem1
            )


            # ----------------------------------------------
            # Layer 2
            # ----------------------------------------------

            z = self.conv2(spk1)

            z = self.bn2(z)

            spk2, mem2 = self.lif2(
                z,
                mem2
            )


            # ----------------------------------------------
            # Downsample
            # ----------------------------------------------

            z = self.pool(spk2)


            # ----------------------------------------------
            # Layer 3
            # ----------------------------------------------

            z = self.conv3(z)

            z = self.bn3(z)

            spk3, mem3 = self.lif3(
                z,
                mem3
            )


            z = self.dropout(spk3)


            # ----------------------------------------------
            # Heatmap head
            # ----------------------------------------------

            z = self.head(z)


            # 64 x 64 → 128 x 128

            z = F.interpolate(
                z,
                size=(HEIGHT, WIDTH),
                mode="bilinear",
                align_corners=False
            )


            z = torch.sigmoid(z)


            outputs.append(z)


        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs


# ============================================================
# FOCAL HEATMAP LOSS
# ============================================================

def focal_heatmap_loss(
    prediction,
    target,
    alpha=FOCAL_ALPHA,
    beta=FOCAL_BETA
):

    eps = 1e-6

    prediction = torch.clamp(
        prediction,
        eps,
        1.0 - eps
    )


    # --------------------------------------------------------
    # Positive pixels
    # --------------------------------------------------------

    positive_mask = target >= 0.1

    negative_mask = target < 0.1


    positive_loss = -(
        torch.pow(
            1.0 - prediction,
            alpha
        )
        *
        torch.log(prediction)
        *
        target
    )


    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    negative_weight = torch.pow(
        1.0 - target,
        beta
    )


    negative_loss = -(
        torch.pow(
            prediction,
            alpha
        )
        *
        torch.log(
            1.0 - prediction
        )
        *
        negative_weight
    )


    # Stronger background suppression
    positive_loss = positive_loss * positive_mask

    negative_loss = negative_loss * negative_mask


    positive_count = positive_mask.sum().clamp(
        min=1
    )

    negative_count = negative_mask.sum().clamp(
        min=1
    )


    positive_loss = (
        positive_loss.sum()
        /
        positive_count
    )

    negative_loss = (
        negative_loss.sum()
        /
        negative_count
    )


    return (
        positive_loss +
        negative_loss
    )


# ============================================================
# CREATE MODEL
# ============================================================

model = TemporalSNNV3().to(device)

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print(model)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


loss_history = []

best_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()


    prediction = model(X)


    # --------------------------------------------------------
    # prediction:
    #
    # (2, 50, 1, 128, 128)
    #
    # target:
    #
    # (2, 50, 128, 128)
    # --------------------------------------------------------

    target = Y.unsqueeze(2)


    loss = focal_heatmap_loss(
        prediction,
        target
    )


    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )


    optimizer.step()


    loss_value = loss.item()

    loss_history.append(
        loss_value
    )


    if loss_value < best_loss:

        best_loss = loss_value

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"Loss: {loss_value:.6f} "
            f"Best: {best_loss:.6f}"
        )


# ============================================================
# SAVE TRAINING PREDICTIONS
# ============================================================

model.eval()

with torch.no_grad():

    final_prediction = model(X)


final_prediction = (
    final_prediction
    .squeeze(2)
    .cpu()
    .numpy()
)


np.save(
    PREDICTION_PATH,
    final_prediction
)


# ============================================================
# SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w"
) as f:

    f.write(
        "Temporal SNN Center Detector V3\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    if device.type == "cuda":

        f.write(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}\n"
        )

    f.write(
        f"\nEpochs: {EPOCHS}\n"
    )

    f.write(
        f"Learning rate: "
        f"{LEARNING_RATE}\n"
    )

    f.write(
        f"Best loss: "
        f"{best_loss:.6f}\n"
    )

    f.write(
        f"Final loss: "
        f"{loss_history[-1]:.6f}\n"
    )

    f.write(
        f"\nPrediction shape: "
        f"{final_prediction.shape}\n"
    )


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best loss: {best_loss:.6f}"
)

print(
    f"Final prediction shape: "
    f"{final_prediction.shape}"
)

print(
    f"Prediction min: "
    f"{final_prediction.min():.6f}"
)

print(
    f"Prediction max: "
    f"{final_prediction.max():.6f}"
)

print(
    f"Prediction mean: "
    f"{final_prediction.mean():.6f}"
)

print("\nModel saved:")
print(MODEL_PATH)

print("\nPredictions saved:")
print(PREDICTION_PATH)

print("\nSummary saved:")
print(SUMMARY_PATH)