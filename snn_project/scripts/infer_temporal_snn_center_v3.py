import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


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

MODEL_PATH = os.path.join(
    SNN_PROJECT_DIR,
    "models",
    "temporal_snn_center_v3.pth"
)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3"
)

PRED_045 = os.path.join(
    RESULT_DIR,
    "ship_045_predicted_heatmaps_v3.npy"
)

PRED_047 = os.path.join(
    RESULT_DIR,
    "ship_047_predicted_heatmaps_v3.npy"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

HEIGHT = 128
WIDTH = 128
BETA = 0.90


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V3 - INFERENCE")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)

files = {
    "045 voxel": VOXEL_045,
    "047 voxel": VOXEL_047,
    "model": MODEL_PATH
}

for name, path in files.items():

    status = (
        "FOUND"
        if os.path.exists(path)
        else "NOT FOUND"
    )

    print(
        f"{name:<15}: {status}"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)


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

        # Dropout is disabled automatically in eval mode
        self.dropout = nn.Dropout2d(
            p=0.15
        )

        self.head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        outputs = []

        for t in range(x.shape[1]):

            xt = x[:, t]

            # Layer 1
            z = self.conv1(xt)
            z = self.bn1(z)

            spk1, mem1 = self.lif1(
                z,
                mem1
            )

            # Layer 2
            z = self.conv2(spk1)
            z = self.bn2(z)

            spk2, mem2 = self.lif2(
                z,
                mem2
            )

            # Downsample
            z = self.pool(spk2)

            # Layer 3
            z = self.conv3(z)
            z = self.bn3(z)

            spk3, mem3 = self.lif3(
                z,
                mem3
            )

            z = self.dropout(spk3)

            # Heatmap head
            z = self.head(z)

            z = F.interpolate(
                z,
                size=(HEIGHT, WIDTH),
                mode="bilinear",
                align_corners=False
            )

            z = torch.sigmoid(z)

            outputs.append(z)

        return torch.stack(
            outputs,
            dim=1
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = TemporalSNNV3().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_voxel(voxel):

    output = np.zeros_like(
        voxel,
        dtype=np.float32
    )

    for t in range(voxel.shape[0]):

        current = voxel[t]

        max_value = current.max()

        if max_value > 0:

            output[t] = (
                current / max_value
            )

    return output


# ============================================================
# PROCESS SEQUENCE
# ============================================================

def process_sequence(
    sequence_name,
    voxel_path,
    output_path
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {sequence_name}"
    )
    print("=" * 70)

    voxel = np.load(
        voxel_path
    ).astype(np.float32)

    print(
        "Original voxel:",
        voxel.shape
    )

    voxel = normalize_voxel(
        voxel
    )

    x = torch.tensor(
        voxel,
        dtype=torch.float32
    )

    x = x.unsqueeze(0)

    x = x.to(device)

    print(
        "Model input:",
        x.shape
    )

    with torch.no_grad():

        prediction = model(x)

    print(
        "Raw prediction:",
        prediction.shape
    )

    prediction = (
        prediction
        .squeeze(0)
        .squeeze(1)
        .cpu()
        .numpy()
    )

    print(
        "Saved prediction:",
        prediction.shape
    )

    print(
        "Min:",
        prediction.min()
    )

    print(
        "Max:",
        prediction.max()
    )

    print(
        "Mean:",
        prediction.mean()
    )

    print(
        "Non-zero:",
        np.count_nonzero(prediction)
    )

    np.save(
        output_path,
        prediction
    )

    print(
        "\nSaved:",
        output_path
    )


# ============================================================
# RUN
# ============================================================

process_sequence(
    "ship_045",
    VOXEL_045,
    PRED_045
)

process_sequence(
    "ship_047",
    VOXEL_047,
    PRED_047
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)