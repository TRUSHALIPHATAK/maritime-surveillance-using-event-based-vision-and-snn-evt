import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


# ============================================================
# TEMPORAL SNN CENTER DETECTOR V2 - INFERENCE
# ============================================================

print("=" * 70)
print("TEMPORAL SNN CENTER DETECTOR V2 - INFERENCE")
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
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    SNN_PROJECT_DIR
)


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
    "temporal_snn_center_v2.pth"
)

RESULT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v2"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

HEIGHT = 128
WIDTH = 128

BETA = 0.90


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING FILES")
print("=" * 70)


required_files = {
    "045 voxel": VOXEL_045,
    "047 voxel": VOXEL_047,
    "model": MODEL_PATH
}


for name, path in required_files.items():

    if os.path.exists(path):

        print(
            f"{name:12s}: FOUND"
        )

    else:

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_voxel_grid(voxel):

    voxel = voxel.copy()

    for t in range(
        voxel.shape[0]
    ):

        maximum = voxel[t].max()

        if maximum > 0:

            voxel[t] /= maximum

    return voxel


# ============================================================
# MODEL
# ============================================================

class TemporalSNNCenterV2(nn.Module):

    def __init__(self):

        super().__init__()

        spike_grad = surrogate.fast_sigmoid()


        self.conv1 = nn.Conv2d(
            2,
            16,
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


        self.conv2 = nn.Conv2d(
            16,
            32,
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


        self.pool = nn.AvgPool2d(
            kernel_size=2
        )


        self.conv3 = nn.Conv2d(
            32,
            32,
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


        self.head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        num_timesteps = x.shape[1]

        mem1 = self.lif1.init_leaky()

        mem2 = self.lif2.init_leaky()

        mem3 = self.lif3.init_leaky()

        outputs = []


        for t in range(
            num_timesteps
        ):

            current = x[:, t]


            # BLOCK 1

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


            # BLOCK 2

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


            # POOL

            pooled = self.pool(
                spk2
            )


            # BLOCK 3

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


            # HEATMAP

            heatmap = self.head(
                spk3
            )


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
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)


model = TemporalSNNCenterV2().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def run_inference(
    voxel_path,
    output_name
):

    print()
    print("=" * 70)
    print(f"PROCESSING {output_name}")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD VOXEL GRID
    # --------------------------------------------------------

    voxel = np.load(
        voxel_path
    ).astype(
        np.float32
    )


    print(
        "Original voxel:",
        voxel.shape
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    voxel = normalize_voxel_grid(
        voxel
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    x = torch.tensor(
        voxel,
        dtype=torch.float32,
        device=DEVICE
    )

    x = x.unsqueeze(
        0
    )


    print(
        "Model input:",
        x.shape
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            x
        )


    print(
        "Raw prediction:",
        prediction.shape
    )


    # --------------------------------------------------------
    # REMOVE BATCH/CHANNEL DIMENSIONS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        RESULT_DIR,
        output_name
    )

    np.save(
        output_path,
        prediction
    )


    print()
    print(
        "Saved:",
        os.path.abspath(
            output_path
        )
    )


# ============================================================
# RUN 045
# ============================================================

run_inference(
    VOXEL_045,
    "ship_045_predicted_heatmaps_v2.npy"
)


# ============================================================
# RUN 047
# ============================================================

run_inference(
    VOXEL_047,
    "ship_047_predicted_heatmaps_v2.npy"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)