import numpy as np
import torch
import torch.nn as nn
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

MODEL_PATH = (
    BASE_DIR
    / "../models/temporal_snn_center.pth"
).resolve()

RESULT_DIR = (
    BASE_DIR
    / "../results/temporal_snn_inference"
).resolve()

RESULT_DIR.mkdir(parents=True, exist_ok=True)


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

class TemporalSNNCenter(nn.Module):

    def __init__(self):

        super().__init__()

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

        # x = [T, B, 2, 128, 128]

        T = x.shape[0]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        outputs = []

        for t in range(T):

            current = x[t]

            current = self.conv1(current)

            spk1, mem1 = self.lif1(
                current,
                mem1
            )

            current = self.pool1(spk1)

            current = self.conv2(current)

            spk2, mem2 = self.lif2(
                current,
                mem2
            )

            current = self.pool2(spk2)

            current = self.head(current)

            spk3, mem3 = self.lif3(
                current,
                mem3
            )

            heatmap = torch.nn.functional.interpolate(
                spk3,
                size=(128, 128),
                mode="bilinear",
                align_corners=False
            )

            outputs.append(heatmap)

        return torch.stack(outputs)


# ============================================================
# LOAD MODEL
# ============================================================

model = TemporalSNNCenter().to(device)

state = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state)

model.eval()

print()
print("Model loaded:")
print(MODEL_PATH)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_voxel(x):

    x = x.copy()

    for t in range(x.shape[0]):

        maximum = x[t].max()

        if maximum > 0:
            x[t] = x[t] / maximum

    return x


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    voxel_path,
    gt_heatmap_path,
    name
):

    print()
    print("=" * 60)
    print("Inference:", name)
    print("=" * 60)

    voxel = np.load(
        voxel_path
    ).astype(np.float32)

    gt_heatmap = np.load(
        gt_heatmap_path
    ).astype(np.float32)

    voxel = normalize_voxel(voxel)

    print("Voxel:", voxel.shape)
    print("GT heatmap:", gt_heatmap.shape)

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    x = torch.from_numpy(voxel)

    # [T, 2, H, W]
    # ->
    # [T, 1, 2, H, W]

    x = x.unsqueeze(1)

    x = x.to(device)

    # --------------------------------------------------------
    # Run SNN
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(x)

    print(
        "Prediction:",
        tuple(prediction.shape)
    )

    # --------------------------------------------------------
    # Remove batch/channel dimensions
    # --------------------------------------------------------

    prediction = prediction.squeeze(1).squeeze(1)

    # [50,128,128]

    prediction = prediction.cpu().numpy()

    # --------------------------------------------------------
    # Save predicted heatmaps
    # --------------------------------------------------------

    output_file = (
        RESULT_DIR
        / f"{name}_predicted_heatmaps.npy"
    )

    np.save(
        output_file,
        prediction
    )

    print(
        "Saved:",
        output_file
    )

    # --------------------------------------------------------
    # Compare basic statistics
    # --------------------------------------------------------

    print()
    print("Prediction statistics:")

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

    print()
    print("GT statistics:")

    print(
        "Min:",
        gt_heatmap.min()
    )

    print(
        "Max:",
        gt_heatmap.max()
    )

    print(
        "Mean:",
        gt_heatmap.mean()
    )

    return prediction, gt_heatmap


# ============================================================
# RUN 045
# ============================================================

pred_045, gt_045 = run_inference(
    VOXEL_045,
    HEATMAP_045,
    "ship_045"
)


# ============================================================
# RUN 047
# ============================================================

pred_047, gt_047 = run_inference(
    VOXEL_047,
    HEATMAP_047,
    "ship_047"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print(
    "Results directory:",
    RESULT_DIR
)