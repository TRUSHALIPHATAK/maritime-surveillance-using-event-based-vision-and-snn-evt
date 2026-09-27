import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)
np.random.seed(42)


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
# LOAD VOXEL GRID
# ============================================================

print("\nLoading 047 voxel grid...")

event_data = np.load(
    "../ship_047_event_dataset/"
    "3_voxel_grid/"
    "ship_047_voxel_grid.npy"
)

print(
    "Voxel grid shape:",
    event_data.shape
)


# ============================================================
# CONVERT TO PYTORCH TENSOR
# ============================================================

event_tensor = torch.tensor(
    event_data,
    dtype=torch.float32,
    device=device
)


# Original:
# (time, polarity, height, width)
#
# New:
# (batch, time, polarity, height, width)

event_tensor = event_tensor.unsqueeze(0)

print(
    "Input tensor shape:",
    event_tensor.shape
)


# ============================================================
# SIMPLE SNN MODEL
# ============================================================

class SimpleSNN(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # Convolution layer
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1
        )


        # ----------------------------------------------------
        # Leaky Integrate-and-Fire neuron
        # ----------------------------------------------------

        spike_grad = surrogate.fast_sigmoid()

        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )


    def forward(self, x):

        # Initialize membrane potential

        mem1 = self.lif1.init_leaky()

        spike_record = []


        # ----------------------------------------------------
        # Process each temporal bin
        # ----------------------------------------------------

        for step in range(
            x.size(1)
        ):

            # Current time step
            current_input = x[:, step]


            # Convolution
            current = self.conv1(
                current_input
            )


            # LIF neuron
            spike, mem1 = self.lif1(
                current,
                mem1
            )


            # Store spikes
            spike_record.append(
                spike
            )


        return torch.stack(
            spike_record,
            dim=1
        )


# ============================================================
# CREATE MODEL
# ============================================================

model = SimpleSNN().to(device)

model.eval()


print("\nRunning SNN...")


# ============================================================
# RUN SNN
# ============================================================

with torch.no_grad():

    spikes = model(
        event_tensor
    )


# ============================================================
# RESULTS
# ============================================================

print("\nSNN completed successfully!")

print(
    "Output spike tensor shape:",
    spikes.shape
)


# ============================================================
# SPIKE STATISTICS
# ============================================================

total_spikes = spikes.sum().item()


print("\n----------------------------")
print("SNN SPIKE STATISTICS")
print("----------------------------")

print(
    "Total spikes:",
    total_spikes
)


for t in range(
    spikes.shape[1]
):

    timestep_spikes = (
        spikes[:, t].sum().item()
    )

    print(
        f"Time step {t}: "
        f"{timestep_spikes} spikes"
    )


print("----------------------------")


# ============================================================
# SAVE SPIKE OUTPUT
# ============================================================

output_file = (
    "../data/"
    "ship_047_full_snn_spikes.npy"
)


np.save(
    output_file,
    spikes.cpu().numpy()
)


print(
    "\nSpike output saved to:"
)

print(
    output_file
)