import numpy as np
import cv2
import os


print("Loading FULL SNN spike output...")

spikes = np.load(
    "snn_project/data/ship_045_full_snn_spikes.npy"
)

print("Spike tensor shape:", spikes.shape)


# Remove batch dimension
# Original shape:
# (1, time, channels, height, width)

spikes = spikes[0]

print(
    "After removing batch dimension:",
    spikes.shape
)


# Create output directory

output_dir = (
    "snn_project/results/"
    "snn_spike_visualization_full"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ----------------------------------------
# VISUALIZE EACH TIME STEP
# ----------------------------------------

for t in range(spikes.shape[0]):

    # Shape:
    # (16, 128, 128)

    timestep_spikes = spikes[t]


    # Combine all feature channels

    combined = np.sum(
        timestep_spikes,
        axis=0
    )


    # Convert to binary spike map

    binary_map = (
        combined > 0
    ).astype(np.uint8)


    # Convert to image

    image = binary_map * 255


    # Output filename

    output_path = os.path.join(
        output_dir,
        f"spikes_time_{t:02d}.png"
    )


    # Save image

    cv2.imwrite(
        output_path,
        image
    )


    # Count spikes

    spike_count = timestep_spikes.sum()


    print(
        f"Time step {t}: "
        f"{int(spike_count)} spikes saved"
    )


print("\nVisualization completed!")

print("Results saved to:")
print(output_dir)