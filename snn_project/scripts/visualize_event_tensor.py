import numpy as np
import cv2
import os

# ----------------------------------------
# SETTINGS
# ----------------------------------------

INPUT_FILE = "snn_project/data/ship_045_events.npy"

OUTPUT_DIR = "snn_project/results/tensor_visualization"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------
# LOAD TENSOR
# ----------------------------------------

tensor = np.load(INPUT_FILE)

print("Tensor loaded successfully")
print("Tensor shape:", tensor.shape)


# Shape:
# [time, polarity, height, width]


# ----------------------------------------
# VISUALIZE EACH TIME STEP
# ----------------------------------------

for t in range(tensor.shape[0]):

    off_events = tensor[t, 0]
    on_events = tensor[t, 1]

    # Normalize each channel for visualization
    off_image = cv2.normalize(
        off_events,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    on_image = cv2.normalize(
        on_events,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Create a combined visualization
    #
    # Red   = OFF events
    # Green = ON events
    #

    combined = np.zeros(
        (128, 128, 3),
        dtype=np.uint8
    )

    combined[:, :, 2] = off_image
    combined[:, :, 1] = on_image


    # Save individual channels

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"time_{t:02d}_off.png"
        ),
        off_image
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"time_{t:02d}_on.png"
        ),
        on_image
    )


    # Save combined image

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"time_{t:02d}_combined.png"
        ),
        combined
    )


    print(
        f"Saved visualization for "
        f"time step {t}"
    )


print("\nVisualization completed!")
print("Results saved to:", OUTPUT_DIR)