import numpy as np
import cv2
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

EVENT_FILE = "data_samples/output/ship_045.txt"

OUTPUT_DIR = "snn_project/data"

NUM_TIME_BINS = 10

OUTPUT_WIDTH = 128
OUTPUT_HEIGHT = 128


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD EVENTS
# --------------------------------------------------

print("Loading event data...")

events = np.loadtxt(
    EVENT_FILE,
    dtype=np.int32
)

print("Event shape:", events.shape)


# Event format:
# timestamp, x, y, polarity

timestamps = events[:, 0]
xs = events[:, 1]
ys = events[:, 2]
polarities = events[:, 3]


# --------------------------------------------------
# GET ORIGINAL SENSOR SIZE
# --------------------------------------------------

ORIGINAL_WIDTH = xs.max() + 1
ORIGINAL_HEIGHT = ys.max() + 1

print("Original event resolution:")
print("Width:", ORIGINAL_WIDTH)
print("Height:", ORIGINAL_HEIGHT)


# --------------------------------------------------
# TIME RANGE
# --------------------------------------------------

start_time = timestamps.min()
end_time = timestamps.max()

time_edges = np.linspace(
    start_time,
    end_time,
    NUM_TIME_BINS + 1
)

print("Start time:", start_time)
print("End time:", end_time)

print("\nCreating event tensor...")


# --------------------------------------------------
# EVENT TENSOR
#
# Shape:
#
# [time_bins, polarity, height, width]
#
# polarity:
# 0 = OFF
# 1 = ON
# --------------------------------------------------

event_tensor = np.zeros(
    (
        NUM_TIME_BINS,
        2,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH
    ),
    dtype=np.float32
)


# --------------------------------------------------
# PROCESS EACH TIME BIN
# --------------------------------------------------

for t in range(NUM_TIME_BINS):

    start_t = time_edges[t]
    end_t = time_edges[t + 1]

    # Select events inside time window

    mask = (
        (timestamps >= start_t)
        &
        (timestamps < end_t)
    )

    bin_events = events[mask]

    print(
        f"Time bin {t}: "
        f"{len(bin_events)} events"
    )


    # Process events

    for event in bin_events:

        timestamp, x, y, polarity = event


        # Resize coordinates

        new_x = int(
            x * OUTPUT_WIDTH
            / ORIGINAL_WIDTH
        )

        new_y = int(
            y * OUTPUT_HEIGHT
            / ORIGINAL_HEIGHT
        )


        # Prevent boundary errors

        new_x = min(
            new_x,
            OUTPUT_WIDTH - 1
        )

        new_y = min(
            new_y,
            OUTPUT_HEIGHT - 1
        )


        # Add event

        event_tensor[
            t,
            polarity,
            new_y,
            new_x
        ] += 1


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

max_value = event_tensor.max()

if max_value > 0:

    event_tensor = (
        event_tensor
        / max_value
    )


# --------------------------------------------------
# SAVE TENSOR
# --------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "ship_045_events.npy"
)

np.save(
    output_path,
    event_tensor
)


print("\nEvent tensor created successfully!")

print("Tensor shape:", event_tensor.shape)

print(
    "Saved to:",
    output_path
)