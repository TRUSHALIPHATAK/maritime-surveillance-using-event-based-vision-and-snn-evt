import os
import numpy as np
import cv2

# ----------------------------------------
# SETTINGS
# ----------------------------------------

EVENT_FILE = "data_samples/output/ship_045_full.txt"

OUTPUT_DIR = "data_samples/event_visualization_full"

NUM_TIME_BINS = 50

# Original image resolution
WIDTH = 1345
HEIGHT = 451


# ----------------------------------------
# CREATE OUTPUT DIRECTORY
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------
# LOAD EVENTS
# ----------------------------------------

print("Loading event data...")

events = np.loadtxt(
    EVENT_FILE,
    dtype=np.int32
)

print("Events loaded successfully!")
print("Event shape:", events.shape)


# ----------------------------------------
# EVENT INFORMATION
# ----------------------------------------

start_time = events[:, 0].min()
end_time = events[:, 0].max()

print("Start time:", start_time)
print("End time:", end_time)

total_duration = end_time - start_time

print("Total duration:", total_duration, "microseconds")


# ----------------------------------------
# CREATE TEMPORAL BINS
# ----------------------------------------

time_edges = np.linspace(
    start_time,
    end_time,
    NUM_TIME_BINS + 1
)


# ----------------------------------------
# GENERATE EVENT IMAGES
# ----------------------------------------

for i in range(NUM_TIME_BINS):

    bin_start = time_edges[i]
    bin_end = time_edges[i + 1]

    # Select events in this time interval

    mask = (
        (events[:, 0] >= bin_start) &
        (events[:, 0] < bin_end)
    )

    bin_events = events[mask]

    print(
        f"Processing bin {i:03d} | "
        f"Events: {len(bin_events)}"
    )


    # ----------------------------------------
    # CREATE ON / OFF EVENT IMAGES
    # ----------------------------------------

    on_image = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8
    )

    off_image = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8
    )


    # ----------------------------------------
    # ADD EVENTS TO IMAGES
    # ----------------------------------------

    for event in bin_events:

        timestamp, x, y, polarity = event

        if (
            0 <= x < WIDTH and
            0 <= y < HEIGHT
        ):

            if polarity == 1:

                on_image[y, x] = 255

            else:

                off_image[y, x] = 255


    # ----------------------------------------
    # CREATE COMBINED IMAGE
    # ----------------------------------------

    combined = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )


    # ON events = Red
    combined[:, :, 2] = on_image

    # OFF events = Blue
    combined[:, :, 0] = off_image


    # ----------------------------------------
    # SAVE IMAGES
    # ----------------------------------------

    on_path = os.path.join(
        OUTPUT_DIR,
        f"event_{i:03d}_on.png"
    )

    off_path = os.path.join(
        OUTPUT_DIR,
        f"event_{i:03d}_off.png"
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        f"event_{i:03d}_combined.png"
    )


    cv2.imwrite(
        on_path,
        on_image
    )

    cv2.imwrite(
        off_path,
        off_image
    )

    cv2.imwrite(
        combined_path,
        combined
    )


print("\n--------------------------------")
print("Visualization completed!")
print("--------------------------------")

print(
    "Results saved to:",
    OUTPUT_DIR
)