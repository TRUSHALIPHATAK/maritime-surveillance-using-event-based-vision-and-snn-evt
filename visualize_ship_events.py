import numpy as np
import cv2
import os

# ==============================
# PATHS
# ==============================

EVENT_FILE = "data_samples/output/ship_045.txt"
OUTPUT_FOLDER = "data_samples/event_visualization"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==============================
# LOAD EVENTS
# ==============================

events = np.loadtxt(EVENT_FILE, dtype=np.int32)

print("Events loaded successfully")
print("Event shape:", events.shape)
print("First event:", events[0])
print("Last event:", events[-1])


# ==============================
# IMAGE SIZE
# ==============================

HEIGHT = 451
WIDTH = 1345


# ==============================
# TIME WINDOWS
# ==============================

NUM_WINDOWS = 10

start_time = events[0, 0]
end_time = events[-1, 0]

time_edges = np.linspace(
    start_time,
    end_time,
    NUM_WINDOWS + 1
)


# ==============================
# CREATE EVENT IMAGES
# ==============================

for i in range(NUM_WINDOWS):

    start = time_edges[i]
    end = time_edges[i + 1]

    mask = (
        (events[:, 0] >= start) &
        (events[:, 0] < end)
    )

    window_events = events[mask]

    image = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    for event in window_events:

        t, x, y, polarity = event

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:

            # ON event
            if polarity == 1:
                image[y, x] = [255, 255, 255]

            # OFF event
            else:
                image[y, x] = [128, 128, 128]

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"event_{i:02d}.png"
    )

    cv2.imwrite(filename, image)

    print(
        f"Saved {filename} | "
        f"Events: {len(window_events)}"
    )


print("\nVisualization completed successfully!")