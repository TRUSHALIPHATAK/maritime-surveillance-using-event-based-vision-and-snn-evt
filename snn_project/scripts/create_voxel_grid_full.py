import numpy as np
import os


print("Loading event stream...")


# --------------------------------------------------
# LOAD 047 EVENT STREAM
# --------------------------------------------------

events = np.loadtxt(
    "../../data_samples/output/ship_047_full.txt",
    dtype=np.int64
)


print("Event shape:", events.shape)


# --------------------------------------------------
# EVENT FORMAT
#
# timestamp x y polarity
# --------------------------------------------------

timestamps = events[:, 0]
x = events[:, 1]
y = events[:, 2]
polarity = events[:, 3]


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

NUM_BINS = 50

HEIGHT = 128
WIDTH = 128


# Original image resolution

ORIGINAL_WIDTH = 1345
ORIGINAL_HEIGHT = 451


print("\nCreating voxel grid...")


# --------------------------------------------------
# CREATE VOXEL GRID
#
# Shape:
#
# (time_bins, polarity, height, width)
# --------------------------------------------------

voxel_grid = np.zeros(
    (
        NUM_BINS,
        2,
        HEIGHT,
        WIDTH
    ),
    dtype=np.float32
)


# --------------------------------------------------
# TIME RANGE
# --------------------------------------------------

start_time = timestamps.min()

end_time = timestamps.max()


bin_duration = (
    end_time - start_time
) / NUM_BINS


print("Start time:", start_time)

print("End time:", end_time)

print("Number of bins:", NUM_BINS)

print(
    "Bin duration:",
    bin_duration,
    "microseconds"
)


# --------------------------------------------------
# PROCESS EVENTS
# --------------------------------------------------

for i in range(len(events)):

    # -----------------------------
    # TIME BIN
    # -----------------------------

    t = timestamps[i]

    time_bin = int(
        (t - start_time)
        / bin_duration
    )

    # Prevent overflow

    if time_bin >= NUM_BINS:

        time_bin = NUM_BINS - 1

    # -----------------------------
    # SPATIAL RESIZE
    # -----------------------------

    x_resized = int(
        x[i]
        * WIDTH
        / ORIGINAL_WIDTH
    )

    y_resized = int(
        y[i]
        * HEIGHT
        / ORIGINAL_HEIGHT
    )

    # Prevent boundary errors

    x_resized = min(
        max(x_resized, 0),
        WIDTH - 1
    )

    y_resized = min(
        max(y_resized, 0),
        HEIGHT - 1
    )

    # -----------------------------
    # POLARITY
    # -----------------------------

    p = polarity[i]

    # -----------------------------
    # ADD EVENT
    # -----------------------------

    voxel_grid[
        time_bin,
        p,
        y_resized,
        x_resized
    ] += 1


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

print("\nNormalizing voxel grid...")


for t in range(NUM_BINS):

    maximum = voxel_grid[t].max()

    if maximum > 0:

        voxel_grid[t] = (
            voxel_grid[t]
            / maximum
        )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = (
    "../ship_047_event_dataset/"
    "3_voxel_grid/"
    "ship_047_voxel_grid.npy"
)


os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)


np.save(
    output_path,
    voxel_grid
)


print("\nVoxel grid created successfully!")

print(
    "Voxel grid shape:",
    voxel_grid.shape
)


print(
    "\nSaved to:"
)


print(output_path)