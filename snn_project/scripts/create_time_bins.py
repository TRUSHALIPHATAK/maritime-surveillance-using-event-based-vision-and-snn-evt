import numpy as np
import csv

# ---------------------------------------
# SETTINGS
# ---------------------------------------

EVENT_FILE = "../../data_samples/output/ship_047_full.txt"

OUTPUT_FILE = (
    "../ship_047_event_dataset/4_time_bins/"
    "ship_047_time_bins.csv"
)

NUM_BINS = 50


# ---------------------------------------
# LOAD EVENTS
# ---------------------------------------

print("Loading event stream...")

events = np.loadtxt(
    EVENT_FILE,
    dtype=np.int64
)

timestamps = events[:, 0]

start_time = timestamps.min()
end_time = timestamps.max()

duration = end_time - start_time

bin_duration = duration / NUM_BINS


# ---------------------------------------
# CREATE TIME BIN INFORMATION
# ---------------------------------------

rows = []

for i in range(NUM_BINS):

    bin_start = start_time + i * bin_duration
    bin_end = start_time + (i + 1) * bin_duration

    if i == NUM_BINS - 1:

        mask = (
            (timestamps >= bin_start)
            &
            (timestamps <= bin_end)
        )

    else:

        mask = (
            (timestamps >= bin_start)
            &
            (timestamps < bin_end)
        )

    event_count = np.sum(mask)

    rows.append([
        i,
        int(bin_start),
        int(bin_end),
        int(event_count)
    ])


# ---------------------------------------
# SAVE CSV
# ---------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "bin_id",
        "start_time_us",
        "end_time_us",
        "event_count"
    ])

    writer.writerows(rows)


print("\nTime bin information created!")

print("Number of bins:", NUM_BINS)

print("Start time:", start_time)

print("End time:", end_time)

print(
    "Bin duration:",
    bin_duration,
    "microseconds"
)

print("\nSaved to:")

print(OUTPUT_FILE)