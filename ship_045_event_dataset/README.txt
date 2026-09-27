SHIP 045 – SYNTHETIC EVENT-BASED MARITIME DATASET
=================================================

1. DATASET SOURCE
-----------------

Source dataset:
VISO – Single Object Tracking (SOT)

Sequence:
Ship / 045

Original input:
320 satellite image frames

Frame resolution:
1345 × 451 pixels

The complete sequence of 320 frames was used to generate
a continuous synthetic event stream.


2. EVENT GENERATION
-------------------

Event generation method:
DVS-Voltmeter

Camera configuration:
DVS346

Total generated events:
1,428,729

Event stream duration:
Approximately 10.63 seconds


3. RAW EVENT STREAM
-------------------

Folder:

1_raw_event_stream/

File:

ship_045_full.txt


Each row represents one event.

Format:

timestamp  x  y  polarity


Example:

5766 670 279 0
6223 671 279 0
6485 670 278 0


Column description:

timestamp:
Time at which the event occurred, in microseconds.

x:
Horizontal pixel coordinate.

y:
Vertical pixel coordinate.

polarity:
0 = OFF event
1 = ON event


4. EVENT FRAME VISUALIZATIONS
-----------------------------

Folder:

2_event_frames/


This folder contains 50 event-frame visualizations.

The complete continuous event stream was divided into
50 temporal windows.

Each event frame represents the events occurring within
one temporal window.

IMPORTANT:

The 50 event images do NOT represent only 50 original
satellite frames.

All 320 original satellite frames were used to generate
the complete event stream.


5. VOXEL GRID
-------------

Folder:

3_voxel_grid/

File:

ship_045_voxel_grid.npy


Voxel grid shape:

(50, 2, 128, 128)


Dimension description:

50:
Number of temporal bins.

2:
Event polarity channels.

Channel 0 = OFF events
Channel 1 = ON events

128 × 128:
Spatially resized event representation.


The voxel grid is normalized between 0 and 1.


Example Python code:

import numpy as np

voxel_grid = np.load(
    "ship_045_voxel_grid.npy"
)

print(voxel_grid.shape)


6. TIME BINS
------------

Folder:

4_time_bins/

File:

ship_045_time_bins.csv


Columns:

bin_id

start_time_us

end_time_us

event_count


The file provides temporal information about the
50 event windows.


7. DATA REPRESENTATIONS
-----------------------

This dataset contains four representations of the
same event sequence:


1. Raw asynchronous event stream

timestamp, x, y, polarity


2. Event-frame visualizations

PNG images generated from temporal event windows.


3. Voxel grid

Structured tensor representation:

(50, 2, 128, 128)


4. Time-bin metadata

CSV file containing the temporal boundaries and
event counts for each bin.


8. IMPORTANT NOTE
-----------------

The event data is synthetically generated from
satellite image frames using DVS-Voltmeter.

It is not recorded using a physical event camera.

The dataset is intended for experimentation with
event-based vision and machine learning approaches,
including CNN and SNN models.