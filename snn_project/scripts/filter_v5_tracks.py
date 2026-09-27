import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

TRACKING_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5"
)


# ============================================================
# PARAMETERS
# ============================================================

MIN_TRACK_LENGTH = 20


# ============================================================
# FILTER
# ============================================================

def filter_tracks(sequence_name):

    input_file = os.path.join(
        TRACKING_DIR,
        f"{sequence_name}_tracks.csv"
    )

    output_file = os.path.join(
        TRACKING_DIR,
        f"{sequence_name}_tracks_filtered.csv"
    )


    df = pd.read_csv(
        input_file
    )


    print()
    print("=" * 70)
    print(
        f"FILTERING {sequence_name}"
    )
    print("=" * 70)


    track_lengths = (
        df.groupby(
            "track_id"
        )
        .size()
    )


    print()
    print(
        "Original tracks:"
    )


    for track_id, length in track_lengths.items():

        print(
            f"Track {int(track_id)}: "
            f"{int(length)} bins"
        )


    # --------------------------------------------------------
    # Keep sufficiently long trajectories
    # --------------------------------------------------------

    valid_ids = track_lengths[
        track_lengths >= MIN_TRACK_LENGTH
    ].index


    filtered = df[
        df["track_id"].isin(
            valid_ids
        )
    ].copy()


    # --------------------------------------------------------
    # Renumber tracks
    # --------------------------------------------------------

    mapping = {}

    for new_id, old_id in enumerate(
        sorted(
            filtered["track_id"].unique()
        ),
        start=1
    ):

        mapping[
            old_id
        ] = new_id


    filtered["track_id"] = (
        filtered["track_id"]
        .map(mapping)
    )


    filtered.to_csv(
        output_file,
        index=False
    )


    print()
    print(
        "Kept tracks:"
    )


    for track_id in sorted(
        filtered["track_id"].unique()
    ):

        track = filtered[
            filtered["track_id"]
            ==
            track_id
        ]


        print(
            f"Track {int(track_id)}: "
            f"{len(track)} bins"
        )


    print()
    print(
        "Removed short tracks:"
    )


    removed = set(
        track_lengths.index
    ) - set(
        valid_ids
    )


    for track_id in sorted(
        removed
    ):

        print(
            f"Track {int(track_id)}: "
            f"{int(track_lengths[track_id])} bins"
        )


    print()
    print(
        "Saved:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    filter_tracks(
        "ship_045"
    )

    filter_tracks(
        "ship_047"
    )

    print()
    print("=" * 70)
    print(
        "V5 FILTERING COMPLETE"
    )
    print("=" * 70)