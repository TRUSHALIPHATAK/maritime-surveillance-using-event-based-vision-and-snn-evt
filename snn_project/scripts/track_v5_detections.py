import os
import numpy as np
import pandas as pd


# ============================================================
# V5 - TRAJECTORY-AWARE MULTI-OBJECT TRACKER
# ============================================================
#
# Strategy:
#
# Strong detections:
#       score >= 0.72
#       -> can start / update tracks
#
# Weak detections:
#       0.60 <= score < 0.72
#       -> ONLY accepted when they are consistent with
#          an existing predicted trajectory
#
# Tracking:
#       position + velocity prediction
#       one-to-one association
#       short-term missed detections
#       tentative -> confirmed tracks
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SNN_PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    SNN_PROJECT_DIR
)


# ============================================================
# INPUT HEATMAPS
# ============================================================

PRED_045 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_045_predicted_heatmaps_v3.npy"
)

PRED_047 = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_snn_center_v3",
    "ship_047_predicted_heatmaps_v3.npy"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    SNN_PROJECT_DIR,
    "results",
    "temporal_tracking_v5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DETECTION PARAMETERS
# ============================================================

# V3 strong threshold.
STRONG_THRESHOLD = 0.72

# Weak responses are considered only for trajectory recovery.
WEAK_THRESHOLD = 0.60

# Local maxima separation.
MIN_DISTANCE = 6

# We don't need 10 arbitrary peaks.
MAX_CANDIDATES = 10


# ============================================================
# TRACK PARAMETERS
# ============================================================

# Strong detections can be farther away because they may
# represent a real object even if velocity prediction is
# imperfect.
STRONG_MAX_DISTANCE = 12.0

# Weak detections need to be closer to predicted position.
WEAK_MAX_DISTANCE = 7.0

# Maximum bins a confirmed track can survive without
# an actual detection.
MAX_MISSED = 3

# A tentative track must accumulate this many hits.
CONFIRMATION_HITS = 3

# Maximum age of a tentative track.
MAX_TENTATIVE_MISSED = 1

# Number of previous positions used for velocity estimation.
VELOCITY_HISTORY = 3

# Prevent two confirmed tracks from collapsing onto the same
# physical location.
TRACK_SEPARATION = 5.0

# Maximum number of active tracks.
#
# This is deliberately small because these sequences contain
# only a small number of physical ships.
MAX_ACTIVE_TRACKS = 5


# ============================================================
# PEAK DETECTION
# ============================================================

def detect_candidates(
    heatmap
):

    candidates = []

    height, width = heatmap.shape


    # --------------------------------------------------------
    # Find local maxima
    # --------------------------------------------------------

    for y in range(height):

        for x in range(width):

            score = float(
                heatmap[y, x]
            )

            if score < WEAK_THRESHOLD:
                continue


            y0 = max(
                0,
                y - MIN_DISTANCE
            )

            y1 = min(
                height,
                y + MIN_DISTANCE + 1
            )

            x0 = max(
                0,
                x - MIN_DISTANCE
            )

            x1 = min(
                width,
                x + MIN_DISTANCE + 1
            )


            neighbourhood = heatmap[
                y0:y1,
                x0:x1
            ]


            if score >= float(
                neighbourhood.max()
            ):

                candidates.append(
                    {
                        "score": score,
                        "x": float(x),
                        "y": float(y),
                        "strong": (
                            score >= STRONG_THRESHOLD
                        )
                    }
                )


    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Non-maximum suppression
    # --------------------------------------------------------

    selected = []

    for candidate in candidates:

        too_close = False


        for previous in selected:

            distance = np.sqrt(
                (
                    candidate["x"]
                    -
                    previous["x"]
                ) ** 2
                +
                (
                    candidate["y"]
                    -
                    previous["y"]
                ) ** 2
            )


            if distance < MIN_DISTANCE:

                too_close = True
                break


        if not too_close:

            selected.append(
                candidate
            )


        if len(selected) >= MAX_CANDIDATES:

            break


    return selected


# ============================================================
# TRACK CLASS
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        detection,
        bin_id
    ):

        self.track_id = track_id

        self.positions = [
            np.array(
                [
                    detection["x"],
                    detection["y"]
                ],
                dtype=np.float64
            )
        ]

        self.scores = [
            float(
                detection["score"]
            )
        ]

        self.bins = [
            int(bin_id)
        ]

        self.velocity = np.array(
            [0.0, 0.0],
            dtype=np.float64
        )

        self.missed = 0

        self.hits = 1

        self.confirmed = False

        self.age = 1

        self.weak_hits = 0

        self.strong_hits = int(
            detection["strong"]
        )


    # --------------------------------------------------------
    # Current position
    # --------------------------------------------------------

    @property
    def position(self):

        return self.positions[-1]


    # --------------------------------------------------------
    # Predict next position
    # --------------------------------------------------------

    def predict(
        self
    ):

        return (
            self.position
            +
            self.velocity
        )


    # --------------------------------------------------------
    # Estimate velocity
    # --------------------------------------------------------

    def estimate_velocity(
        self
    ):

        if len(
            self.positions
        ) < 2:

            return np.array(
                [0.0, 0.0],
                dtype=np.float64
            )


        history = self.positions[
            -VELOCITY_HISTORY:
        ]


        velocities = []

        for i in range(
            1,
            len(history)
        ):

            velocities.append(
                history[i]
                -
                history[i - 1]
            )


        if len(velocities) == 0:

            return np.array(
                [0.0, 0.0],
                dtype=np.float64
            )


        return np.mean(
            velocities,
            axis=0
        )


    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(
        self,
        detection,
        bin_id
    ):

        old_position = self.position.copy()


        new_position = np.array(
            [
                detection["x"],
                detection["y"]
            ],
            dtype=np.float64
        )


        # ----------------------------------------------------
        # Calculate instantaneous velocity
        # ----------------------------------------------------

        instantaneous_velocity = (
            new_position
            -
            old_position
        )


        # ----------------------------------------------------
        # Smooth velocity
        # ----------------------------------------------------

        self.velocity = (
            0.65 * self.velocity
            +
            0.35 * instantaneous_velocity
        )


        self.positions.append(
            new_position
        )

        self.scores.append(
            float(
                detection["score"]
            )
        )

        self.bins.append(
            int(bin_id)
        )


        self.hits += 1

        self.age += 1

        self.missed = 0


        if detection["strong"]:

            self.strong_hits += 1

        else:

            self.weak_hits += 1


        # ----------------------------------------------------
        # Confirm after repeated observations
        # ----------------------------------------------------

        if (
            self.hits >= CONFIRMATION_HITS
        ):

            self.confirmed = True


    # --------------------------------------------------------
    # Miss
    # --------------------------------------------------------

    def miss(
        self
    ):

        self.missed += 1

        self.age += 1


# ============================================================
# DISTANCE
# ============================================================

def distance(
    a,
    b
):

    return float(
        np.linalg.norm(
            np.asarray(a)
            -
            np.asarray(b)
        )
    )


# ============================================================
# MATCH COST
# ============================================================

def calculate_match_cost(
    track,
    detection
):

    predicted = track.predict()


    detection_position = np.array(
        [
            detection["x"],
            detection["y"]
        ],
        dtype=np.float64
    )


    position_error = distance(
        predicted,
        detection_position
    )


    # --------------------------------------------------------
    # Distance gate
    # --------------------------------------------------------

    if detection["strong"]:

        max_distance = STRONG_MAX_DISTANCE

    else:

        max_distance = WEAK_MAX_DISTANCE


    if position_error > max_distance:

        return None


    # --------------------------------------------------------
    # Confidence penalty
    #
    # Stronger detections get lower cost.
    # --------------------------------------------------------

    confidence_penalty = (
        1.0
        -
        detection["score"]
    ) * 8.0


    # --------------------------------------------------------
    # Motion consistency
    #
    # Compare the new displacement against previous velocity.
    # --------------------------------------------------------

    if len(
        track.positions
    ) >= 2:

        previous_position = (
            track.positions[-1]
        )

        displacement = (
            detection_position
            -
            previous_position
        )

        velocity_error = distance(
            track.velocity,
            displacement
        )

    else:

        velocity_error = 0.0


    motion_penalty = (
        velocity_error * 0.5
    )


    total_cost = (
        position_error
        +
        confidence_penalty
        +
        motion_penalty
    )


    return total_cost


# ============================================================
# ONE-TO-ONE ASSOCIATION
# ============================================================

def associate(
    tracks,
    detections
):

    possible_matches = []


    # --------------------------------------------------------
    # Build candidate matches
    # --------------------------------------------------------

    for track_index, track in enumerate(
        tracks
    ):

        for detection_index, detection in enumerate(
            detections
        ):

            cost = calculate_match_cost(
                track,
                detection
            )


            if cost is None:

                continue


            possible_matches.append(
                (
                    cost,
                    track_index,
                    detection_index
                )
            )


    # --------------------------------------------------------
    # Lowest cost first
    # --------------------------------------------------------

    possible_matches.sort(
        key=lambda item: item[0]
    )


    used_tracks = set()

    used_detections = set()

    matches = []


    # --------------------------------------------------------
    # One-to-one assignment
    # --------------------------------------------------------

    for (
        cost,
        track_index,
        detection_index
    ) in possible_matches:


        if track_index in used_tracks:

            continue


        if detection_index in used_detections:

            continue


        matches.append(
            (
                track_index,
                detection_index,
                cost
            )
        )


        used_tracks.add(
            track_index
        )

        used_detections.add(
            detection_index
        )


    unmatched_detections = [
        i
        for i in range(
            len(detections)
        )
        if i not in used_detections
    ]


    unmatched_tracks = [
        i
        for i in range(
            len(tracks)
        )
        if i not in used_tracks
    ]


    return (
        matches,
        unmatched_tracks,
        unmatched_detections
    )


# ============================================================
# CHECK DUPLICATE TRACK POSITION
# ============================================================

def near_existing_track(
    detection,
    tracks
):

    position = np.array(
        [
            detection["x"],
            detection["y"]
        ],
        dtype=np.float64
    )


    for track in tracks:

        predicted = track.predict()


        if distance(
            position,
            predicted
        ) < TRACK_SEPARATION:

            return True


    return False


# ============================================================
# TRACK QUALITY
# ============================================================

def track_quality(
    track
):

    if track.hits == 0:

        return 0.0


    hit_ratio = (
        track.hits
        /
        max(
            track.age,
            1
        )
    )


    mean_score = np.mean(
        track.scores
    )


    strong_ratio = (
        track.strong_hits
        /
        max(
            track.hits,
            1
        )
    )


    quality = (
        0.45 * hit_ratio
        +
        0.35 * mean_score
        +
        0.20 * strong_ratio
    )


    return float(
        quality
    )


# ============================================================
# RUN TRACKER
# ============================================================

def run_tracker(
    sequence_name,
    prediction_file
):

    print()
    print("=" * 70)
    print(
        f"V5 TRACKING: {sequence_name}"
    )
    print("=" * 70)


    heatmaps = np.load(
        prediction_file
    )


    print(
        "Heatmap shape:",
        heatmaps.shape
    )

    print(
        f"Strong threshold: "
        f"{STRONG_THRESHOLD}"
    )

    print(
        f"Weak threshold: "
        f"{WEAK_THRESHOLD}"
    )


    tracks = []

    next_track_id = 1


    # ========================================================
    # PROCESS TEMPORAL BINS
    # ========================================================

    for bin_id in range(
        len(heatmaps)
    ):


        # ----------------------------------------------------
        # Detect candidates
        # ----------------------------------------------------

        detections = detect_candidates(
            heatmaps[bin_id]
        )


        strong_count = sum(
            d["strong"]
            for d in detections
        )

        weak_count = (
            len(detections)
            -
            strong_count
        )


        # ----------------------------------------------------
        # Existing tracks → detections
        # ----------------------------------------------------

        matches, unmatched_tracks, unmatched_detections = (
            associate(
                tracks,
                detections
            )
        )


        # ----------------------------------------------------
        # Update matched tracks
        # ----------------------------------------------------

        matched_track_indices = set()


        for (
            track_index,
            detection_index,
            cost
        ) in matches:

            track = tracks[
                track_index
            ]

            detection = detections[
                detection_index
            ]


            track.update(
                detection,
                bin_id
            )


            matched_track_indices.add(
                track_index
            )


        # ----------------------------------------------------
        # Mark unmatched tracks as missed
        # ----------------------------------------------------

        for track_index in unmatched_tracks:

            tracks[
                track_index
            ].miss()


        # ----------------------------------------------------
        # Remove expired tracks
        # ----------------------------------------------------

        surviving_tracks = []


        for track in tracks:

            if track.confirmed:

                if track.missed <= MAX_MISSED:

                    surviving_tracks.append(
                        track
                    )

            else:

                if track.missed <= MAX_TENTATIVE_MISSED:

                    surviving_tracks.append(
                        track
                    )


        tracks = surviving_tracks


        # ----------------------------------------------------
        # Create new tracks
        #
        # IMPORTANT:
        #
        # Strong detections may start tracks.
        #
        # Weak detections can ALSO start tentative tracks,
        # but only if they aren't close to an existing track.
        #
        # Weak tracks must survive repeated observations before
        # becoming confirmed.
        # ----------------------------------------------------

        for detection_index in unmatched_detections:

            detection = detections[
                detection_index
            ]


            # ------------------------------------------------
            # Don't create tracks beyond our active limit.
            # ------------------------------------------------

            if len(tracks) >= MAX_ACTIVE_TRACKS:

                break


            # ------------------------------------------------
            # Don't create duplicate nearby tracks.
            # ------------------------------------------------

            if near_existing_track(
                detection,
                tracks
            ):

                continue


            # ------------------------------------------------
            # New track
            # ------------------------------------------------

            new_track = Track(
                next_track_id,
                detection,
                bin_id
            )


            tracks.append(
                new_track
            )


            next_track_id += 1


        # ----------------------------------------------------
        # Remove duplicate confirmed tracks
        # ----------------------------------------------------

        # Sort strongest / longest first.
        tracks.sort(
            key=lambda t: (
                t.confirmed,
                t.hits,
                np.mean(t.scores)
            ),
            reverse=True
        )


        filtered_tracks = []


        for track in tracks:

            duplicate = False


            for kept_track in filtered_tracks:

                if not (
                    track.confirmed
                    and
                    kept_track.confirmed
                ):

                    continue


                if distance(
                    track.position,
                    kept_track.position
                ) < TRACK_SEPARATION:

                    # Keep the stronger track.
                    duplicate = True
                    break


            if not duplicate:

                filtered_tracks.append(
                    track
                )


        tracks = filtered_tracks


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        confirmed_count = sum(
            t.confirmed
            for t in tracks
        )


        print(
            f"Bin {bin_id:02d}: "
            f"candidates={len(detections):2d} "
            f"(strong={strong_count:2d}, "
            f"weak={weak_count:2d}) | "
            f"active={len(tracks):2d} | "
            f"confirmed={confirmed_count:2d}"
        )


    # ========================================================
    # FINAL TRACK FILTER
    # ========================================================

    confirmed_tracks = [
        track
        for track in tracks
        if track.confirmed
    ]


    print()
    print(
        "Confirmed tracks:",
        len(confirmed_tracks)
    )


    # ========================================================
    # SAVE
    # ========================================================

    rows = []


    for track in confirmed_tracks:

        quality = track_quality(
            track
        )


        for i in range(
            len(track.positions)
        ):

            x, y = track.positions[i]


            rows.append(
                {
                    "track_id":
                        track.track_id,

                    "bin_id":
                        track.bins[i],

                    "x":
                        float(x),

                    "y":
                        float(y),

                    "score":
                        float(
                            track.scores[i]
                        ),

                    "track_quality":
                        quality
                }
            )


    output_df = pd.DataFrame(
        rows
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        f"{sequence_name}_tracks.csv"
    )


    output_df.to_csv(
        output_file,
        index=False
    )


    print()
    print(
        "Saved:"
    )

    print(
        output_file
    )


    # ========================================================
    # TRACK SUMMARY
    # ========================================================

    print()
    print(
        "Track summary:"
    )


    for track in confirmed_tracks:

        print(
            f"Track {track.track_id}: "
            f"hits={track.hits}, "
            f"bins={track.bins[0]}-"
            f"{track.bins[-1]}, "
            f"mean_score="
            f"{np.mean(track.scores):.3f}, "
            f"strong_hits="
            f"{track.strong_hits}, "
            f"weak_hits="
            f"{track.weak_hits}, "
            f"quality="
            f"{track_quality(track):.3f}"
        )


    return output_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_tracker(
        "ship_045",
        PRED_045
    )


    run_tracker(
        "ship_047",
        PRED_047
    )


    print()
    print("=" * 70)
    print(
        "V5 TRACKING COMPLETE"
    )
    print("=" * 70)