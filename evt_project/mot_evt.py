import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "detections"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "mot_evt_v4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Detection
CONF_THRESHOLD = 0.45
NEW_TRACK_CONFIDENCE = 0.55

# Association
MAX_ASSOCIATION_DISTANCE = 18.0
STRONG_ASSOCIATION_DISTANCE = 10.0
MAX_MOVEMENT_PER_GROUP = 20.0

# Tracking
MAX_MISSED_GROUPS = 3
CONFIRMATION_HITS = 2
MAX_TRACKS = 5

# Velocity
VELOCITY_ALPHA = 0.65

# Border protection
BORDER_MARGIN = 3

# GT matching
GT_MATCH_DISTANCE = 10.0


# ============================================================
# TRACK CLASS
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        x,
        y,
        confidence,
        group_id
    ):

        self.track_id = track_id

        self.x = float(x)
        self.y = float(y)

        self.vx = 0.0
        self.vy = 0.0

        self.confidence = float(confidence)

        self.hits = 1
        self.missed = 0

        self.first_group = group_id
        self.last_group = group_id

        self.confirmed = False


    def predict(self):

        return (
            self.x + self.vx,
            self.y + self.vy
        )


    def update(
        self,
        x,
        y,
        confidence,
        group_id
    ):

        x = float(x)
        y = float(y)

        instant_vx = x - self.x
        instant_vy = y - self.y

        self.vx = (
            VELOCITY_ALPHA * instant_vx
            + (1.0 - VELOCITY_ALPHA) * self.vx
        )

        self.vy = (
            VELOCITY_ALPHA * instant_vy
            + (1.0 - VELOCITY_ALPHA) * self.vy
        )

        self.x = x
        self.y = y

        self.confidence = float(confidence)

        self.hits += 1
        self.missed = 0
        self.last_group = group_id

        if self.hits >= CONFIRMATION_HITS:
            self.confirmed = True


    def mark_missed(self):

        self.x += self.vx
        self.y += self.vy

        self.missed += 1


# ============================================================
# LOAD CANDIDATES
# ============================================================

def load_candidates(sequence_name):

    path = os.path.join(
        INPUT_DIR,
        f"ship_{sequence_name}_evt_candidates_v2.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Candidate file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "group_id",
        "x",
        "y",
        "confidence"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column '{column}' in {path}"
            )

    return df


# ============================================================
# BORDER FILTER
# ============================================================

def filter_border_candidates(candidates):

    filtered = []

    for candidate in candidates:

        x = float(candidate["x"])
        y = float(candidate["y"])

        corner_artifact = (

            (
                x <= BORDER_MARGIN
                and y <= BORDER_MARGIN
            )

            or

            (
                x <= BORDER_MARGIN
                and y >= 127 - BORDER_MARGIN
            )

            or

            (
                x >= 127 - BORDER_MARGIN
                and y >= 127 - BORDER_MARGIN
            )
        )

        if corner_artifact:
            continue

        filtered.append(candidate)

    return filtered


# ============================================================
# NEW TRACK VALIDATION
# ============================================================

def valid_new_track_candidate(x, y):

    if x <= BORDER_MARGIN:
        return False

    if y <= BORDER_MARGIN:
        return False

    if x >= 124:
        return False

    if y >= 124:
        return False

    return True


# ============================================================
# ASSOCIATION
# ============================================================

def associate_tracks(
    tracks,
    candidates
):

    matches = []

    unmatched_tracks = set(
        range(len(tracks))
    )

    unmatched_candidates = set(
        range(len(candidates))
    )

    if not tracks or not candidates:

        return (
            matches,
            unmatched_tracks,
            unmatched_candidates
        )


    associations = []


    for ti, track in enumerate(tracks):

        px, py = track.predict()

        for ci, candidate in enumerate(candidates):

            x = float(candidate["x"])
            y = float(candidate["y"])
            confidence = float(
                candidate["confidence"]
            )

            distance = np.sqrt(
                (px - x) ** 2
                + (py - y) ** 2
            )

            if track.confirmed:

                max_distance = (
                    MAX_ASSOCIATION_DISTANCE
                )

            else:

                max_distance = (
                    MAX_ASSOCIATION_DISTANCE - 3
                )

            if distance > max_distance:
                continue

            if distance > MAX_MOVEMENT_PER_GROUP:
                continue

            score = (
                distance
                - 1.5 * confidence
            )

            if distance <= STRONG_ASSOCIATION_DISTANCE:

                score -= 1.0

            associations.append(
                (
                    score,
                    distance,
                    ti,
                    ci
                )
            )


    associations.sort(
        key=lambda item: item[0]
    )


    for (
        score,
        distance,
        ti,
        ci
    ) in associations:

        if ti not in unmatched_tracks:
            continue

        if ci not in unmatched_candidates:
            continue

        matches.append(
            (
                ti,
                ci,
                distance
            )
        )

        unmatched_tracks.remove(ti)
        unmatched_candidates.remove(ci)


    return (
        matches,
        unmatched_tracks,
        unmatched_candidates
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(sequence_name):

    gt_path = os.path.join(
        BASE_DIR,
        "..",
        "snn_project",
        "data",
        f"ship_{sequence_name}_event_gt_labels.csv"
    )

    gt_path = os.path.abspath(gt_path)

    if not os.path.exists(gt_path):

        raise FileNotFoundError(
            f"GT file not found:\n{gt_path}"
        )

    return pd.read_csv(gt_path)


# ============================================================
# GT CENTERS
# ============================================================

def get_gt_centers(
    gt,
    group_id
):

    group = gt[
        gt["bin_id"].between(
            group_id * 5,
            group_id * 5 + 4
        )
    ]

    if len(group) == 0:
        return []

    visible = group[
        group["visible"] == 1
    ]

    if len(visible) == 0:
        return []

    centers = []


    for target_id in visible["target_id"].unique():

        target = visible[
            visible["target_id"] == target_id
        ]

        if len(target) == 0:
            continue

        x = target["x_128"].median()
        y = target["y_128"].median()

        centers.append(
            (
                float(x),
                float(y),
                int(target_id)
            )
        )

    return centers


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    track_rows,
    gt,
    sequence_name
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    errors = []

    gt_visible_total = 0


    for group_id in range(10):

        gt_centers = get_gt_centers(
            gt,
            group_id
        )

        gt_visible_total += len(
            gt_centers
        )

        predictions = [
            row
            for row in track_rows
            if row["group_id"] == group_id
        ]

        matched_gt = set()


        for prediction in predictions:

            px = prediction["x"]
            py = prediction["y"]

            best_distance = float("inf")
            best_index = None


            for gi, gt_point in enumerate(
                gt_centers
            ):

                if gi in matched_gt:
                    continue

                gx, gy, target_id = gt_point

                distance = np.sqrt(
                    (px - gx) ** 2
                    + (py - gy) ** 2
                )

                if distance < best_distance:

                    best_distance = distance
                    best_index = gi


            if (
                best_index is not None
                and best_distance <= GT_MATCH_DISTANCE
            ):

                total_tp += 1

                matched_gt.add(
                    best_index
                )

                errors.append(
                    best_distance
                )

            else:

                total_fp += 1


        total_fn += (
            len(gt_centers)
            - len(matched_gt)
        )


    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


    if errors:

        errors_np = np.array(
            errors,
            dtype=float
        )

        mean_error = float(
            np.mean(errors_np)
        )

        median_error = float(
            np.median(errors_np)
        )

        max_error = float(
            np.max(errors_np)
        )

        within_2 = float(
            np.mean(errors_np <= 2)
            * 100
        )

        within_4 = float(
            np.mean(errors_np <= 4)
            * 100
        )

        within_6 = float(
            np.mean(errors_np <= 6)
            * 100
        )

    else:

        mean_error = 0.0
        median_error = 0.0
        max_error = 0.0

        within_2 = 0.0
        within_4 = 0.0
        within_6 = 0.0


    coverage = (
        total_tp / gt_visible_total
        if gt_visible_total > 0
        else 0.0
    )


    # --------------------------------------------------------
    # ID switches
    # --------------------------------------------------------

    id_switches = 0
    previous_assignment = {}


    for group_id in range(10):

        gt_centers = get_gt_centers(
            gt,
            group_id
        )

        predictions = [
            row
            for row in track_rows
            if row["group_id"] == group_id
        ]

        current_assignment = {}


        for prediction in predictions:

            px = prediction["x"]
            py = prediction["y"]

            best_distance = float("inf")
            best_target = None


            for gx, gy, target_id in gt_centers:

                distance = np.sqrt(
                    (px - gx) ** 2
                    + (py - gy) ** 2
                )

                if distance < best_distance:

                    best_distance = distance
                    best_target = target_id


            if (
                best_target is not None
                and best_distance <= GT_MATCH_DISTANCE
            ):

                current_assignment[
                    best_target
                ] = prediction["track_id"]


        for target_id, track_id in (
            current_assignment.items()
        ):

            if target_id in previous_assignment:

                if (
                    previous_assignment[target_id]
                    != track_id
                ):

                    id_switches += 1


        previous_assignment = (
            current_assignment
        )


    return {
        "sequence": sequence_name,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_center_error": mean_error,
        "median_center_error": median_error,
        "max_center_error": max_error,
        "within_2px_percent": within_2,
        "within_4px_percent": within_4,
        "within_6px_percent": within_6,
        "coverage": coverage,
        "ID_switches": id_switches
    }


# ============================================================
# RUN SEQUENCE
# ============================================================

def run_sequence(sequence_name):

    print()
    print("=" * 70)
    print(
        f"EVT MOT V4: ship_{sequence_name}"
    )
    print("=" * 70)


    candidates_df = load_candidates(
        sequence_name
    )

    gt = load_ground_truth(
        sequence_name
    )


    tracks = []
    next_track_id = 1

    track_rows = []


    # ========================================================
    # 10 TEMPORAL GROUPS
    # ========================================================

    for group_id in range(10):

        print()
        print(f"Group {group_id}")


        group_df = candidates_df[
            candidates_df["group_id"] == group_id
        ]


        candidates = []


        for _, row in group_df.iterrows():

            confidence = float(
                row["confidence"]
            )

            if confidence < CONF_THRESHOLD:
                continue

            candidates.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "confidence": confidence
                }
            )


        candidates = filter_border_candidates(
            candidates
        )


        # ----------------------------------------------------
        # Associate
        # ----------------------------------------------------

        (
            matches,
            unmatched_track_indices,
            unmatched_candidate_indices
        ) = associate_tracks(
            tracks,
            candidates
        )


        # ----------------------------------------------------
        # Update matched tracks
        # ----------------------------------------------------

        for ti, ci, distance in matches:

            track = tracks[ti]
            candidate = candidates[ci]


            if track.confirmed:

                track.update(
                    candidate["x"],
                    candidate["y"],
                    candidate["confidence"],
                    group_id
                )

            else:

                if (
                    candidate["confidence"]
                    >= NEW_TRACK_CONFIDENCE
                ):

                    track.update(
                        candidate["x"],
                        candidate["y"],
                        candidate["confidence"],
                        group_id
                    )

                else:

                    track.mark_missed()


        # ----------------------------------------------------
        # Handle unmatched tracks
        # ----------------------------------------------------

        for ti in unmatched_track_indices:

            tracks[ti].mark_missed()


        # ----------------------------------------------------
        # Remove dead tracks
        # ----------------------------------------------------

        tracks = [
            track
            for track in tracks
            if track.missed <= MAX_MISSED_GROUPS
        ]


        # ----------------------------------------------------
        # Create new tracks
        # ----------------------------------------------------

        for ci in unmatched_candidate_indices:

            candidate = candidates[ci]

            x = candidate["x"]
            y = candidate["y"]
            confidence = candidate["confidence"]


            if confidence < NEW_TRACK_CONFIDENCE:
                continue

            if not valid_new_track_candidate(
                x,
                y
            ):
                continue


            too_close = False


            for track in tracks:

                distance = np.sqrt(
                    (track.x - x) ** 2
                    + (track.y - y) ** 2
                )

                if distance < 8:

                    too_close = True
                    break


            if too_close:
                continue

            if len(tracks) >= MAX_TRACKS:
                continue


            tracks.append(
                Track(
                    next_track_id,
                    x,
                    y,
                    confidence,
                    group_id
                )
            )

            next_track_id += 1


        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        for track in tracks:

            if track.missed > 0:
                continue


            # V4:
            # Save confirmed tracks normally.
            #
            # Save Group 0 tentative tracks as INITIAL.
            #
            # Do not save later tentative tracks.

            if track.confirmed:

                status = "CONFIRMED"

            else:

                if group_id != 0:
                    continue

                status = "INITIAL"


            track_rows.append(
                {
                    "group_id": group_id,
                    "track_id": track.track_id,
                    "x": track.x,
                    "y": track.y,
                    "confidence": track.confidence,
                    "hits": track.hits,
                    "missed": track.missed,
                    "status": status
                }
            )


            print(
                f"  Track {track.track_id}: "
                f"({track.x:.1f}, {track.y:.1f}) "
                f"conf={track.confidence:.3f} "
                f"hits={track.hits} "
                f"missed={track.missed} "
                f"{status}"
            )


    # ========================================================
    # SAVE TRACKS
    # ========================================================

    tracks_path = os.path.join(
        OUTPUT_DIR,
        f"ship_{sequence_name}_evt_mot_v4_tracks.csv"
    )

    pd.DataFrame(
        track_rows
    ).to_csv(
        tracks_path,
        index=False
    )


    print()
    print("Tracking file:")
    print(tracks_path)


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        track_rows,
        gt,
        sequence_name
    )


    print()
    print("=" * 70)
    print(
        f"EVT MOT V4 METRICS: "
        f"ship_{sequence_name}"
    )
    print("=" * 70)

    print(
        f"TP                  : "
        f"{metrics['TP']}"
    )

    print(
        f"FP                  : "
        f"{metrics['FP']}"
    )

    print(
        f"FN                  : "
        f"{metrics['FN']}"
    )

    print(
        f"\nPrecision           : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall              : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                  : "
        f"{metrics['F1']:.4f}"
    )

    print(
        f"\nMean center error   : "
        f"{metrics['mean_center_error']:.3f} px"
    )

    print(
        f"Median center error : "
        f"{metrics['median_center_error']:.3f} px"
    )

    print(
        f"Max center error    : "
        f"{metrics['max_center_error']:.3f} px"
    )

    print(
        f"\nWithin 2 px         : "
        f"{metrics['within_2px_percent']:.2f}%"
    )

    print(
        f"Within 4 px         : "
        f"{metrics['within_4px_percent']:.2f}%"
    )

    print(
        f"Within 6 px         : "
        f"{metrics['within_6px_percent']:.2f}%"
    )

    print(
        f"\nCoverage            : "
        f"{metrics['coverage']:.4f}"
    )

    print(
        f"ID switches         : "
        f"{metrics['ID_switches']}"
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"ship_{sequence_name}_evt_mot_v4_metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False
    )


    print()
    print("Metrics saved:")
    print(metrics_path)


    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    all_metrics = []


    for sequence_name in [
        "045",
        "047"
    ]:

        metrics = run_sequence(
            sequence_name
        )

        all_metrics.append(
            metrics
        )


    final_path = os.path.join(
        OUTPUT_DIR,
        "evt_mot_v4_final_metrics.csv"
    )

    pd.DataFrame(
        all_metrics
    ).to_csv(
        final_path,
        index=False
    )


    print()
    print("=" * 70)
    print("EVT MOT V4 COMPLETE")
    print("=" * 70)

    print()
    print("Final metrics:")
    print(final_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()