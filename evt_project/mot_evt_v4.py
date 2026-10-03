import os
import csv
import math
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCES = ["045", "047"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_DIR = os.path.join(
    BASE_DIR, "results", "detections"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "mot_evt_v4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


CONF_THRESHOLD = 0.45
NEW_TRACK_CONFIDENCE = 0.55

MAX_ASSOCIATION_DISTANCE = 18.0
STRONG_ASSOCIATION_DISTANCE = 10.0

MAX_MOVEMENT_PER_GROUP = 20.0
MAX_MISSED_GROUPS = 3

CONFIRMATION_HITS = 2
MAX_TRACKS = 5

VELOCITY_ALPHA = 0.65

BORDER_MARGIN = 3

GT_MATCH_DISTANCE = 10.0


# ============================================================
# TRACK CLASS
# ============================================================

class Track:

    def __init__(self, track_id, x, y, confidence, group_id):

        self.id = track_id

        self.x = float(x)
        self.y = float(y)

        self.prev_x = float(x)
        self.prev_y = float(y)

        self.vx = 0.0
        self.vy = 0.0

        self.confidence = float(confidence)

        self.hits = 1
        self.missed = 0

        self.confirmed = False

        # Important for V4:
        # remember where the track started
        self.first_group = group_id


    def predict(self):

        return (
            self.x + self.vx,
            self.y + self.vy
        )


    def update(self, x, y, confidence):

        new_x = float(x)
        new_y = float(y)

        dx = new_x - self.x
        dy = new_y - self.y

        # Update velocity
        self.vx = (
            VELOCITY_ALPHA * dx
            + (1.0 - VELOCITY_ALPHA) * self.vx
        )

        self.vy = (
            VELOCITY_ALPHA * dy
            + (1.0 - VELOCITY_ALPHA) * self.vy
        )

        self.prev_x = self.x
        self.prev_y = self.y

        self.x = new_x
        self.y = new_y

        self.confidence = float(confidence)

        self.hits += 1
        self.missed = 0

        if self.hits >= CONFIRMATION_HITS:
            self.confirmed = True


    def mark_missed(self):

        self.missed += 1

        # Continue with velocity prediction
        self.x += self.vx
        self.y += self.vy


# ============================================================
# LOAD CANDIDATES
# ============================================================

def load_candidates(sequence):

    path = os.path.join(
        CANDIDATE_DIR,
        f"ship_{sequence}_evt_candidates_v2.csv"
    )

    candidates = {}

    with open(path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            group_id = int(row["group_id"])

            x = float(row["x"])
            y = float(row["y"])
            confidence = float(row["confidence"])

            if confidence < CONF_THRESHOLD:
                continue

            candidates.setdefault(group_id, []).append(
                {
                    "x": x,
                    "y": y,
                    "confidence": confidence
                }
            )

    return candidates


# ============================================================
# BORDER FILTER
# ============================================================

def is_border_candidate(x, y):

    if x <= BORDER_MARGIN:
        return True

    if y <= BORDER_MARGIN:
        return True

    if x <= BORDER_MARGIN and y >= 122:
        return True

    if x >= 124 and y >= 124:
        return True

    return False


def valid_new_track_candidate(candidate):

    x = candidate["x"]
    y = candidate["y"]
    confidence = candidate["confidence"]

    if confidence < NEW_TRACK_CONFIDENCE:
        return False

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
# DISTANCE
# ============================================================

def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x1 - x2) ** 2
        + (y1 - y2) ** 2
    )


# ============================================================
# GT LOADING
# ============================================================

def load_ground_truth(sequence):

    if sequence == "045":

        path = os.path.join(
            BASE_DIR,
            "..",
            "snn_project",
            "data",
            "ship_045_event_gt_labels.csv"
        )

    elif sequence == "047":

                path = os.path.join(
                    BASE_DIR,
                    "..",
                    "snn_project",
                    "data",
                    "ship_047_event_gt_labels.csv"
                )

    else:

        raise ValueError(
            f"Unknown sequence: {sequence}"
        )


    gt = {}

    with open(path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            visible = int(row["visible"])

            if visible == 0:
                continue

            group_id = int(row["bin_id"]) // 5

            target_id = int(row["target_id"])

            x = float(row["x_128"])
            y = float(row["y_128"])

            gt.setdefault(group_id, {})
            
            gt[group_id].setdefault(
                target_id,
                []
            )

            gt[group_id][target_id].append(
                (x, y)
            )


    # Median center for each target within each temporal group
    final_gt = {}

    for group_id, targets in gt.items():

        final_gt[group_id] = []

        for target_id, points in targets.items():

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            final_gt[group_id].append(
                {
                    "target_id": target_id,
                    "x": float(np.median(xs)),
                    "y": float(np.median(ys))
                }
            )

    return final_gt


# ============================================================
# TRACK ASSOCIATION
# ============================================================

def associate_tracks(tracks, candidates):

    if len(tracks) == 0 or len(candidates) == 0:
        return [], list(range(len(tracks))), list(range(len(candidates)))


    possible = []

    for ti, track in enumerate(tracks):

        px, py = track.predict()

        for ci, candidate in enumerate(candidates):

            d = distance(
                px,
                py,
                candidate["x"],
                candidate["y"]
            )

            if d <= MAX_ASSOCIATION_DISTANCE:

                score = (
                    d
                    - 1.5 * candidate["confidence"]
                )

                possible.append(
                    (
                        score,
                        d,
                        ti,
                        ci
                    )
                )


    possible.sort(key=lambda x: x[0])

    matched_tracks = set()
    matched_candidates = set()

    matches = []

    for score, d, ti, ci in possible:

        if ti in matched_tracks:
            continue

        if ci in matched_candidates:
            continue

        # Additional movement protection
        if d > MAX_MOVEMENT_PER_GROUP:
            continue

        matches.append(
            (ti, ci, d)
        )

        matched_tracks.add(ti)
        matched_candidates.add(ci)


    unmatched_tracks = [
        i for i in range(len(tracks))
        if i not in matched_tracks
    ]

    unmatched_candidates = [
        i for i in range(len(candidates))
        if i not in matched_candidates
    ]

    return (
        matches,
        unmatched_tracks,
        unmatched_candidates
    )


# ============================================================
# TRACKING
# ============================================================

def run_tracking(sequence):

    print("\n" + "=" * 70)
    print(f"EVT MOT V4 — SEQUENCE {sequence}")
    print("=" * 70)


    candidates_by_group = load_candidates(sequence)
    gt_by_group = load_ground_truth(sequence)


    tracks = []

    next_track_id = 1

    output_rows = []


    # --------------------------------------------------------
    # Process all 10 temporal groups
    # --------------------------------------------------------

    for group_id in range(10):

        candidates = candidates_by_group.get(
            group_id,
            []
        )


        # ====================================================
        # 1. ASSOCIATE EXISTING TRACKS
        # ====================================================

        matches, unmatched_tracks, unmatched_candidates = (
            associate_tracks(
                tracks,
                candidates
            )
        )


        # ====================================================
        # 2. UPDATE MATCHED TRACKS
        # ====================================================

        for ti, ci, d in matches:

            track = tracks[ti]

            candidate = candidates[ci]

            track.update(
                candidate["x"],
                candidate["y"],
                candidate["confidence"]
            )


        # ====================================================
        # 3. HANDLE MISSED TRACKS
        # ====================================================

        for ti in unmatched_tracks:

            track = tracks[ti]

            track.mark_missed()


        # ====================================================
        # 4. REMOVE DEAD TRACKS
        # ====================================================

        tracks = [
            t for t in tracks
            if t.missed <= MAX_MISSED_GROUPS
        ]


        # ====================================================
        # 5. CREATE NEW TRACKS
        # ====================================================

        for ci in unmatched_candidates:

            candidate = candidates[ci]

            if len(tracks) >= MAX_TRACKS:
                break

            if not valid_new_track_candidate(candidate):
                continue


            # Don't create another track too close to an
            # existing track

            too_close = False

            for track in tracks:

                d = distance(
                    track.x,
                    track.y,
                    candidate["x"],
                    candidate["y"]
                )

                if d < 7:

                    too_close = True
                    break


            if too_close:
                continue


            new_track = Track(
                next_track_id,
                candidate["x"],
                candidate["y"],
                candidate["confidence"],
                group_id
            )

            tracks.append(new_track)

            next_track_id += 1


        # ====================================================
        # 6. SAVE OUTPUT
        # ====================================================

        group_output = []

        for track in tracks:

            if track.missed > 0:
                continue


            # ------------------------------------------------
            # V4 CHANGE
            #
            # Confirmed tracks are always saved.
            #
            # Tentative tracks are saved ONLY when they are
            # first created in GROUP 0.
            #
            # This recovers the valid initial detections while
            # avoiding later false tentative tracks.
            # ------------------------------------------------

            if not track.confirmed:

                if group_id != 0:
                    continue


            group_output.append(
                {
                    "group_id": group_id,
                    "track_id": track.id,
                    "x": track.x,
                    "y": track.y,
                    "confidence": track.confidence,
                    "confirmed": int(track.confirmed),
                    "status":
                        "confirmed"
                        if track.confirmed
                        else "initial"
                }
            )


            output_rows.append(
                group_output[-1]
            )


        # ====================================================
        # PRINT GROUP RESULT
        # ====================================================

        print(
            f"\nGroup {group_id}:"
        )

        for row in group_output:

            print(
                f"  Track {row['track_id']}: "
                f"({row['x']:.1f}, {row['y']:.1f}) "
                f"conf={row['confidence']:.3f} "
                f"[{row['status']}]"
            )


    # ========================================================
    # SAVE TRACK CSV
    # ========================================================

    output_csv = os.path.join(
        OUTPUT_DIR,
        f"ship_{sequence}_evt_tracks_v4.csv"
    )


    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "group_id",
            "track_id",
            "x",
            "y",
            "confidence",
            "confirmed",
            "status"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(output_rows)


    print(
        f"\nSaved tracks:"
        f"\n{output_csv}"
    )


    # ========================================================
    # METRICS
    # ========================================================

    tp = 0
    fp = 0
    fn = 0

    errors = []

    matches_for_group = {}


    for group_id in range(10):

        gt_points = gt_by_group.get(
            group_id,
            []
        )

        predictions = [
            r for r in output_rows
            if r["group_id"] == group_id
        ]


        gt_used = set()
        pred_used = set()

        group_matches = []


        # ----------------------------------------------------
        # Greedy nearest-neighbour matching
        # ----------------------------------------------------

        possible = []

        for pi, pred in enumerate(predictions):

            for gi, gt in enumerate(gt_points):

                d = distance(
                    pred["x"],
                    pred["y"],
                    gt["x"],
                    gt["y"]
                )

                if d <= GT_MATCH_DISTANCE:

                    possible.append(
                        (
                            d,
                            pi,
                            gi
                        )
                    )


        possible.sort(
            key=lambda x: x[0]
        )


        for d, pi, gi in possible:

            if pi in pred_used:
                continue

            if gi in gt_used:
                continue

            pred_used.add(pi)
            gt_used.add(gi)

            tp += 1
            errors.append(d)

            group_matches.append(
                (
                    predictions[pi],
                    gt_points[gi],
                    d
                )
            )


        group_fp = len(predictions) - len(pred_used)
        group_fn = len(gt_points) - len(gt_used)

        fp += group_fp
        fn += group_fn

        matches_for_group[group_id] = group_matches


    # ========================================================
    # FINAL METRICS
    # ========================================================

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )


    if errors:

        mean_error = float(
            np.mean(errors)
        )

        median_error = float(
            np.median(errors)
        )

        max_error = float(
            np.max(errors)
        )

        within_2 = (
            100.0
            * np.mean(
                np.array(errors) <= 2
            )
        )

        within_4 = (
            100.0
            * np.mean(
                np.array(errors) <= 4
            )
        )

        within_6 = (
            100.0
            * np.mean(
                np.array(errors) <= 6
            )
        )

    else:

        mean_error = 0
        median_error = 0
        max_error = 0
        within_2 = 0
        within_4 = 0
        within_6 = 0


    total_gt = tp + fn

    coverage = (
        tp / total_gt
        if total_gt > 0
        else 0
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\n" + "-" * 70)
    print(f"FINAL EVT MOT V4 METRICS — {sequence}")
    print("-" * 70)

    print(f"TP                 : {tp}")
    print(f"FP                 : {fp}")
    print(f"FN                 : {fn}")

    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1                 : {f1:.4f}")

    print(f"Mean center error  : {mean_error:.3f} px")
    print(f"Median error       : {median_error:.3f} px")
    print(f"Max error          : {max_error:.3f} px")

    print(f"Within 2 px        : {within_2:.2f}%")
    print(f"Within 4 px        : {within_4:.2f}%")
    print(f"Within 6 px        : {within_6:.2f}%")

    print(f"Coverage           : {coverage:.4f}")

    print("-" * 70)


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_csv = os.path.join(
        OUTPUT_DIR,
        f"ship_{sequence}_evt_metrics_v4.csv"
    )


    with open(
        metrics_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sequence",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "F1",
                "mean_center_error",
                "median_center_error",
                "max_center_error",
                "within_2px_percent",
                "within_4px_percent",
                "within_6px_percent",
                "coverage"
            ]
        )

        writer.writerow(
            [
                sequence,
                tp,
                fp,
                fn,
                precision,
                recall,
                f1,
                mean_error,
                median_error,
                max_error,
                within_2,
                within_4,
                within_6,
                coverage
            ]
        )


    print(
        f"Saved metrics:"
        f"\n{metrics_csv}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for sequence in SEQUENCES:

        run_tracking(sequence)

    print("\n" + "=" * 70)
    print("EVT MOT V4 COMPLETE")
    print("=" * 70)