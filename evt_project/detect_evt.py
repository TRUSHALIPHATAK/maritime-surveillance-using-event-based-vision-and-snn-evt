import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEATMAP_DIR = os.path.join(PROJECT_ROOT, "evt_project", "results", "inference_v2")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evt_project", "results", "detections")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Detection parameters
CONF_THRESHOLD = 0.45
MIN_DISTANCE = 8
MAX_DETECTIONS = 10
LOCAL_MAX_KERNEL = 7

# ============================================================
# LOCAL PEAK DETECTION
# ============================================================

def find_local_peaks(heatmap):
    """
    Find multiple local maxima in an EVT heatmap.
    """

    heatmap_tensor = torch.from_numpy(
        heatmap.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    # Maximum value in local neighborhood
    pooled = F.max_pool2d(
        heatmap_tensor,
        kernel_size=LOCAL_MAX_KERNEL,
        stride=1,
        padding=LOCAL_MAX_KERNEL // 2
    )

    pooled = pooled.squeeze().numpy()

    # A pixel is a local maximum if it equals the local maximum
    peak_mask = (
        (heatmap >= pooled - 1e-6) &
        (heatmap >= CONF_THRESHOLD)
    )

    ys, xs = np.where(peak_mask)

    candidates = []

    for x, y in zip(xs, ys):
        candidates.append({
            "x": int(x),
            "y": int(y),
            "confidence": float(heatmap[y, x])
        })

    # Highest confidence first
    candidates.sort(
        key=lambda p: p["confidence"],
        reverse=True
    )

    # ========================================================
    # NMS / MINIMUM DISTANCE FILTER
    # ========================================================

    selected = []

    for candidate in candidates:

        too_close = False

        for selected_peak in selected:

            distance = np.sqrt(
                (candidate["x"] - selected_peak["x"]) ** 2 +
                (candidate["y"] - selected_peak["y"]) ** 2
            )

            if distance < MIN_DISTANCE:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

        if len(selected) >= MAX_DETECTIONS:
            break

    return selected


# ============================================================
# LOAD GT CENTERS
# ============================================================

def load_gt_centers(sequence_name):

    if sequence_name == "ship_045":
        gt_file = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "data",
            "ship_045_event_gt_labels.csv"
        )

    elif sequence_name == "ship_047":
        gt_file = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "data",
            "ship_047_event_gt_labels.csv"
        )

    else:
        raise ValueError(sequence_name)

    gt = pd.read_csv(gt_file)

    return gt


# ============================================================
# GET UNIQUE GT TARGET CENTERS FOR EACH GROUP
# ============================================================

def get_group_gt(gt, group_id):

    # Each EVT group contains 5 event bins
    start_bin = group_id * 5
    end_bin = start_bin + 5

    group = gt[
        (gt["bin_id"] >= start_bin) &
        (gt["bin_id"] < end_bin) &
        (gt["visible"] == 1)
    ]

    targets = []

    for target_id, target_data in group.groupby("target_id"):

        # Median prevents repeated rows from affecting the center
        x = target_data["x_128"].median()
        y = target_data["y_128"].median()

        targets.append({
            "target_id": int(target_id),
            "x": float(x),
            "y": float(y)
        })

    return targets


# ============================================================
# MAIN
# ============================================================

def process_sequence(sequence_name):

    heatmap_file = os.path.join(
        HEATMAP_DIR,
        f"{sequence_name}_evt_heatmaps_v2.npy"
    )

    heatmaps = np.load(heatmap_file)

    print("\n" + "=" * 60)
    print(f"PROCESSING {sequence_name}")
    print("=" * 60)

    print(f"Heatmap shape: {heatmaps.shape}")
    print(f"Heatmap range: {heatmaps.min():.4f} - {heatmaps.max():.4f}")

    gt = load_gt_centers(sequence_name)

    all_rows = []

    for group_id, heatmap in enumerate(heatmaps):

        detections = find_local_peaks(heatmap)
        gt_targets = get_group_gt(gt, group_id)

        print(f"\nGroup {group_id}")
        print(f"GT targets: {len(gt_targets)}")
        print(f"EVT peaks : {len(detections)}")

        # ----------------------------------------------------
        # PRINT GT
        # ----------------------------------------------------

        for target in gt_targets:

            print(
                f"  GT target {target['target_id']}: "
                f"({target['x']:.1f}, {target['y']:.1f})"
            )

        # ----------------------------------------------------
        # PRINT DETECTIONS
        # ----------------------------------------------------

        for i, detection in enumerate(detections):

            print(
                f"  EVT peak {i+1}: "
                f"({detection['x']}, {detection['y']}) "
                f"conf={detection['confidence']:.4f}"
            )

        # ----------------------------------------------------
        # SAVE DETECTIONS
        # ----------------------------------------------------

        for rank, detection in enumerate(detections):

            all_rows.append({
                "sequence": sequence_name,
                "group_id": group_id,
                "detection_rank": rank + 1,
                "x": detection["x"],
                "y": detection["y"],
                "confidence": detection["confidence"]
            })

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_file = os.path.join(
        RESULTS_DIR,
        f"{sequence_name}_evt_candidates_v2.csv"
    )

    df = pd.DataFrame(all_rows)

    df.to_csv(output_file, index=False)

    print("\n" + "-" * 60)
    print(f"Saved: {output_file}")
    print(f"Total candidate detections: {len(df)}")
    print("-" * 60)


# ============================================================
# RUN BOTH SEQUENCES
# ============================================================

if __name__ == "__main__":

    process_sequence("ship_045")
    process_sequence("ship_047")

    print("\nDetection candidate generation complete.")