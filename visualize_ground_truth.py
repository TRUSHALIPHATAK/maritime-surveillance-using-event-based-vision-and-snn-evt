import os
import cv2

# ==========================================
# PATHS
# ==========================================

image_dir = r"E:\maritime\VISO\SOT\sot\ship\045\img"
gt_dir = r"E:\maritime\VISO\SOT\sot\ship\045\gt"

output_dir = r"data_samples\ground_truth_visualization"

os.makedirs(output_dir, exist_ok=True)


# ==========================================
# LOAD GROUND TRUTH FILES
# ==========================================

gt_files = [
    "1_1_320.txt",
    "2_1_320.txt",
    "3_1_319.txt"
]

ground_truth = []

for file_name in gt_files:

    file_path = os.path.join(gt_dir, file_name)

    boxes = []

    with open(file_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            x, y, w, h = map(int, line.split(","))

            boxes.append((x, y, w, h))

    ground_truth.append(boxes)

    print(f"{file_name}: {len(boxes)} annotations loaded")


# ==========================================
# PROCESS FIRST 20 FRAMES
# ==========================================

num_frames = 20

for frame_index in range(num_frames):

    image_name = f"{frame_index + 1:06d}.jpg"

    image_path = os.path.join(image_dir, image_name)

    image = cv2.imread(image_path)

    if image is None:

        print(f"Could not load {image_path}")

        continue


    # --------------------------------------
    # DRAW BOUNDING BOXES
    # --------------------------------------

    for object_index, boxes in enumerate(ground_truth):

        # Skip if annotation does not exist
        if frame_index >= len(boxes):
            continue

        x, y, w, h = boxes[frame_index]

        # Skip invalid bounding boxes
        if x == 0 and y == 0 and w == 0 and h == 0:

            continue


        # Draw bounding box
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Object label
        label = f"Ship {object_index + 1}"

        cv2.putText(
            image,
            label,
            (x, max(y - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # --------------------------------------
    # SAVE IMAGE
    # --------------------------------------

    output_path = os.path.join(
        output_dir,
        f"ground_truth_{frame_index + 1:02d}.jpg"
    )

    cv2.imwrite(output_path, image)

    print(f"Saved {output_path}")


print("\nGround truth visualization completed successfully!")