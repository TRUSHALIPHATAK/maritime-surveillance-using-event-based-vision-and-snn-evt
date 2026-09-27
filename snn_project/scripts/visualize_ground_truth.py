import cv2
import os


# ----------------------------------------
# PATHS
# ----------------------------------------

image_dir = (
    "E:/maritime/VISO/SOT/sot/ship/045/img"
)

gt_dir = (
    "E:/maritime/VISO/SOT/sot/ship/045/gt"
)


output_dir = (
    "snn_project/results/"
    "ground_truth_visualization"
)


os.makedirs(
    output_dir,
    exist_ok=True
)


# ----------------------------------------
# LOAD GROUND TRUTH FILES
# ----------------------------------------

gt_files = [
    "1_1_320.txt",
    "2_1_320.txt",
    "3_1_319.txt"
]


ground_truth = []


for filename in gt_files:

    path = os.path.join(
        gt_dir,
        filename
    )

    with open(path, "r") as f:

        lines = f.readlines()


    boxes = []

    for line in lines:

        values = line.strip().split(",")


        if len(values) == 4:

            x, y, w, h = map(
                int,
                values
            )

            boxes.append(
                (x, y, w, h)
            )


    ground_truth.append(
        boxes
    )


print("Ground truth loaded!")

for i, boxes in enumerate(ground_truth):

    print(
        f"Target {i + 1}: "
        f"{len(boxes)} entries"
    )


# ----------------------------------------
# VISUALIZE SELECTED FRAMES
# ----------------------------------------

frames_to_visualize = [
    0,
    50,
    100,
    150,
    200,
    250,
    319
]


for frame_index in frames_to_visualize:


    # Dataset numbering starts from 1

    image_name = (
        f"{frame_index + 1:06d}.jpg"
    )


    image_path = os.path.join(
        image_dir,
        image_name
    )


    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "Could not load:",
            image_path
        )

        continue


    # ------------------------------------
    # DRAW BOUNDING BOXES
    # ------------------------------------

    for target_index in range(3):


        boxes = ground_truth[
            target_index
        ]


        # Skip if GT does not have
        # this frame

        if frame_index >= len(boxes):

            continue


        x, y, w, h = boxes[
            frame_index
        ]


        # Skip invalid annotation

        if (
            x == 0
            and y == 0
            and w == 0
            and h == 0
        ):

            continue


        # Draw rectangle

        cv2.rectangle(

            image,

            (x, y),

            (x + w, y + h),

            (255, 255, 255),

            2

        )


        # Add target label

        cv2.putText(

            image,

            f"Target {target_index + 1}",

            (x, y - 5),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2

        )


    # ------------------------------------
    # SAVE IMAGE
    # ------------------------------------

    output_path = os.path.join(

        output_dir,

        f"frame_{frame_index + 1:03d}_gt.png"

    )


    cv2.imwrite(

        output_path,

        image

    )


    print(

        f"Frame {frame_index + 1} saved"

    )


print("\nVisualization completed!")

print(

    "Results saved to:"

)

print(output_dir)