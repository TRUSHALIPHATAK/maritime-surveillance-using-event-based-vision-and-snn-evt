import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from data.evt_dataset import EVTEventDataset


datasets = {

    "045": {
        "voxel":
        "../../ship_045_event_dataset/3_voxel_grid/ship_045_voxel_grid.npy",

        "gt":
        "../../snn_project/data/ship_045_event_gt_labels.csv",
    },

    "047": {
        "voxel":
        "../../snn_project/ship_047_event_dataset/3_voxel_grid/ship_047_voxel_grid.npy",

        "gt":
        "../../snn_project/data/ship_047_event_gt_labels.csv",
    }
}


for name, paths in datasets.items():

    print("\n")
    print("=" * 70)
    print("TESTING SHIP", name)
    print("=" * 70)

    dataset = EVTEventDataset(
        voxel_path=paths["voxel"],
        gt_path=paths["gt"],
        bins_per_group=5,
        sigma=3.0,
    )

    print("Dataset length:", len(dataset))

    for idx in [0, len(dataset) - 1]:

        events, heatmap, group_id = dataset[idx]

        print("\nGroup:", idx)
        print("Events :", events.shape)
        print("Heatmap:", heatmap.shape)
        print("Group ID:", group_id.item())

        print(
            "Events min/max:",
            events.min().item(),
            events.max().item()
        )

        print(
            "Heatmap min/max:",
            heatmap.min().item(),
            heatmap.max().item()
        )

        max_location = heatmap.argmax()

        y = max_location // 128
        x = max_location % 128

        print(
            "Strongest GT point:",
            "x =", x.item(),
            "y =", y.item()
        )

print("\n")
print("=" * 70)
print("EVT DATASET TEST COMPLETE")
print("=" * 70)