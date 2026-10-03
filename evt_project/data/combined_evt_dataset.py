import os
import sys
import torch
from torch.utils.data import Dataset

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

from evt_project.data.evt_dataset import EVTEventDataset


class CombinedEVTSequenceDataset(Dataset):
    """
    Sequence-level EVT dataset.

    Each sequence contains:
        50 event bins
        2 polarity channels
        128 x 128 spatial resolution

    Temporal grouping:
        5 bins -> 1 group
        50 bins -> 10 groups

    Output:
        events   : (10, 2, 128, 128)
        heatmaps : (10, 128, 128)
        name     : sequence name
    """

    def __init__(self):

        # ====================================================
        # SEQUENCE 045
        # ====================================================

        voxel_045 = os.path.join(
            PROJECT_ROOT,
            "ship_045_event_dataset",
            "3_voxel_grid",
            "ship_045_voxel_grid.npy"
        )

        gt_045 = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "data",
            "ship_045_event_gt_labels.csv"
        )

        # ====================================================
        # SEQUENCE 047
        # ====================================================

        voxel_047 = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "ship_047_event_dataset",
            "3_voxel_grid",
            "ship_047_voxel_grid.npy"
        )

        gt_047 = os.path.join(
            PROJECT_ROOT,
            "snn_project",
            "data",
            "ship_047_event_gt_labels.csv"
        )

        # ====================================================
        # CHECK FILES
        # ====================================================

        required_files = {
            "045 voxel": voxel_045,
            "045 GT": gt_045,
            "047 voxel": voxel_047,
            "047 GT": gt_047,
        }

        for name, path in required_files.items():

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"\n{name} not found:\n{path}"
                )

        # ====================================================
        # CREATE DATASETS
        # ====================================================

        self.datasets = [

            EVTEventDataset(
                voxel_path=voxel_045,
                gt_path=gt_045,
                bins_per_group=5,
                sigma=3.0,
            ),

            EVTEventDataset(
                voxel_path=voxel_047,
                gt_path=gt_047,
                bins_per_group=5,
                sigma=3.0,
            ),
        ]

        self.sequence_names = [
            "045",
            "047"
        ]

    # ========================================================
    # NUMBER OF SEQUENCES
    # ========================================================

    def __len__(self):
        return len(self.datasets)

    # ========================================================
    # GET COMPLETE SEQUENCE
    # ========================================================

    def __getitem__(self, index):

        dataset = self.datasets[index]

        events = []
        heatmaps = []

        # Collect all 10 temporal groups
        for group_id in range(len(dataset)):

            event_group, heatmap, returned_group_id = dataset[group_id]

            events.append(event_group)
            heatmaps.append(heatmap)

        # ----------------------------------------------------
        # Stack temporal groups
        # ----------------------------------------------------

        events = torch.stack(events)

        heatmaps = torch.stack(heatmaps)

        sequence_name = self.sequence_names[index]

        return (
            events,
            heatmaps,
            sequence_name
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    dataset = CombinedEVTSequenceDataset()

    print()
    print("=" * 70)
    print("COMBINED EVT SEQUENCE DATASET")
    print("=" * 70)

    print()
    print("Project root:", PROJECT_ROOT)
    print("Total sequences:", len(dataset))

    for i in range(len(dataset)):

        events, heatmaps, name = dataset[i]

        print()
        print("-" * 70)
        print("Sequence:", name)
        print("Events shape :", events.shape)
        print("Heatmap shape:", heatmaps.shape)
        print(
            "Events min/max:",
            float(events.min()),
            float(events.max())
        )
        print(
            "Heatmap min/max:",
            float(heatmaps.min()),
            float(heatmaps.max())
        )

    print()
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)