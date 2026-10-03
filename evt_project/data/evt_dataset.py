import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EVTEventDataset(Dataset):
    """
    Event-based dataset for EVT training.

    Input:
        voxel grid: (50, 2, 128, 128)

    Temporal grouping:
        5 bins -> 1 EVT temporal group

    Therefore:
        50 bins -> 10 temporal groups

    Output:
        events  : (10, 2, 128, 128)
        heatmap : (10, 128, 128)
        group_id: (10,)
    """

    def __init__(
        self,
        voxel_path,
        gt_path,
        bins_per_group=5,
        sigma=3.0,
    ):
        self.voxel_path = voxel_path
        self.gt_path = gt_path
        self.bins_per_group = bins_per_group
        self.sigma = sigma

        # Load voxel grid
        self.voxel = np.load(voxel_path).astype(np.float32)

        if self.voxel.shape != (50, 2, 128, 128):
            raise ValueError(
                f"Expected voxel shape (50,2,128,128), "
                f"got {self.voxel.shape}"
            )

        # Load ground truth
        self.gt = pd.read_csv(gt_path)

        required_columns = [
            "bin_id",
            "target_id",
            "visible",
            "x_128",
            "y_128",
        ]

        for col in required_columns:
            if col not in self.gt.columns:
                raise ValueError(
                    f"Missing GT column: {col}"
                )

        self.num_bins = self.voxel.shape[0]
        self.num_groups = self.num_bins // self.bins_per_group

        print("EVT Dataset")
        print("Voxel shape       :", self.voxel.shape)
        print("GT shape          :", self.gt.shape)
        print("Temporal groups   :", self.num_groups)
        print("Bins/group        :", self.bins_per_group)
        print("Spatial size      : 128x128")

    def __len__(self):
        return self.num_groups

    def _create_heatmap(self, points):
        heatmap = np.zeros(
            (128, 128),
            dtype=np.float32
        )

        radius = int(self.sigma * 3)

        for x, y in points:

            x = int(round(x))
            y = int(round(y))

            if x < 0 or x >= 128:
                continue

            if y < 0 or y >= 128:
                continue

            x0 = max(0, x - radius)
            x1 = min(128, x + radius + 1)

            y0 = max(0, y - radius)
            y1 = min(128, y + radius + 1)

            xx, yy = np.meshgrid(
                np.arange(x0, x1),
                np.arange(y0, y1)
            )

            gaussian = np.exp(
                -(
                    (xx - x) ** 2 +
                    (yy - y) ** 2
                ) /
                (2 * self.sigma ** 2)
            )

            heatmap[y0:y1, x0:x1] = np.maximum(
                heatmap[y0:y1, x0:x1],
                gaussian
            )

        return heatmap

    def __getitem__(self, index):

        start_bin = index * self.bins_per_group
        end_bin = start_bin + self.bins_per_group

        # --------------------------------------------------
        # EVENT INPUT
        # --------------------------------------------------

        group = self.voxel[
            start_bin:end_bin
        ]

        # Combine temporal bins
        events = group.sum(axis=0)

        # Normalize
        max_value = events.max()

        if max_value > 0:
            events = events / max_value

        # --------------------------------------------------
        # GROUND TRUTH HEATMAP
        # --------------------------------------------------

        heatmap = np.zeros(
            (128, 128),
            dtype=np.float32
        )

        gt_group = self.gt[
            (self.gt["bin_id"] >= start_bin) &
            (self.gt["bin_id"] < end_bin) &
            (self.gt["visible"] == 1)
        ]

        points = []

        for _, row in gt_group.iterrows():

            x = row["x_128"]
            y = row["y_128"]

            if pd.isna(x) or pd.isna(y):
                continue

            points.append((x, y))

        heatmap = self._create_heatmap(points)

        return (
            torch.from_numpy(events),
            torch.from_numpy(heatmap),
            torch.tensor(index, dtype=torch.long),
        )


if __name__ == "__main__":

    dataset = EVTEventDataset(
        voxel_path="../../ship_045_event_dataset/3_voxel_grid/ship_045_voxel_grid.npy",
        gt_path="../../snn_project/data/ship_045_event_gt_labels.csv",
    )

    print("\nDataset length:", len(dataset))

    events, heatmap, group_id = dataset[0]

    print("Events :", events.shape)
    print("Heatmap:", heatmap.shape)
    print("Group  :", group_id)
    print("Event min/max:", events.min().item(), events.max().item())
    print("Heatmap min/max:", heatmap.min().item(), heatmap.max().item())