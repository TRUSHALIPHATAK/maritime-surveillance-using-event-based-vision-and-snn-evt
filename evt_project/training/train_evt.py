import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =========================================================
# PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EVT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(EVT_DIR)

from data.combined_evt_dataset import (
    CombinedEVTSequenceDataset
)

from models.evt_model import EVTModel


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("EVT V2 TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# =========================================================
# DATASET
# =========================================================

dataset = CombinedEVTSequenceDataset()

print(
    "\nDataset sequences:",
    len(dataset)
)

for i in range(len(dataset)):

    events, heatmaps, sequence = dataset[i]

    print(
        f"Sequence {sequence}: "
        f"events={events.shape}, "
        f"heatmaps={heatmaps.shape}"
    )


# =========================================================
# DATALOADER
# =========================================================

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# =========================================================
# MODEL
# =========================================================

model = EVTModel(
    input_channels=2,
    image_size=128,

    # V2: finer spatial patches
    patch_size=8,

    temporal_groups=10,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout=0.1,
)

model = model.to(device)


# =========================================================
# MODEL PARAMETERS
# =========================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel parameters:")
print("Total:", total_params)
print("Trainable:", trainable_params)


# =========================================================
# LOSS FUNCTIONS
# =========================================================

bce_loss = nn.BCEWithLogitsLoss()

mse_loss = nn.MSELoss()


def heatmap_loss(predictions, targets):

    # -----------------------------------------------------
    # Model output:
    #
    # [B, T, 1, 128, 128]
    #
    # Remove channel dimension
    # -----------------------------------------------------

    predictions = predictions.squeeze(2)

    # -----------------------------------------------------
    # BCEWithLogitsLoss
    #
    # IMPORTANT:
    # Use RAW logits here.
    # Do NOT apply sigmoid before BCEWithLogitsLoss.
    # -----------------------------------------------------

    bce = bce_loss(
        predictions,
        targets
    )

    # -----------------------------------------------------
    # MSE
    #
    # Convert logits -> probability heatmap
    # -----------------------------------------------------

    probabilities = torch.sigmoid(
        predictions
    )

    mse = mse_loss(
        probabilities,
        targets
    )

    # -----------------------------------------------------
    # Combined loss
    # -----------------------------------------------------

    loss = bce + mse

    return loss


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)


# =========================================================
# LR SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
)


# =========================================================
# TRAINING SETTINGS
# =========================================================

EPOCHS = 150

best_loss = float("inf")


# =========================================================
# RESULTS DIRECTORY
# =========================================================

results_dir = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "results"
    )
)

os.makedirs(
    results_dir,
    exist_ok=True
)


# =========================================================
# MODEL CHECKPOINT
# =========================================================

model_path = os.path.join(
    results_dir,
    "best_evt_model_v2.pth"
)


# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    epoch_loss = 0.0


    # =====================================================
    # BATCH LOOP
    # =====================================================

    for (
        events,
        heatmaps,
        sequences
    ) in loader:

        events = events.to(
            device,
            non_blocking=True
        )

        heatmaps = heatmaps.to(
            device,
            non_blocking=True
        )


        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        predictions = model(
            events
        )


        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = heatmap_loss(
            predictions,
            heatmaps
        )


        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()


        # -------------------------------------------------
        # Gradient clipping
        # -------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # -------------------------------------------------
        # Optimizer step
        # -------------------------------------------------

        optimizer.step()


        epoch_loss += loss.item()


    # =====================================================
    # AVERAGE EPOCH LOSS
    # =====================================================

    epoch_loss /= len(loader)


    # =====================================================
    # LEARNING RATE
    # =====================================================

    scheduler.step(
        epoch_loss
    )

    lr = optimizer.param_groups[0]["lr"]


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss":
                    epoch_loss,

            },
            model_path
        )

        marker = " <-- BEST"

    else:

        marker = ""


    # =====================================================
    # PRINT PROGRESS
    # =====================================================

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {epoch_loss:.6f} | "
        f"LR: {lr:.2e}"
        f"{marker}"
    )


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 70)
print("EVT V2 TRAINING COMPLETE")
print("=" * 70)

print(
    "Best loss:",
    best_loss
)

print(
    "Saved model:",
    model_path
)