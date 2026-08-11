from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.mvtec_dataset import MVTecTrainDataset
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import train_transform, test_transform


# ==================================================
# Configuration
# ==================================================

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

CATEGORY = "bottle"

MODEL_DIR = Path(
    "/content/Vision-Inspect-AI/ai/models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "autoencoder_bottle_clean.pth"
)

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.001

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Dataset
# ==================================================

full_dataset = MVTecTrainDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=train_transform,
)

print(
    "Total normal images:",
    len(full_dataset)
)


# ==================================================
# Train / Validation Split
# ==================================================

validation_size = int(
    len(full_dataset) * VALIDATION_RATIO
)

training_size = (
    len(full_dataset) - validation_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    full_dataset,
    [training_size, validation_size],
    generator=generator,
)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)


# ==================================================
# DataLoaders
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(validation_loader)
)


# ==================================================
# Model
# ==================================================

model = ConvAutoencoder()

model = model.to(device)


# ==================================================
# Loss + Optimizer
# ==================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# Best model tracking
# ==================================================

best_validation_loss = float("inf")


# ==================================================
# Training
# ==================================================

for epoch in range(EPOCHS):

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model.train()

    training_loss = 0.0

    for batch in train_loader:

        images = batch["image"].to(device)

        optimizer.zero_grad()

        reconstructed = model(images)

        loss = criterion(
            reconstructed,
            images,
        )

        loss.backward()

        optimizer.step()

        training_loss += (
            loss.item()
            * images.size(0)
        )

    training_loss /= len(
        train_loader.dataset
    )


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for batch in validation_loader:

            images = batch["image"].to(device)

            reconstructed = model(images)

            loss = criterion(
                reconstructed,
                images,
            )

            validation_loss += (
                loss.item()
                * images.size(0)
            )

    validation_loss /= len(
        validation_loader.dataset
    )


    # ----------------------------------------------
    # Print progress
    # ----------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {training_loss:.6f} "
        f"Validation Loss: {validation_loss:.6f}"
    )


    # ----------------------------------------------
    # Save best model
    # ----------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print(
            f"  ✓ Best model saved "
            f"(val loss: {validation_loss:.6f})"
        )


print("\nTraining completed.")

print(
    "Best validation loss:",
    f"{best_validation_loss:.6f}"
)

print(
    "Model saved to:",
    MODEL_PATH
)