import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data.mvtec_dataset import MVTecTrainDataset
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import train_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

CATEGORY = "bottle"

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 1e-3


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = MVTecTrainDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=train_transform,
)


print(
    "Training samples:",
    len(train_dataset)
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)


print(
    "Number of batches:",
    len(train_loader)
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ConvAutoencoder()

model = model.to(device)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = nn.MSELoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch in train_loader:

        images = batch["image"]

        images = images.to(
            device,
            non_blocking=True,
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(images)

        # Reconstruction loss
        loss = criterion(
            reconstructed,
            images,
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.6f}"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/autoencoder_bottle.pth"
)

torch.save(
    model.state_dict(),
    MODEL_PATH,
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)