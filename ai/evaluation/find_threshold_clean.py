import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.mvtec_dataset import MVTecTrainDataset
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import test_transform


# ==================================================
# Configuration
# ==================================================

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

MODEL_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/autoencoder_bottle_clean.pth"
)

CATEGORY = "bottle"

BATCH_SIZE = 8

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42

THRESHOLD_PERCENTILE = 95


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

dataset = MVTecTrainDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
)

total_size = len(dataset)

validation_size = int(
    total_size * VALIDATION_RATIO
)

training_size = (
    total_size - validation_size
)


# ==================================================
# Recreate EXACT validation indices
# ==================================================

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

indices = torch.randperm(
    total_size,
    generator=generator,
).tolist()

validation_indices = indices[
    training_size:
]


validation_dataset = Subset(
    dataset,
    validation_indices,
)


print(
    "Total normal samples:",
    total_size
)

print(
    "Validation samples:",
    len(validation_dataset)
)


# ==================================================
# DataLoader
# ==================================================

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ==================================================
# Model
# ==================================================

model = ConvAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model = model.to(device)

model.eval()


# ==================================================
# Calculate reconstruction scores
# ==================================================

normal_scores = []


with torch.no_grad():

    for batch in validation_loader:

        images = batch["image"].to(device)

        reconstructed = model(images)

        # Reconstruction error
        anomaly_map = (
            images - reconstructed
        ) ** 2

        anomaly_map = anomaly_map.mean(
            dim=1
        )

        # Image-level anomaly score
        scores = anomaly_map.flatten(
            start_dim=1
        ).mean(dim=1)

        normal_scores.extend(
            scores.cpu().numpy()
        )


normal_scores = np.array(
    normal_scores
)


# ==================================================
# Statistics
# ==================================================

print(
    "\nNormal reconstruction scores"
)

print(
    f"Minimum : {normal_scores.min():.6f}"
)

print(
    f"Maximum : {normal_scores.max():.6f}"
)

print(
    f"Mean    : {normal_scores.mean():.6f}"
)

print(
    f"Median  : {np.median(normal_scores):.6f}"
)


# ==================================================
# Threshold
# ==================================================

threshold = np.percentile(
    normal_scores,
    THRESHOLD_PERCENTILE,
)


print(
    f"\n{THRESHOLD_PERCENTILE}th percentile "
    f"threshold: {threshold:.6f}"
)