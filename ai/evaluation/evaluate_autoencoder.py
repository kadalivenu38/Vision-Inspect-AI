import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from data.mvtec_test_dataset import MVTecTestDataset
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

CATEGORY = "bottle"

MODEL_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/autoencoder_bottle.pth"
)

BATCH_SIZE = 8


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

test_dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print(
    "Test samples:",
    len(test_dataset)
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ConvAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Storage
# --------------------------------------------------

image_scores = []
image_labels = []

pixel_scores = []
pixel_labels = []


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    for batch in test_loader:

        images = batch["image"].to(device)

        labels = batch["label"]

        masks = batch["mask"]

        # ------------------------------
        # Reconstruction
        # ------------------------------

        reconstructed = model(images)

        # ------------------------------
        # Pixel-level reconstruction error
        # ------------------------------

        anomaly_map = (
            images - reconstructed
        ) ** 2

        # Average RGB channels
        anomaly_map = anomaly_map.mean(
            dim=1
        )

        # ------------------------------
        # Image-level anomaly score
        # ------------------------------

        scores = anomaly_map.flatten(
            start_dim=1
        ).mean(dim=1)

        image_scores.extend(
            scores.cpu().numpy()
        )

        image_labels.extend(
            labels.numpy()
        )

        # ------------------------------
        # Pixel-level scores
        # ------------------------------
        for i in range(anomaly_map.shape[0]):
            current_map = (
                anomaly_map[i]
                .cpu()
                .numpy()
                .flatten()
            )

            current_mask = (
                masks[i]
                .cpu()
                .numpy()
                .flatten()
                .astype(np.uint8)
            )

            pixel_scores.extend(
                current_map
            )

            pixel_labels.extend(
                current_mask
            )

# --------------------------------------------------
# Image-level AUROC
# --------------------------------------------------

image_auc = roc_auc_score(
    image_labels,
    image_scores,
)


# --------------------------------------------------
# Pixel-level AUROC
# --------------------------------------------------

pixel_auc = roc_auc_score(
    pixel_labels,
    pixel_scores,
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("Autoencoder Evaluation Results")
print("=" * 50)

print(
    f"Image-level AUROC : {image_auc:.4f}"
)

print(
    f"Pixel-level AUROC : {pixel_auc:.4f}"
)