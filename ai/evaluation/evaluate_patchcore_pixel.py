import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from data.mvtec_test_dataset import MVTecTestDataset

from preprocessing.transforms import (
    test_transform,
    mask_transform,
)

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)

from models.patch_embedding import (
    extract_patch_embeddings,
)

from models.anomaly_scorer import (
    NearestNeighborAnomalyScorer,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

CATEGORY = "bottle"

MEMORY_BANK_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/coreset_memory_bank.pt"
)

BATCH_SIZE = 4


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# Dataset
# ============================================================

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
    num_workers=0,
)

print("Test samples:", len(test_dataset))


# ============================================================
# Feature extractor
# ============================================================

feature_extractor = (
    ResNet18FeatureExtractor()
    .to(device)
)

feature_extractor.eval()


# ============================================================
# Memory bank
# ============================================================

memory_bank = torch.load(
    MEMORY_BANK_PATH,
    map_location="cpu",
)

print(
    "Memory bank:",
    memory_bank.shape,
)


# ============================================================
# Anomaly scorer
# ============================================================

scorer = NearestNeighborAnomalyScorer(
    memory_bank=memory_bank,
    device=device,
)


# ============================================================
# Storage
# ============================================================

all_anomaly_maps = []
all_masks = []


# ============================================================
# Evaluation
# ============================================================

with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        images = batch["image"].to(device)

        masks = batch["mask"]

        # ----------------------------------------
        # Extract features
        # ----------------------------------------

        features = feature_extractor(images)

        embeddings = extract_patch_embeddings(
            features
        )

        # ----------------------------------------
        # Calculate anomaly maps
        # ----------------------------------------

        _, anomaly_maps = scorer.score(
            embeddings
        )

        # anomaly_maps:
        # [B, 32, 32]

        # ----------------------------------------
        # Resize anomaly maps
        # ----------------------------------------

        anomaly_maps = anomaly_maps.unsqueeze(1)

        anomaly_maps = torch.nn.functional.interpolate(
            anomaly_maps,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        anomaly_maps = anomaly_maps.squeeze(1)

        all_anomaly_maps.append(
            anomaly_maps.cpu()
        )

        # ----------------------------------------
        # Ground truth masks
        # ----------------------------------------

        for i in range(len(masks)):

            if masks[i] is None:

                mask = torch.zeros(
                    (256, 256),
                    dtype=torch.float32,
                )

            else:

                mask = masks[i]

                if mask.ndim == 3:
                    mask = mask.squeeze(0)

            all_masks.append(mask)

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(test_loader)}"
        )


# ============================================================
# Combine
# ============================================================

anomaly_maps = torch.cat(
    all_anomaly_maps,
    dim=0,
)

masks = torch.stack(
    all_masks,
    dim=0,
)


print()
print("=" * 60)
print("Pixel-level Evaluation")
print("=" * 60)

print(
    "Anomaly maps:",
    anomaly_maps.shape,
)

print(
    "Ground truth masks:",
    masks.shape,
)


# ============================================================
# Flatten
# ============================================================

scores = anomaly_maps.numpy().reshape(-1)

ground_truth = masks.numpy().reshape(-1)


# ============================================================
# Pixel AUROC
# ============================================================

pixel_auroc = roc_auc_score(
    ground_truth,
    scores,
)


print(
    f"Pixel AUROC : "
    f"{pixel_auroc:.4f}"
)