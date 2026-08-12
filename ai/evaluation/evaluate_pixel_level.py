import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from preprocessing.transforms import (
    test_transform,
    mask_transform,
)

from data.mvtec_test_dataset import (
    MVTecTestDataset,
)

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)

from models.patch_embedding import (
    extract_patch_embeddings,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = (
    "/content/drive/MyDrive/MVTec_Dataset"
)

CATEGORY = "bottle"

CORESET_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/coreset_memory_bank_clean.pt"
)


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

print(
    "Test samples:",
    len(test_dataset),
)


# ============================================================
# DataLoader
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# Load Coreset
# ============================================================

coreset = torch.load(
    CORESET_PATH,
    map_location=device,
)

coreset = coreset.float().to(device)

coreset = F.normalize(
    coreset,
    p=2,
    dim=1,
)

print(
    "Coreset:",
    coreset.shape,
)


# ============================================================
# Feature extractor
# ============================================================

feature_extractor = (
    ResNet18FeatureExtractor()
)

feature_extractor = (
    feature_extractor.to(device)
)

feature_extractor.eval()


# ============================================================
# Storage
# ============================================================

all_anomaly_maps = []
all_ground_truth_masks = []


# ============================================================
# Evaluation
# ============================================================

with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        images = batch["image"].to(
            device
        )

        masks = batch["mask"]

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = feature_extractor(
            images
        )

        # ----------------------------------------------------
        # Patch embeddings
        # ----------------------------------------------------

        embeddings = extract_patch_embeddings(
            features
        )

        # Expected:
        #
        # [B, 1024, 384]

        B, N, C = embeddings.shape

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=2,
        )

        # ----------------------------------------------------
        # Process every image
        # ----------------------------------------------------

        for i in range(B):

            image_embeddings = (
                embeddings[i]
            )

            # ------------------------------------------------
            # Compare image patches against Coreset
            # ------------------------------------------------

            similarity = torch.matmul(
                image_embeddings,
                coreset.T,
            )

            # ------------------------------------------------
            # Find closest normal patch
            # ------------------------------------------------

            max_similarity = similarity.max(
                dim=1
            ).values

            # ------------------------------------------------
            # Convert similarity to anomaly distance
            # ------------------------------------------------

            patch_scores = (
                1.0 - max_similarity
            )

            # ------------------------------------------------
            # Convert 1024 patches into spatial map
            #
            # 1024 = 32 × 32
            # ------------------------------------------------

            patch_map = patch_scores.reshape(
                1,
                1,
                32,
                32,
            )

            # ------------------------------------------------
            # Resize to image size
            # ------------------------------------------------

            anomaly_map = F.interpolate(
                patch_map,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )

            anomaly_map = anomaly_map.squeeze()

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            all_anomaly_maps.append(
                anomaly_map.cpu()
            )

            all_ground_truth_masks.append(
                masks[i].squeeze(0).cpu()
            )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(test_loader)}"
        )


# ============================================================
# Convert to tensors
# ============================================================

anomaly_maps = torch.stack(
    all_anomaly_maps
)

ground_truth_masks = torch.stack(
    all_ground_truth_masks
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
    ground_truth_masks.shape,
)


# ============================================================
# Flatten
# ============================================================

predictions = (
    anomaly_maps.numpy()
    .reshape(-1)
)

targets = (
    ground_truth_masks.numpy()
    .reshape(-1)
)


# ============================================================
# Pixel AUROC
# ============================================================

pixel_auroc = roc_auc_score(
    targets,
    predictions,
)


print(
    f"Pixel AUROC : "
    f"{pixel_auroc:.4f}"
)