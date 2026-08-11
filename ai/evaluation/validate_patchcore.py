import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import numpy as np

from preprocessing.transforms import train_transform

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

VALIDATION_RATIO = 0.20
SEED = 42


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
# Validation Dataset
# ============================================================

class ValidationDataset(Dataset):

    def __init__(
        self,
        image_paths,
        transform=None,
    ):

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return {
            "image": image,
            "path": str(image_path),
        }


# ============================================================
# Get normal images
# ============================================================

train_good_dir = (
    Path(DATASET_ROOT)
    / CATEGORY
    / "train"
    / "good"
)

all_images = sorted(
    train_good_dir.glob("*.png")
)


print(
    "Total normal images:",
    len(all_images),
)


# ============================================================
# Reproduce same 168 / 41 split
# ============================================================

generator = torch.Generator()

generator.manual_seed(SEED)

indices = torch.randperm(
    len(all_images),
    generator=generator,
).tolist()


validation_size = int(
    len(all_images) * VALIDATION_RATIO
)

validation_indices = indices[
    :validation_size
]

validation_images = [
    all_images[i]
    for i in validation_indices
]


print(
    "Validation images:",
    len(validation_images),
)


# ============================================================
# Dataset
# ============================================================

validation_dataset = ValidationDataset(
    image_paths=validation_images,
    transform=train_transform,
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# ============================================================
# Load coreset
# ============================================================

coreset = torch.load(
    CORESET_PATH,
    map_location=device,
)

coreset = coreset.to(device)

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
# Calculate anomaly score
# ============================================================

scores = []


with torch.no_grad():

    for batch_index, batch in enumerate(
        validation_loader
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        # --------------------------------------------
        # Extract CNN features
        # --------------------------------------------

        features = feature_extractor(
            images
        )

        # --------------------------------------------
        # Convert features to patches
        # --------------------------------------------

        embeddings = extract_patch_embeddings(
            features
        )

        # Shape:
        # [B, 1024, 384]

        B, N, C = embeddings.shape

        embeddings = embeddings.reshape(
            B * N,
            C,
        )

        # --------------------------------------------
        # Normalize
        # --------------------------------------------

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        coreset_normalized = (
            torch.nn.functional.normalize(
                coreset,
                p=2,
                dim=1,
            )
        )

        # --------------------------------------------
        # Compare each patch to coreset
        # --------------------------------------------

        similarity = torch.matmul(
            embeddings,
            coreset_normalized.T,
        )

        # Highest similarity = closest normal patch

        max_similarity = similarity.max(
            dim=1
        ).values

        # Convert similarity into distance

        distances = (
            1.0 - max_similarity
        )

        # --------------------------------------------
        # One score per image
        # --------------------------------------------

        distances = distances.reshape(
            B,
            N,
        )

        image_scores = distances.max(
            dim=1
        ).values

        scores.extend(
            image_scores.cpu().numpy().tolist()
        )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(validation_loader)}"
        )


# ============================================================
# Convert to NumPy
# ============================================================

scores = np.array(scores)


# ============================================================
# Statistics
# ============================================================

print()
print("=" * 60)
print("Validation Anomaly Scores")
print("=" * 60)

print(
    "Number of validation images:",
    len(scores),
)

print(
    f"Minimum : {scores.min():.6f}"
)

print(
    f"Maximum : {scores.max():.6f}"
)

print(
    f"Mean    : {scores.mean():.6f}"
)

print(
    f"Median  : {np.median(scores):.6f}"
)


# ============================================================
# Threshold
# ============================================================

threshold = np.percentile(
    scores,
    95,
)


print()
print(
    f"95th percentile threshold: "
    f"{threshold:.6f}"
)


# ============================================================
# Score distribution
# ============================================================

print()
print("=" * 60)
print("Validation Scores")
print("=" * 60)

for index, score in enumerate(scores):

    print(
        f"{index + 1:02d}. "
        f"{score:.6f}"
    )