import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

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

THRESHOLD = 0.144791

BATCH_SIZE = 1


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device,
)


# ============================================================
# Dataset
# ============================================================

dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


# ============================================================
# DataLoader
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# Coreset
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


# ============================================================
# Feature Extractor
# ============================================================

feature_extractor = (
    ResNet18FeatureExtractor()
    .to(device)
)

feature_extractor.eval()


# ============================================================
# Find False Negatives
# ============================================================

false_negatives = []


with torch.no_grad():

    for index, batch in enumerate(
        loader
    ):

        image = batch["image"].to(
            device,
            non_blocking=True,
        )

        label = int(
            batch["label"].item()
        )

        defect_type = batch[
            "defect_type"
        ][0]

        # ----------------------------------------------------
        # Only interested in defective images
        # ----------------------------------------------------

        if label != 1:
            continue

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = feature_extractor(
            image
        )

        # ----------------------------------------------------
        # Patch embeddings
        # ----------------------------------------------------

        embeddings = extract_patch_embeddings(
            features
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=2,
        )

        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        similarity = torch.matmul(
            embeddings[0],
            coreset.T,
        )

        max_similarity = (
            similarity.max(
                dim=1
            ).values
        )

        distances = (
            1.0 - max_similarity
        )

        # ----------------------------------------------------
        # Image score
        # ----------------------------------------------------

        image_score = (
            distances.max().item()
        )

        # ----------------------------------------------------
        # False negative
        # ----------------------------------------------------

        if image_score < THRESHOLD:

            false_negatives.append(
                {
                    "index": index,
                    "score": image_score,
                    "defect_type": defect_type,
                }
            )


# ============================================================
# Print False Negatives
# ============================================================

print()

print("=" * 60)
print("False Negative Analysis")
print("=" * 60)

print()

print(
    f"Threshold: {THRESHOLD:.6f}"
)

print(
    f"False negatives found: "
    f"{len(false_negatives)}"
)


for item in false_negatives:

    print(
        f"Index={item['index']} "
        f"Score={item['score']:.6f} "
        f"Type={item['defect_type']}"
    )


# ============================================================
# Inspect the Lowest-Scoring Defective Image
# ============================================================

if len(false_negatives) == 0:

    print()
    print(
        "No false-negative defective images "
        "found at this threshold."
    )

    raise SystemExit


target = min(
    false_negatives,
    key=lambda x: x["score"],
)


target_index = target["index"]


print()

print("=" * 60)
print("Inspecting False Negative")
print("=" * 60)

print(
    f"Dataset index : "
    f"{target_index}"
)

print(
    f"Anomaly score : "
    f"{target['score']:.6f}"
)

print(
    f"Threshold     : "
    f"{THRESHOLD:.6f}"
)

print(
    f"Defect type   : "
    f"{target['defect_type']}"
)


# ============================================================
# Load Target Image
# ============================================================

target_batch = None

for index, batch in enumerate(
    loader
):

    if index == target_index:

        target_batch = batch

        break


if target_batch is None:

    raise RuntimeError(
        "Could not load target image."
    )


image = target_batch[
    "image"
].to(device)


# ============================================================
# Feature Extraction
# ============================================================

with torch.no_grad():

    features = feature_extractor(
        image
    )

    embeddings = extract_patch_embeddings(
        features
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=2,
    )

    similarity = torch.matmul(
        embeddings[0],
        coreset.T,
    )

    max_similarity = (
        similarity.max(
            dim=1
        ).values
    )

    distances = (
        1.0 - max_similarity
    )


# ============================================================
# Create Patch Anomaly Map
# ============================================================

patch_scores = (
    distances
    .detach()
    .cpu()
    .numpy()
)


# Expected patch layout:
#
# 1024 patches = 32 x 32

patch_map = patch_scores.reshape(
    32,
    32,
)


# ============================================================
# Resize Anomaly Map
# ============================================================

anomaly_map = torch.tensor(
    patch_map,
    dtype=torch.float32,
)

anomaly_map = F.interpolate(
    anomaly_map.unsqueeze(0).unsqueeze(0),
    size=(256, 256),
    mode="bilinear",
    align_corners=False,
)

anomaly_map = (
    anomaly_map[0, 0]
    .numpy()
)


# ============================================================
# Get Ground Truth Mask
# ============================================================

mask = None

if "mask" in target_batch:

    mask = target_batch["mask"][0]

    if torch.is_tensor(mask):

        mask = mask.cpu().numpy()


# ============================================================
# Prepare Image
# ============================================================

image_np = (
    target_batch["image"][0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)


# ============================================================
# Normalize Image For Display
# ============================================================

image_np = (
    image_np - image_np.min()
)

if image_np.max() > 0:

    image_np = (
        image_np
        / image_np.max()
    )


# ============================================================
# Visualization
# ============================================================

if mask is not None:

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        image_np
    )

    axes[0].set_title(
        "Defective Image"
    )

    axes[0].axis("off")


    axes[1].imshow(
        anomaly_map,
        cmap="jet",
    )

    axes[1].set_title(
        f"Anomaly Map\n"
        f"Score={target['score']:.6f}"
    )

    axes[1].axis("off")


    axes[2].imshow(
        mask,
        cmap="gray",
    )

    axes[2].set_title(
        "Ground Truth Mask"
    )

    axes[2].axis("off")

else:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(
        image_np
    )

    axes[0].set_title(
        "Defective Image"
    )

    axes[0].axis("off")


    axes[1].imshow(
        anomaly_map,
        cmap="jet",
    )

    axes[1].set_title(
        f"Anomaly Map\n"
        f"Score={target['score']:.6f}"
    )

    axes[1].axis("off")


plt.tight_layout()

plt.show()