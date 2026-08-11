import torch

from preprocessing.transforms import train_transform

from data.mvtec_split import split_normal_images

from data.mvtec_train_dataset import (
    MVTecTrainDataset,
)

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)

from models.patch_embedding import (
    extract_patch_embeddings,
)

from models.memory_bank import (
    MemoryBankBuilder,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = (
    "/content/drive/MyDrive/MVTec_Dataset"
)

CATEGORY = "bottle"

MEMORY_BANK_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/memory_bank_clean.pt"
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
# Step 1: Split normal images
# ============================================================

memory_images, validation_images = (
    split_normal_images(
        root_dir=DATASET_ROOT,
        category=CATEGORY,
        validation_ratio=0.20,
        seed=42,
    )
)


print()
print("=" * 50)
print("Dataset Split")
print("=" * 50)

print(
    "Total normal images:",
    len(memory_images) + len(validation_images),
)

print(
    "Memory-bank images:",
    len(memory_images),
)

print(
    "Validation images:",
    len(validation_images),
)


# ============================================================
# Step 2: Create dataset using ONLY 168 images
# ============================================================

memory_dataset = MVTecTrainDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=train_transform,
    image_paths=memory_images,
)


print()
print(
    "Memory dataset size:",
    len(memory_dataset),
)


# ============================================================
# Step 3: Create feature extractor
# ============================================================

feature_extractor = (
    ResNet18FeatureExtractor()
)

feature_extractor = (
    feature_extractor.to(device)
)

feature_extractor.eval()


# ============================================================
# Step 4: Create Memory Bank Builder
# ============================================================

builder = MemoryBankBuilder(
    feature_extractor=feature_extractor,
    embedding_function=extract_patch_embeddings,
    device=device,
    batch_size=8,
)


# ============================================================
# Step 5: Build memory bank
# ============================================================

print()
print("=" * 50)
print("Building Memory Bank")
print("=" * 50)

memory_bank = builder.build(
    memory_dataset
)


# ============================================================
# Step 6: Display results
# ============================================================

print()
print("=" * 50)
print("Memory Bank Created")
print("=" * 50)

print(
    "Memory bank shape:",
    memory_bank.shape,
)

print(
    "Total patches:",
    memory_bank.shape[0],
)

print(
    "Feature dimension:",
    memory_bank.shape[1],
)


# ============================================================
# Step 7: Save memory bank
# ============================================================

torch.save(
    memory_bank,
    MEMORY_BANK_PATH,
)

print()
print(
    "Memory bank saved to:",
    MEMORY_BANK_PATH,
)