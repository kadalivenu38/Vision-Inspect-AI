import torch

from preprocessing.transforms import (
    train_transform,
)

from data.mvtec_dataset import (
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


# ==================================================
# Configuration
# ==================================================

DATASET_ROOT = (
    "/content/drive/MyDrive/MVTec_Dataset"
)

CATEGORY = "bottle"


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
    transform=train_transform,
)

print(
    "Training images:",
    len(dataset)
)


# ==================================================
# Feature extractor
# ==================================================

feature_extractor = (
    ResNet18FeatureExtractor()
)

feature_extractor = (
    feature_extractor.to(device)
)


# ==================================================
# Memory bank builder
# ==================================================

builder = MemoryBankBuilder(
    feature_extractor=feature_extractor,
    embedding_function=extract_patch_embeddings,
    device=device,
    batch_size=8,
)


# ==================================================
# Build memory bank
# ==================================================

memory_bank = builder.build(
    dataset
)


# ==================================================
# Results
# ==================================================

print("\n" + "=" * 50)

print("Memory Bank Created")

print("=" * 50)

print(
    "Memory bank shape:",
    memory_bank.shape
)

print("\n" + "=" * 50)

print("Memory Bank Created")

print("=" * 50)

print(
    "Memory bank shape:",
    memory_bank.shape
)

print(
    "Total patches:",
    memory_bank.shape[0]
)

print(
    "Feature dimension:",
    memory_bank.shape[1]
)
MEMORY_BANK_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/memory_bank.pt"
)

torch.save(
    memory_bank,
    MEMORY_BANK_PATH,
)

print(
    "Memory bank saved to:",
    MEMORY_BANK_PATH,
)