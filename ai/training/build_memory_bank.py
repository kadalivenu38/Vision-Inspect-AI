import torch

from data.mvtec_split import split_normal_images
from data.mvtec_train_dataset import MVTecTrainDataset

from preprocessing.transforms import train_transform

from models.memory_bank import MemoryBankBuilder

# Use the SAME feature extractor and embedding code
# that you already used successfully for your previous
# PatchCore memory bank.


DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

MODEL_SAVE_PATH = (
    "/content/Vision-Inspect-AI/ai/models/"
    "memory_bank_clean.pt"
)


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
# Split normal images
# ==================================================

memory_images, validation_images = split_normal_images(
    root_dir=DATASET_ROOT,
    category="bottle",
)


print()
print("Total normal images:", 
      len(memory_images) + len(validation_images))

print("Memory-bank images:", 
      len(memory_images))

print("Validation images:", 
      len(validation_images))


# ==================================================
# Create dataset
# ONLY the 168 memory images
# ==================================================

memory_dataset = MVTecTrainDataset(
    root_dir=DATASET_ROOT,
    category="bottle",
    transform=train_transform,
    image_paths=memory_images,
)


print()
print("Memory dataset size:",
      len(memory_dataset))


# ==================================================
# IMPORTANT
# ==================================================
#
# Keep the feature_extractor and embedding_function
# from your EXISTING working PatchCore script.
#
# Example:
#
# feature_extractor = ...
# embedding_function = ...
#
# Do NOT create new ones here if you already have
# working code.
#
# ==================================================