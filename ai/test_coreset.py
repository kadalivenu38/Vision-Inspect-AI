import torch

from models.coreset import (
    greedy_coreset_sampling,
)


# -----------------------------------------
# Load memory bank
# -----------------------------------------

MEMORY_BANK_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/memory_bank.pt"
)


memory_bank = torch.load(
    MEMORY_BANK_PATH,
    map_location="cpu",
)


print(
    "Original memory bank:",
    memory_bank.shape,
)


# -----------------------------------------
# Coreset sampling
# -----------------------------------------

coreset = greedy_coreset_sampling(
    memory_bank,
    sampling_ratio=0.05,
)


print("\n" + "=" * 50)

print("Coreset created")

print("=" * 50)

print(
    "Original:",
    memory_bank.shape,
)

print(
    "Coreset:",
    coreset.shape,
)


# -----------------------------------------
# Save
# -----------------------------------------

OUTPUT_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/"
    "coreset_memory_bank.pt"
)

torch.save(
    coreset,
    OUTPUT_PATH,
)

print(
    "\nSaved to:",
    OUTPUT_PATH,
)