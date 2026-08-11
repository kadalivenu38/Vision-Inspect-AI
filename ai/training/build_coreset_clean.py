import torch

from models.coreset import greedy_coreset_sampling


# ============================================================
# Configuration
# ============================================================

MEMORY_BANK_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/memory_bank_clean.pt"
)

CORESET_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/coreset_memory_bank_clean.pt"
)

SAMPLING_RATIO = 0.05


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
# Load memory bank
# ============================================================

memory_bank = torch.load(
    MEMORY_BANK_PATH,
    map_location=device,
)


print()
print("=" * 50)
print("Memory Bank Loaded")
print("=" * 50)

print(
    "Memory bank shape:",
    memory_bank.shape,
)


# ============================================================
# Create coreset
# ============================================================

print()
print("=" * 50)
print("Creating Coreset")
print("=" * 50)

coreset = greedy_coreset_sampling(
    embeddings=memory_bank,
    sampling_ratio=SAMPLING_RATIO,
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 50)
print("Coreset Created")
print("=" * 50)

print(
    "Original memory bank:",
    memory_bank.shape,
)

print(
    "Coreset:",
    coreset.shape,
)


# ============================================================
# Save coreset
# ============================================================

torch.save(
    coreset,
    CORESET_PATH,
)

print()
print(
    "Coreset saved to:",
    CORESET_PATH,
)