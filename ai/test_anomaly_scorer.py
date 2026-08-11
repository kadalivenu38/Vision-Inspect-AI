import torch

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)

from models.patch_embedding import (
    extract_patch_embeddings,
)

from models.anomaly_scorer import (
    NearestNeighborAnomalyScorer,
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


# ==================================================
# Load memory bank
# ==================================================

memory_bank = torch.load(
    "/content/Vision-Inspect-AI/"
    "ai/models/coreset_memory_bank.pt",
    map_location="cpu",
)

print(
    "Memory bank:",
    memory_bank.shape
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

feature_extractor.eval()


# ==================================================
# Anomaly scorer
# ==================================================

scorer = NearestNeighborAnomalyScorer(
    memory_bank=memory_bank,
    device=device,
)


# ==================================================
# Fake test images
# ==================================================

images = torch.randn(
    2,
    3,
    256,
    256,
).to(device)


# ==================================================
# Feature extraction
# ==================================================

with torch.no_grad():

    features = feature_extractor(
        images
    )

    embeddings = extract_patch_embeddings(
        features
    )


# ==================================================
# Anomaly scoring
# ==================================================

with torch.no_grad():

    image_scores, anomaly_maps = (
        scorer.score(
            embeddings
        )
    )


# ==================================================
# Results
# ==================================================

print("\n" + "=" * 50)

print("Anomaly Scoring Test")

print("=" * 50)

print(
    "Embeddings:",
    embeddings.shape
)

print(
    "Image scores:",
    image_scores.shape
)

print(
    "Anomaly maps:",
    anomaly_maps.shape
)

print(
    "\nImage scores:",
    image_scores
)

print(
    "\nAnomaly map min:",
    anomaly_maps.min().item()
)

print(
    "Anomaly map max:",
    anomaly_maps.max().item()
)

print(
    "Anomaly map mean:",
    anomaly_maps.mean().item()
)