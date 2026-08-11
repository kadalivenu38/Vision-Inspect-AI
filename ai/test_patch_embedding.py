import torch

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)

from models.patch_embedding import (
    extract_patch_embeddings,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Feature extractor
# --------------------------------------------------

model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# --------------------------------------------------
# Test image
# --------------------------------------------------

images = torch.randn(
    2,
    3,
    256,
    256,
).to(device)


# --------------------------------------------------
# Extract features
# --------------------------------------------------

with torch.no_grad():

    features = model(images)


# --------------------------------------------------
# Extract patch embeddings
# --------------------------------------------------

with torch.no_grad():

    embeddings = extract_patch_embeddings(
        features
    )


# --------------------------------------------------
# Print results
# --------------------------------------------------

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Expected shape:",
    "[2, 1024, 384]"
)

print(
    "Number of patches:",
    embeddings.shape[1]
)

print(
    "Feature dimension:",
    embeddings.shape[2]
)