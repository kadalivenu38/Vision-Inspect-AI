import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from data.mvtec_test_dataset import (
    MVTecTestDataset,
)

from preprocessing.transforms import (
    test_transform,
    mask_transform,
)

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
# Configuration
# ==================================================

DATASET_ROOT = (
    "/content/drive/MyDrive/MVTec_Dataset"
)

CATEGORY = "bottle"

MEMORY_BANK_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/coreset_memory_bank.pt"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# Dataset
# ==================================================

dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


# ==================================================
# Find one defective sample
# ==================================================

defective_index = None

for i in range(len(dataset)):

    sample = dataset[i]

    if sample["label"] == 1:

        defective_index = i

        break


sample = dataset[defective_index]

image = sample["image"].unsqueeze(0).to(device)

mask = sample["mask"]


# ==================================================
# Load model
# ==================================================

feature_extractor = (
    ResNet18FeatureExtractor()
    .to(device)
)

feature_extractor.eval()


# ==================================================
# Load memory bank
# ==================================================

memory_bank = torch.load(
    MEMORY_BANK_PATH,
    map_location="cpu",
)


scorer = NearestNeighborAnomalyScorer(
    memory_bank=memory_bank,
    device=device,
)


# ==================================================
# Extract features
# ==================================================

with torch.no_grad():

    features = feature_extractor(
        image
    )

    embeddings = extract_patch_embeddings(
        features
    )

    scores, anomaly_maps = scorer.score(
        embeddings
    )


# ==================================================
# Resize anomaly map
# ==================================================

anomaly_map = anomaly_maps[
    0
].unsqueeze(0).unsqueeze(0)


anomaly_map = torch.nn.functional.interpolate(
    anomaly_map,
    size=(256, 256),
    mode="bilinear",
    align_corners=False,
)


anomaly_map = anomaly_map[
    0, 0
].cpu().numpy()


# ==================================================
# Image
# ==================================================

image_np = (
    image[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)


image_np = np.clip(
    image_np,
    0,
    1,
)


# ==================================================
# Mask
# ==================================================

mask_np = mask[
    0
].cpu().numpy()


# ==================================================
# Visualization
# ==================================================

plt.figure(figsize=(16, 4))


plt.subplot(1, 4, 1)

plt.imshow(image_np)

plt.title(
    f"Input\n{sample['defect_type']}"
)

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(mask_np)

plt.title(
    "Ground Truth Mask"
)

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(anomaly_map)

plt.title(
    "PatchCore Anomaly Map"
)

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(image_np)

plt.imshow(
    anomaly_map,
    alpha=0.5,
)

plt.title(
    f"Overlay\nScore: "
    f"{scores[0].item():.4f}"
)

plt.axis("off")


plt.tight_layout()

output_path = (
    "/content/Vision-Inspect-AI/"
    "patchcore_visualization.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()

print(
    "Visualization saved to:",
    output_path,
)