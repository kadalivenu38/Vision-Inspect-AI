import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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

BATCH_SIZE = 4


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
    device,
)


# ==================================================
# Dataset
# ==================================================

test_dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


print(
    "Test samples:",
    len(test_dataset),
)


# ==================================================
# Load memory bank
# ==================================================

memory_bank = torch.load(
    MEMORY_BANK_PATH,
    map_location="cpu",
)

print(
    "Memory bank:",
    memory_bank.shape,
)


# ==================================================
# Feature extractor
# ==================================================

feature_extractor = (
    ResNet18FeatureExtractor()
    .to(device)
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
# Evaluation
# ==================================================

all_scores = []
all_labels = []
all_defect_types = []


with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"]

        defect_types = batch[
            "defect_type"
        ]

        # ------------------------------------------
        # Feature extraction
        # ------------------------------------------

        features = feature_extractor(
            images
        )

        embeddings = extract_patch_embeddings(
            features
        )

        # ------------------------------------------
        # Anomaly score
        # ------------------------------------------

        scores, anomaly_maps = scorer.score(
            embeddings
        )

        all_scores.extend(
            scores.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_defect_types.extend(
            defect_types
        )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(test_loader)}"
        )


# ==================================================
# Convert to numpy
# ==================================================

scores = np.array(
    all_scores
)

labels = np.array(
    all_labels
)


# ==================================================
# AUROC
# ==================================================

auroc = roc_auc_score(
    labels,
    scores,
)


# ==================================================
# Threshold
# ==================================================

# Use percentile of normal test scores
# ONLY for initial analysis.
#
# Later we will derive threshold from
# validation normal images.

normal_scores = scores[
    labels == 0
]

threshold = np.percentile(
    normal_scores,
    95,
)


predictions = (
    scores >= threshold
).astype(int)


# ==================================================
# Classification metrics
# ==================================================

accuracy = accuracy_score(
    labels,
    predictions,
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0,
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0,
)

cm = confusion_matrix(
    labels,
    predictions,
)


# ==================================================
# Results
# ==================================================

print("\n")
print("=" * 60)
print("PatchCore-style Evaluation")
print("=" * 60)

print(
    f"AUROC     : {auroc:.4f}"
)

print(
    f"Threshold : {threshold:.6f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(cm)


# ==================================================
# Per defect type
# ==================================================

print("\n")
print("=" * 60)
print("Per Defect Type")
print("=" * 60)


unique_types = sorted(
    set(all_defect_types)
)

for defect_type in unique_types:

    indices = [
        i
        for i, value
        in enumerate(all_defect_types)
        if value == defect_type
    ]

    type_scores = scores[
        indices
    ]

    type_predictions = (
        type_scores >= threshold
    ).astype(int)

    detection_rate = (
        type_predictions.mean()
    )

    print(
        f"{defect_type:<20}"
        f"Samples: {len(indices):<5}"
        f"Predicted defective: "
        f"{type_predictions.sum():<5}"
        f"Detection rate: "
        f"{detection_rate:.4f}"
    )