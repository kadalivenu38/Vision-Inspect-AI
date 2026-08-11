import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocessing.transforms import train_transform

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

THRESHOLD = 0.103228


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
# Test Dataset
# ============================================================

test_dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=train_transform,
    mask_transform=None,
)

print(
    "Test samples:",
    len(test_dataset),
)


# ============================================================
# DataLoader
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# ============================================================
# Load Coreset
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

print(
    "Coreset:",
    coreset.shape,
)


# ============================================================
# Feature Extractor
# ============================================================

feature_extractor = (
    ResNet18FeatureExtractor()
)

feature_extractor = (
    feature_extractor.to(device)
)

feature_extractor.eval()


# ============================================================
# Storage
# ============================================================

all_scores = []
all_labels = []
all_defect_types = []


# ============================================================
# Evaluation
# ============================================================

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

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = feature_extractor(
            images
        )

        # ----------------------------------------------------
        # Patch embeddings
        # ----------------------------------------------------

        embeddings = extract_patch_embeddings(
            features
        )

        # Shape:
        #
        # [B, 1024, 384]

        B, N, C = embeddings.shape

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=2,
        )

        # ----------------------------------------------------
        # Compare each image's patches
        # against the normal coreset
        # ----------------------------------------------------

        batch_scores = []

        for i in range(B):

            image_embeddings = (
                embeddings[i]
            )

            # [1024, 384] × [384, 8601]
            #
            # Result:
            #
            # [1024, 8601]

            similarity = torch.matmul(
                image_embeddings,
                coreset.T,
            )

            # For every patch:
            #
            # find the closest normal patch

            max_similarity = similarity.max(
                dim=1
            ).values

            # Convert similarity to distance

            distances = (
                1.0 - max_similarity
            )

            # Image-level anomaly score:
            #
            # maximum patch anomaly

            image_score = distances.max()

            batch_scores.append(
                image_score.item()
            )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_scores.extend(
            batch_scores
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_defect_types.extend(
            list(defect_types)
        )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(test_loader)}"
        )


# ============================================================
# Convert to lists / arrays
# ============================================================

import numpy as np

scores = np.array(
    all_scores
)

labels = np.array(
    all_labels
)


# ============================================================
# Predictions
# ============================================================

predictions = (
    scores >= THRESHOLD
).astype(int)


# ============================================================
# Metrics
# ============================================================

auroc = roc_auc_score(
    labels,
    scores,
)

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


# ============================================================
# Print results
# ============================================================

print()
print("=" * 60)
print("PatchCore-style Evaluation")
print("=" * 60)

print(
    f"Threshold : {THRESHOLD:.6f}"
)

print(
    f"AUROC     : {auroc:.4f}"
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


# ============================================================
# Confusion Matrix
# ============================================================

print()
print("Confusion Matrix:")

print(cm)


# ============================================================
# Classification Report
# ============================================================

print()
print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Normal",
            "Defective",
        ],
        zero_division=0,
    )
)


# ============================================================
# Per Defect Type
# ============================================================

print()
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

    type_predictions = predictions[
        indices
    ]

    detected = (
        type_predictions == 1
    ).sum()

    total = len(indices)

    detection_rate = (
        detected / total
    )

    print(
        f"{defect_type:<20}"
        f" Samples: {total:<3}"
        f" Predicted defective: "
        f"{detected:<3}"
        f" Detection rate: "
        f"{detection_rate:.4f}"
    )