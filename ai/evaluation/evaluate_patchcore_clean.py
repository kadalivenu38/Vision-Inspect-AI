import numpy as np
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

from preprocessing.transforms import (
    test_transform,
    mask_transform,
)

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

# ============================================================
# IMPORTANT
#
# This value MUST come from validation.
#
# Do NOT tune this value using the test set.
# ============================================================

THRESHOLD = 0.124763

BATCH_SIZE = 4


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
    transform=test_transform,
    mask_transform=mask_transform,
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
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
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
all_paths = []


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
        # Calculate anomaly score
        # ----------------------------------------------------

        batch_scores = []

        for i in range(B):

            image_embeddings = (
                embeddings[i]
            )

            similarity = torch.matmul(
                image_embeddings,
                coreset.T,
            )

            max_similarity = (
                similarity.max(
                    dim=1
                ).values
            )

            distances = (
                1.0 - max_similarity
            )

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

        # ----------------------------------------------------
        # Optional path information
        # ----------------------------------------------------

        if "path" in batch:

            all_paths.extend(
                list(batch["path"])
            )

        elif "image_path" in batch:

            all_paths.extend(
                list(batch["image_path"])
            )

        else:

            all_paths.extend(
                [None] * B
            )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(test_loader)}"
        )


# ============================================================
# Convert Results
# ============================================================

scores = np.array(
    all_scores,
    dtype=np.float32,
)

labels = np.array(
    all_labels,
    dtype=np.int64,
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
    labels=[0, 1],
)


# ============================================================
# Print Main Results
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

    type_predictions = (
        predictions[indices]
    )

    detected = (
        type_predictions == 1
    ).sum()

    total = len(indices)

    detection_rate = (
        detected / total
        if total > 0
        else 0.0
    )

    print(
        f"{defect_type:<20}"
        f" Samples: {total:<3}"
        f" Predicted defective: "
        f"{detected:<3}"
        f" Detection rate: "
        f"{detection_rate:.4f}"
    )


# ============================================================
# Overall Anomaly Score Distribution
# ============================================================

print()

print("=" * 60)
print("Overall Anomaly Score Distribution")
print("=" * 60)

normal_scores = scores[
    labels == 0
]

defective_scores = scores[
    labels == 1
]


print()
print("Normal images:")

print(
    f"Minimum : "
    f"{normal_scores.min():.6f}"
)

print(
    f"Maximum : "
    f"{normal_scores.max():.6f}"
)

print(
    f"Mean    : "
    f"{normal_scores.mean():.6f}"
)

print(
    f"Median  : "
    f"{np.median(normal_scores):.6f}"
)


print()
print("Defective images:")

print(
    f"Minimum : "
    f"{defective_scores.min():.6f}"
)

print(
    f"Maximum : "
    f"{defective_scores.max():.6f}"
)

print(
    f"Mean    : "
    f"{defective_scores.mean():.6f}"
)

print(
    f"Median  : "
    f"{np.median(defective_scores):.6f}"
)


# ============================================================
# False Positives
# ============================================================

false_positive_indices = np.where(
    (labels == 0)
    & (predictions == 1)
)[0]


# ============================================================
# False Negatives
# ============================================================

false_negative_indices = np.where(
    (labels == 1)
    & (predictions == 0)
)[0]


print()

print("=" * 60)
print("Error Analysis")
print("=" * 60)


print()
print(
    f"False Positives : "
    f"{len(false_positive_indices)}"
)

for index in false_positive_indices:

    print(
        f"  Index={index} "
        f"Score={scores[index]:.6f} "
        f"Type={all_defect_types[index]} "
        f"Path={all_paths[index]}"
    )


print()
print(
    f"False Negatives : "
    f"{len(false_negative_indices)}"
)

for index in false_negative_indices:

    print(
        f"  Index={index} "
        f"Score={scores[index]:.6f} "
        f"Type={all_defect_types[index]} "
        f"Path={all_paths[index]}"
    )


# ============================================================
# Borderline Samples
# ============================================================

print()

print("=" * 60)
print("Borderline Samples")
print("=" * 60)

distance_from_threshold = np.abs(
    scores - THRESHOLD
)

borderline_indices = np.argsort(
    distance_from_threshold
)[:10]


for index in borderline_indices:

    actual = (
        "Defective"
        if labels[index] == 1
        else "Normal"
    )

    predicted = (
        "Defective"
        if predictions[index] == 1
        else "Normal"
    )

    print(
        f"Index={index:<3} "
        f"Score={scores[index]:.6f} "
        f"Actual={actual:<10} "
        f"Predicted={predicted:<10} "
        f"Type={all_defect_types[index]} "
        f"Path={all_paths[index]}"
    )


# ============================================================
# Final Interpretation
# ============================================================

print()

print("=" * 60)
print("Evaluation Summary")
print("=" * 60)

print(
    f"Correct predictions : "
    f"{(predictions == labels).sum()}/{len(labels)}"
)

print(
    f"False positives     : "
    f"{len(false_positive_indices)}"
)

print(
    f"False negatives     : "
    f"{len(false_negative_indices)}"
)

print(
    f"Normal acceptance   : "
    f"{(
        (predictions[labels == 0] == 0).sum()
        / (labels == 0).sum()
    ):.4f}"
)

print(
    f"Defect detection    : "
    f"{(
        (predictions[labels == 1] == 1).sum()
        / (labels == 1).sum()
    ):.4f}"
)