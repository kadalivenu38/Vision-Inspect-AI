import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from preprocessing.transforms import test_transform
from data.mvtec_test_dataset import MVTecTestDataset
from models.feature_extractor import ResNet18FeatureExtractor
from models.patch_embedding import extract_patch_embeddings


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

BATCH_SIZE = 4

# Maximum allowed false-positive rate on normal images.
#
# 0.05 = maximum 5% of normal images can be
# incorrectly classified as defective.
#
# With 20 normal images:
#
# 5% of 20 = 1 image
#
MAX_NORMAL_FPR = 0.05


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
# Dataset
# ============================================================
#
# IMPORTANT:
#
# We are doing IMAGE-LEVEL threshold selection here.
#
# We do NOT need segmentation masks.
#
# Therefore mask_transform=None is intentional.
#
# The MVTecTestDataset must support this mode.
# ============================================================

validation_dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=None,
    load_masks=False,
)

print(
    "Validation/Test-style samples:",
    len(validation_dataset),
)


# ============================================================
# DataLoader
# ============================================================

validation_loader = DataLoader(
    validation_dataset,
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
    .to(device)
)

feature_extractor.eval()


# ============================================================
# Storage
# ============================================================

all_scores = []
all_labels = []


# ============================================================
# Calculate Image-Level Anomaly Scores
# ============================================================

print()
print("=" * 60)
print("Calculating validation anomaly scores")
print("=" * 60)


with torch.no_grad():

    for batch_index, batch in enumerate(
        validation_loader
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"]

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

        # Expected shape:
        #
        # [B, number_of_patches, feature_dimension]
        #
        # Example:
        #
        # [4, 1024, 384]

        B, N, C = embeddings.shape

        # ----------------------------------------------------
        # Normalize patch embeddings
        # ----------------------------------------------------

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=2,
        )

        # ----------------------------------------------------
        # Calculate image-level anomaly score
        # ----------------------------------------------------

        for i in range(B):

            image_embeddings = (
                embeddings[i]
            )

            # ------------------------------------------------
            # Cosine similarity against memory bank
            # ------------------------------------------------

            similarity = torch.matmul(
                image_embeddings,
                coreset.T,
            )

            # ------------------------------------------------
            # Nearest memory-bank similarity for every patch
            # ------------------------------------------------

            max_similarity = (
                similarity.max(
                    dim=1
                ).values
            )

            # ------------------------------------------------
            # Convert similarity into anomaly distance
            #
            # Higher distance = more anomalous.
            # ------------------------------------------------

            distances = (
                1.0 - max_similarity
            )

            # ------------------------------------------------
            # PatchCore image score:
            #
            # maximum patch anomaly score.
            # ------------------------------------------------

            image_score = distances.max()

            all_scores.append(
                image_score.item()
            )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(validation_loader)}"
        )


# ============================================================
# Convert to NumPy
# ============================================================

scores = np.asarray(
    all_scores,
    dtype=np.float32,
)

labels = np.asarray(
    all_labels,
    dtype=np.int64,
)


# ============================================================
# Basic Validation
# ============================================================

if len(scores) != len(labels):

    raise RuntimeError(
        "Number of anomaly scores does not "
        "match number of labels."
    )

if len(scores) == 0:

    raise RuntimeError(
        "No anomaly scores were calculated."
    )


# ============================================================
# Separate Normal / Defective Scores
# ============================================================

normal_scores = scores[
    labels == 0
]

defective_scores = scores[
    labels == 1
]


if len(normal_scores) == 0:

    raise RuntimeError(
        "No normal samples found."
    )

if len(defective_scores) == 0:

    raise RuntimeError(
        "No defective samples found."
    )


# ============================================================
# Score Distribution
# ============================================================

print()
print("=" * 60)
print("Validation Score Distribution")
print("=" * 60)

print()
print("Normal images:")

print(
    f"Count   : "
    f"{len(normal_scores)}"
)

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
    f"Count   : "
    f"{len(defective_scores)}"
)

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
# Candidate Thresholds
# ============================================================
#
# Every possible classification boundary is considered.
#
# Prediction rule:
#
# score >= threshold
#       -> defective
#
# score < threshold
#       -> normal
# ============================================================

unique_scores = np.unique(
    scores
)

candidate_thresholds = np.concatenate(
    [
        [
            unique_scores.min() - 1e-6
        ],

        (
            unique_scores[:-1]
            + unique_scores[1:]
        ) / 2.0,

        [
            unique_scores.max() + 1e-6
        ],
    ]
)


# ============================================================
# Evaluate Candidate Thresholds
# ============================================================

results = []


for threshold in candidate_thresholds:

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    # --------------------------------------------------------
    # Normal false-positive rate
    # --------------------------------------------------------

    normal_count = (
        tn + fp
    )

    normal_fpr = (
        fp / normal_count
        if normal_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Defective recall
    # --------------------------------------------------------

    defective_count = (
        fn + tp
    )

    defective_recall = (
        tp / defective_count
        if defective_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Standard classification metrics
    # --------------------------------------------------------

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

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    results.append(
        {
            "threshold": float(
                threshold
            ),

            "accuracy": float(
                accuracy
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "balanced_accuracy": float(
                balanced_accuracy
            ),

            "normal_fpr": float(
                normal_fpr
            ),

            "defective_recall": float(
                defective_recall
            ),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )


# ============================================================
# Filter Thresholds
# ============================================================

valid_results = [
    result
    for result in results
    if result["normal_fpr"]
    <= MAX_NORMAL_FPR
]


# ============================================================
# Print Threshold Search Summary
# ============================================================

print()
print("=" * 60)
print("Threshold Search")
print("=" * 60)

print(
    f"Maximum allowed normal FPR: "
    f"{MAX_NORMAL_FPR:.2%}"
)

print(
    f"Valid thresholds: "
    f"{len(valid_results)} / "
    f"{len(results)}"
)


# ============================================================
# Select Best Threshold
# ============================================================
#
# Priority:
#
# 1. Maximum defective recall
# 2. Maximum F1
# 3. Maximum balanced accuracy
# 4. Lower threshold
#
# The fourth condition makes the selection deterministic
# when the previous metrics are identical.
# ============================================================

if not valid_results:

    print()
    print(
        "WARNING:"
    )

    print(
        "No threshold satisfies the "
        "normal false-positive constraint."
    )

    print(
        "Selecting the threshold with "
        "the best F1 score instead."
    )

    best_result = max(
        results,
        key=lambda x: (
            x["f1"],
            x["defective_recall"],
            x["balanced_accuracy"],
            -x["threshold"],
        ),
    )

else:

    best_result = max(
        valid_results,
        key=lambda x: (
            x["defective_recall"],
            x["f1"],
            x["balanced_accuracy"],
            -x["threshold"],
        ),
    )


# ============================================================
# Selected Threshold
# ============================================================

print()
print("=" * 60)
print("Selected Threshold")
print("=" * 60)

print(
    f"Threshold          : "
    f"{best_result['threshold']:.6f}"
)

print(
    f"Accuracy           : "
    f"{best_result['accuracy']:.4f}"
)

print(
    f"Precision          : "
    f"{best_result['precision']:.4f}"
)

print(
    f"Defective Recall   : "
    f"{best_result['defective_recall']:.4f}"
)

print(
    f"F1 Score           : "
    f"{best_result['f1']:.4f}"
)

print(
    f"Balanced Accuracy  : "
    f"{best_result['balanced_accuracy']:.4f}"
)

print(
    f"Normal FPR         : "
    f"{best_result['normal_fpr']:.4f}"
)


# ============================================================
# Confusion Matrix
# ============================================================

print()
print("Confusion Matrix:")

print(
    np.array(
        [
            [
                best_result["tn"],
                best_result["fp"],
            ],
            [
                best_result["fn"],
                best_result["tp"],
            ],
        ]
    )
)


# ============================================================
# Recommended Configuration
# ============================================================

print()
print("=" * 60)
print("Recommended Configuration")
print("=" * 60)

print()

print(
    "Copy this into "
    "evaluate_patchcore_clean.py:"
)

print()

print(
    f"THRESHOLD = "
    f"{best_result['threshold']:.6f}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "Use this threshold only after "
    "selecting it from validation data."
)

print(
    "Do NOT tune the threshold repeatedly "
    "against the final test set."
)