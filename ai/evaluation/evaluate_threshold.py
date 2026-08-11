import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data.mvtec_test_dataset import MVTecTestDataset
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

CATEGORY = "bottle"

MODEL_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/autoencoder_bottle.pth"
)

BATCH_SIZE = 8

THRESHOLD = 0.000980


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

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
)


print(
    "Test samples:",
    len(test_dataset)
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ConvAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Storage
# --------------------------------------------------

true_labels = []

predicted_labels = []

anomaly_scores = []

defect_types = []


# --------------------------------------------------
# Inference
# --------------------------------------------------

with torch.no_grad():

    for batch in test_loader:

        images = batch["image"].to(device)

        labels = batch["label"]

        reconstructed = model(images)

        # ------------------------------------------
        # Pixel reconstruction error
        # ------------------------------------------

        anomaly_map = (
            images - reconstructed
        ) ** 2

        anomaly_map = anomaly_map.mean(
            dim=1
        )

        # ------------------------------------------
        # Image-level anomaly score
        # ------------------------------------------

        scores = anomaly_map.flatten(
            start_dim=1
        ).mean(dim=1)

        scores_np = scores.cpu().numpy()

        # ------------------------------------------
        # Threshold
        # ------------------------------------------

        predictions = (
            scores_np > THRESHOLD
        ).astype(int)

        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predictions
        )

        anomaly_scores.extend(
            scores_np
        )

        defect_types.extend(
            batch["defect_type"]
        )


# --------------------------------------------------
# Convert to NumPy
# --------------------------------------------------

true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)

anomaly_scores = np.array(
    anomaly_scores
)


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predicted_labels,
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)

print("Threshold Evaluation")

print("=" * 60)

print(
    f"Threshold : {THRESHOLD:.6f}"
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


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Normal",
            "Defective",
        ],
        zero_division=0,
    )
)


# --------------------------------------------------
# Per defect type
# --------------------------------------------------

print("\n" + "=" * 60)

print("Per Defect Type")

print("=" * 60)


unique_types = sorted(
    set(defect_types)
)


for defect_type in unique_types:

    indices = [
        i
        for i, d in enumerate(defect_types)
        if d == defect_type
    ]

    type_true = true_labels[
        indices
    ]

    type_pred = predicted_labels[
        indices
    ]

    detection_rate = (
        type_pred.mean()
        if len(type_pred) > 0
        else 0
    )

    print(
        f"{defect_type:20s} "
        f"Samples: {len(indices):2d} "
        f"Predicted defective: "
        f"{type_pred.sum():2d} "
        f"Detection rate: "
        f"{detection_rate:.4f}"
    )