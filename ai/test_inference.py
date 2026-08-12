from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference.predictor import (
    PatchCorePredictor,
)


# ============================================================
# Configuration
# ============================================================

CORESET_PATH = (
    "models/coreset_memory_bank_clean.pt"
)

IMAGE_PATH = (
    "dataset/mvtec/bottle/test/"
    "broken_large/000.png"
)

OUTPUT_PATH = (
    "inference_result.png"
)


# ============================================================
# Predictor
# ============================================================

predictor = PatchCorePredictor(
    coreset_path=CORESET_PATH,
    threshold=0.124763,
)


# ============================================================
# Load Image
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# Prediction
# ============================================================

result = predictor.predict(
    image
)


# ============================================================
# Print Result
# ============================================================

print()
print("=" * 60)
print("PatchCore Inference Result")
print("=" * 60)

print(
    "Prediction:",
    result["prediction"],
)

print(
    "Is anomaly:",
    result["is_anomaly"],
)

print(
    f"Anomaly score: "
    f"{result['anomaly_score']:.6f}"
)

print(
    f"Threshold: "
    f"{result['threshold']:.6f}"
)

print(
    "Anomaly map:",
    result["anomaly_map"].shape,
)


# ============================================================
# Visualization
# ============================================================

image_np = (
    np.asarray(image)
    / 255.0
)

anomaly_map = (
    result["anomaly_map"]
    .numpy()
)


plt.figure(
    figsize=(12, 4)
)


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    1,
)

plt.imshow(
    image_np
)

plt.title(
    "Input"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Anomaly Map
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    2,
)

plt.imshow(
    anomaly_map
)

plt.title(
    "Anomaly Map"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Overlay
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    3,
)

plt.imshow(
    image_np
)

plt.imshow(
    anomaly_map,
    alpha=0.5,
)

plt.title(
    f"{result['prediction']}\n"
    f"Score: "
    f"{result['anomaly_score']:.4f}"
)

plt.axis(
    "off"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


print()
print(
    "Saved:",
    OUTPUT_PATH,
)