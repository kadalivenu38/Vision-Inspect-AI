from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from data.mvtec_test_dataset import MVTecTestDataset
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "dataset/mvtec"
CATEGORY = "bottle"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


print("=" * 50)
print("MVTec AD Dataset EDA")
print("=" * 50)

print(f"Category: {CATEGORY}")
print(f"Total test samples: {len(dataset)}")


# --------------------------------------------------
# 1. Count labels
# --------------------------------------------------

labels = []

for sample in dataset:
    labels.append(sample["label"])


label_counts = Counter(labels)


print("\n--- Image-level Labels ---")

print("Normal images :", label_counts[0])
print("Defect images :", label_counts[1])


# --------------------------------------------------
# 2. Count defect types
# --------------------------------------------------

defect_types = []

for sample in dataset:
    defect_types.append(sample["defect_type"])


defect_counts = Counter(defect_types)


print("\n--- Defect Types ---")

for defect_type, count in sorted(
    defect_counts.items()
):
    print(f"{defect_type:20s}: {count}")


# --------------------------------------------------
# 3. Analyze ground-truth masks
# --------------------------------------------------

defect_area_ratios = []

for sample in dataset:

    if sample["mask"] is None:
        continue

    mask = sample["mask"].squeeze(0).numpy()

    # Number of defective pixels
    defect_pixels = np.sum(mask > 0)

    # Total pixels
    total_pixels = mask.size

    # Percentage of image containing defect
    defect_ratio = (
        defect_pixels / total_pixels
    )

    defect_area_ratios.append(
        defect_ratio
    )


print("\n--- Defect Area Statistics ---")

if defect_area_ratios:

    print(
        "Minimum defect area: "
        f"{min(defect_area_ratios) * 100:.4f}%"
    )

    print(
        "Maximum defect area: "
        f"{max(defect_area_ratios) * 100:.4f}%"
    )

    print(
        "Average defect area: "
        f"{np.mean(defect_area_ratios) * 100:.4f}%"
    )

    print(
        "Median defect area: "
        f"{np.median(defect_area_ratios) * 100:.4f}%"
    )


# --------------------------------------------------
# 4. Check mask values
# --------------------------------------------------

unique_values = set()

for sample in dataset:

    if sample["mask"] is None:
        continue

    mask = sample["mask"].numpy()

    unique_values.update(
        np.unique(mask).tolist()
    )


print("\n--- Mask Values ---")
print("Unique mask values:", sorted(unique_values))


# --------------------------------------------------
# 5. Plot defect type distribution
# --------------------------------------------------

names = list(defect_counts.keys())
counts = list(defect_counts.values())


plt.figure(figsize=(8, 5))

plt.bar(names, counts)

plt.title(
    f"MVTec {CATEGORY} - Test Defect Distribution"
)

plt.xlabel("Defect Type")
plt.ylabel("Number of Images")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# --------------------------------------------------
# 6. Plot defect area distribution
# --------------------------------------------------

if defect_area_ratios:

    plt.figure(figsize=(8, 5))

    plt.hist(
        np.array(defect_area_ratios) * 100,
        bins=20,
    )

    plt.title(
        f"MVTec {CATEGORY} - Defect Area Distribution"
    )

    plt.xlabel(
        "Defect Area (% of Image)"
    )

    plt.ylabel("Number of Images")

    plt.tight_layout()

    plt.show()