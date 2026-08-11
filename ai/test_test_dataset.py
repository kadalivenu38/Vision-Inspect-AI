from data.mvtec_test_dataset import MVTecTestDataset

from preprocessing.transforms import (
    test_transform,
    mask_transform,
)


# ==================================================
# Dataset configuration
# ==================================================

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"
CATEGORY = "bottle"


# ==================================================
# Create test dataset
# ==================================================

dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


# ==================================================
# Basic information
# ==================================================

print("Total test samples:", len(dataset))


# ==================================================
# Test first 5 samples
# ==================================================

for index in range(min(5, len(dataset))):

    sample = dataset[index]

    print("\nSample:", index)

    print(
        "Image shape:",
        sample["image"].shape,
    )

    print(
        "Label:",
        sample["label"],
    )

    print(
        "Defect type:",
        sample["defect_type"],
    )

    if sample["mask"] is not None:
        print(
            "Mask shape:",
            sample["mask"].shape,
        )
    else:
        print("Mask: None")

    print(
        "Path:",
        sample["path"],
    )