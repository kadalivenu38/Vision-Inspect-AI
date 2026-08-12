from data.mvtec_test_dataset import MVTecTestDataset

from preprocessing.transforms import (
    test_transform,
    mask_transform,
)


DATASET_ROOT = (
    "/content/drive/MyDrive/MVTec_Dataset"
)

CATEGORY = "bottle"


dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


print(
    "Total test samples:",
    len(dataset)
)


# ==================================================
# Check normal sample
# ==================================================

normal_index = None

for i in range(len(dataset)):

    if dataset[i]["label"] == 0:

        normal_index = i
        break


if normal_index is not None:

    sample = dataset[normal_index]

    print("\nNormal sample")

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

    print(
        "Mask shape:",
        sample["mask"].shape,
    )

    print(
        "Mask minimum:",
        sample["mask"].min().item(),
    )

    print(
        "Mask maximum:",
        sample["mask"].max().item(),
    )


# ==================================================
# Check defective sample
# ==================================================

defective_index = None

for i in range(len(dataset)):

    if dataset[i]["label"] == 1:

        defective_index = i
        break


if defective_index is not None:

    sample = dataset[defective_index]

    print("\nDefective sample")

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

    print(
        "Mask shape:",
        sample["mask"].shape,
    )

    print(
        "Mask minimum:",
        sample["mask"].min().item(),
    )

    print(
        "Mask maximum:",
        sample["mask"].max().item(),
    )