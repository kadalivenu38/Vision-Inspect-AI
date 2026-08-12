from preprocessing.transforms import (
    test_transform,
    mask_transform,
)

from data.mvtec_test_dataset import (
    MVTecTestDataset,
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


print("Total samples:", len(dataset))

print("=" * 60)


for index in range(len(dataset)):

    sample = dataset[index]

    image = sample["image"]

    mask = sample["mask"]

    label = sample["label"]

    defect_type = sample["defect_type"]

    # --------------------------------------------------
    # Check image
    # --------------------------------------------------

    if image.shape != (3, 256, 256):

        raise ValueError(
            f"Wrong image shape at index {index}: "
            f"{image.shape}"
        )

    # --------------------------------------------------
    # Check mask
    # --------------------------------------------------

    if mask.shape != (1, 256, 256):

        raise ValueError(
            f"Wrong mask shape at index {index}: "
            f"{mask.shape}"
        )

    # --------------------------------------------------
    # Check normal mask
    # --------------------------------------------------

    if label.item() == 0:

        if mask.max().item() != 0:

            raise ValueError(
                f"Normal image has non-zero mask "
                f"at index {index}"
            )

    # --------------------------------------------------
    # Check defective mask
    # --------------------------------------------------

    else:

        if mask.max().item() != 1:

            raise ValueError(
                f"Defective image mask does not "
                f"contain 1 at index {index}"
            )


print("All dataset samples passed!")

print("=" * 60)

print(
    "Every image shape: [3, 256, 256]"
)

print(
    "Every mask shape: [1, 256, 256]"
)

print(
    "Normal masks: all zeros"
)

print(
    "Defective masks: binary"
)