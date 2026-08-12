import torch

from torch.utils.data import DataLoader

from data.mvtec_test_dataset import (
    MVTecTestDataset,
)

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


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)


batch = next(iter(loader))


print("Image batch:")
print(batch["image"].shape)

print("\nLabel batch:")
print(batch["label"])

print("\nMask batch:")
print(batch["mask"].shape)

print("\nDefect types:")
print(batch["defect_type"])

print("\nPaths:")
for path in batch["path"]:
    print(path)