from torch.utils.data import DataLoader

from data.mvtec_test_dataset import MVTecTestDataset
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


dataset = MVTecTestDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=test_transform,
    mask_transform=mask_transform,
)


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
)


batch = next(iter(loader))


print("Image shape:")
print(batch["image"].shape)

print("\nLabels:")
print(batch["label"])

print("\nMask shape:")
print(batch["mask"].shape)

print("\nDefect types:")
print(batch["defect_type"])

print("\nHas mask:")
print(batch["has_mask"])