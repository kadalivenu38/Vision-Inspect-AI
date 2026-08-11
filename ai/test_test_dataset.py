from data.mvtec_test_dataset import MVTecTestDataset
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


dataset = MVTecTestDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=test_transform,
    mask_transform=mask_transform,
)


print("Total test samples:", len(dataset))


for index in range(5):

    sample = dataset[index]

    print("\nSample:", index)
    print("Image shape:", sample["image"].shape)
    print("Label:", sample["label"])
    print("Defect type:", sample["defect_type"])
    print("Mask:", None if sample["mask"] is None else sample["mask"].shape)
    print("Path:", sample["path"])