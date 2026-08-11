from data.mvtec_test_dataset import MVTecTestDataset
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


dataset = MVTecTestDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=test_transform,
    mask_transform=mask_transform,
)


for index, sample in enumerate(dataset):

    if sample["defect_type"] == "good":

        print("\nNORMAL SAMPLE")
        print("Index:", index)
        print("Label:", sample["label"])
        print("Defect type:", sample["defect_type"])
        print("Mask:", sample["mask"])

        break


for index, sample in enumerate(dataset):

    if sample["defect_type"] != "good":

        print("\nDEFECTIVE SAMPLE")
        print("Index:", index)
        print("Label:", sample["label"])
        print("Defect type:", sample["defect_type"])
        print(
            "Mask shape:",
            sample["mask"].shape
        )

        break