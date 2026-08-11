import matplotlib.pyplot as plt

from data.mvtec_test_dataset import MVTecTestDataset
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


dataset = MVTecTestDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=test_transform,
    mask_transform=mask_transform,
)


defect_index = None

for index, sample in enumerate(dataset):

    if sample["defect_type"] != "good":
        defect_index = index
        break


sample = dataset[defect_index]


image = sample["image"]
mask = sample["mask"]


image = image.permute(1, 2, 0)

mask = mask.squeeze(0)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5),
)


axes[0].imshow(image)
axes[0].set_title(
    f"Defect: {sample['defect_type']}"
)
axes[0].axis("off")


axes[1].imshow(mask)
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")


plt.tight_layout()
plt.show()