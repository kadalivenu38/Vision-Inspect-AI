import matplotlib.pyplot as plt

from preprocessing.transforms import train_transform
from data.mvtec_dataset import MVTecTrainDataset


dataset = MVTecTrainDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=train_transform,
)


fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6),
)


for index, ax in enumerate(axes.flat):

    sample = dataset[index]

    image = sample["image"]

    image = image.permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(f"Sample {index}")
    ax.axis("off")


plt.tight_layout()
plt.show()