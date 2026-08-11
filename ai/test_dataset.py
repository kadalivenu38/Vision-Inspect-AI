from preprocessing.transforms import train_transform
from data.mvtec_dataset import MVTecTrainDataset


dataset = MVTecTrainDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=train_transform,
)


print("Dataset size:", len(dataset))


sample = dataset[0]

print("Image shape:", sample["image"].shape)
print("Label:", sample["label"])
print("Path:", sample["path"])