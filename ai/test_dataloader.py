from torch.utils.data import DataLoader

from preprocessing.transforms import train_transform
from data.mvtec_dataset import MVTecTrainDataset


dataset = MVTecTrainDataset(
    root_dir="dataset/mvtec",
    category="bottle",
    transform=train_transform,
)


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


batch = next(iter(loader))


print("Batch image shape:")
print(batch["image"].shape)

print("\nBatch labels:")
print(batch["label"])

print("\nBatch paths:")
for path in batch["path"]:
    print(path)