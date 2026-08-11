from data.mvtec_split import split_normal_images
from data.mvtec_train_dataset import MVTecTrainDataset

from preprocessing.transforms import train_transform


DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"


# --------------------------------------------------
# Step 1: Split the 209 normal images
# --------------------------------------------------

memory_images, validation_images = split_normal_images(
    root_dir=DATASET_ROOT,
    category="bottle",
)


# --------------------------------------------------
# Step 2: Create dataset using only 168 images
# --------------------------------------------------

memory_dataset = MVTecTrainDataset(
    root_dir=DATASET_ROOT,
    category="bottle",
    transform=train_transform,
    image_paths=memory_images,
)


# --------------------------------------------------
# Step 3: Print information
# --------------------------------------------------

print()
print("=" * 50)
print("Memory Dataset Test")
print("=" * 50)

print(
    "Total normal images:",
    len(memory_images) + len(validation_images)
)

print(
    "Memory images:",
    len(memory_images)
)

print(
    "Validation images:",
    len(validation_images)
)

print(
    "Dataset length:",
    len(memory_dataset)
)


sample = memory_dataset[0]

print(
    "Image shape:",
    sample["image"].shape
)

print(
    "Label:",
    sample["label"]
)

print(
    "Path:",
    sample["path"]
)