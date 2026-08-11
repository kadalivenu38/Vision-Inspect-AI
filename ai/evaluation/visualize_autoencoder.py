import matplotlib.pyplot as plt
import torch

from data.mvtec_test_dataset import MVTecTestDataset
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import test_transform
from preprocessing.mask_transforms import mask_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "/content/drive/MyDrive/MVTec_Dataset"

MODEL_PATH = (
    "/content/Vision-Inspect-AI/"
    "ai/models/autoencoder_bottle.pth"
)

CATEGORY = "bottle"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = MVTecTestDataset(
    root_dir=DATASET_ROOT,
    category=CATEGORY,
    transform=test_transform,
    mask_transform=mask_transform,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ConvAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Find defective sample
# --------------------------------------------------

defect_index = None

for index, sample in enumerate(dataset):

    if sample["label"] == 1:

        defect_index = index
        break


sample = dataset[defect_index]

image = sample["image"]


# --------------------------------------------------
# Model inference
# --------------------------------------------------

with torch.no_grad():

    input_tensor = (
        image
        .unsqueeze(0)
        .to(device)
    )

    reconstructed = model(
        input_tensor
    )


# --------------------------------------------------
# Calculate anomaly map
# --------------------------------------------------

anomaly_map = (
    input_tensor - reconstructed
) ** 2

anomaly_map = anomaly_map.mean(
    dim=1
)

anomaly_map = (
    anomaly_map
    .squeeze(0)
    .cpu()
    .numpy()
)


# --------------------------------------------------
# Prepare images
# --------------------------------------------------

original = (
    image
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

reconstructed_image = (
    reconstructed
    .squeeze(0)
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

ground_truth = (
    sample["mask"]
    .squeeze(0)
    .cpu()
    .numpy()
)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4),
)


axes[0].imshow(original)

axes[0].set_title(
    f"Original\n{sample['defect_type']}"
)

axes[0].axis("off")


axes[1].imshow(
    reconstructed_image
)

axes[1].set_title(
    "Reconstruction"
)

axes[1].axis("off")


axes[2].imshow(
    anomaly_map
)

axes[2].set_title(
    "Anomaly Map"
)

axes[2].axis("off")


axes[3].imshow(
    ground_truth
)

axes[3].set_title(
    "Ground Truth"
)

axes[3].axis("off")


plt.tight_layout()

plt.show()