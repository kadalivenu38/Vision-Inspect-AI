from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class MVTecTrainDataset(Dataset):

    def __init__(
        self,
        root_dir,
        category,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.transform = transform

        self.image_dir = (
            self.root_dir
            / category
            / "train"
            / "good"
        )

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": 0,
            "path": str(image_path),
        }