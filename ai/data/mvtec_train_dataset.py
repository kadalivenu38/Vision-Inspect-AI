from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class MVTecTrainDataset(Dataset):

    def __init__(
        self,
        root_dir,
        category,
        transform=None,
        image_paths=None,
    ):

        self.root_dir = Path(root_dir)
        self.category = category
        self.transform = transform

        self.category_dir = (
            self.root_dir / category
        )

        self.train_dir = (
            self.category_dir / "train"
        )

        self.good_dir = (
            self.train_dir / "good"
        )

        self.samples = []

        # If specific image paths are provided,
        # use only those images.
        if image_paths is not None:

            for image_path in image_paths:

                self.samples.append({
                    "image_path": Path(image_path),
                    "label": 0,
                })

        # Otherwise load all good training images.
        else:

            self._build_samples()

    def _build_samples(self):

        image_paths = sorted(
            self.good_dir.glob("*.png")
        )

        for image_path in image_paths:

            self.samples.append({
                "image_path": image_path,
                "label": 0,
            })

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "path": str(sample["image_path"]),
        }