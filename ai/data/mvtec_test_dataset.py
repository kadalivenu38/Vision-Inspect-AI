from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class MVTecTestDataset(Dataset):

    def __init__(
        self,
        root_dir,
        category,
        transform=None,
        mask_transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.transform = transform
        self.mask_transform = mask_transform

        self.category_dir = self.root_dir / category

        self.test_dir = (
            self.category_dir / "test"
        )

        self.ground_truth_dir = (
            self.category_dir / "ground_truth"
        )

        if not self.category_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: "
                f"{self.category_dir}"
            )

        if not self.test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: "
                f"{self.test_dir}"
            )

        if not self.ground_truth_dir.exists():
            raise FileNotFoundError(
                f"Ground truth directory not found: "
                f"{self.ground_truth_dir}"
            )

        self.samples = []

        self._build_samples()

    def _build_samples(self):

        for defect_dir in sorted(
            self.test_dir.iterdir()
        ):

            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name

            image_paths = sorted(
                defect_dir.glob("*.png")
            )

            for image_path in image_paths:

                # --------------------------------
                # Normal image
                # --------------------------------

                if defect_type == "good":

                    self.samples.append({
                        "image_path": image_path,
                        "label": 0,
                        "defect_type": "good",
                        "mask_path": None,
                        "has_mask": False,
                    })

                # --------------------------------
                # Defective image
                # --------------------------------

                else:

                    mask_path = (
                        self.ground_truth_dir
                        / defect_type
                        / f"{image_path.stem}_mask.png"
                    )

                    if not mask_path.exists():
                        raise FileNotFoundError(
                            f"Ground truth mask not found: "
                            f"{mask_path}"
                        )

                    self.samples.append({
                        "image_path": image_path,
                        "label": 1,
                        "defect_type": defect_type,
                        "mask_path": mask_path,
                        "has_mask": True,
                    })

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # --------------------------------
        # Load image
        # --------------------------------

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # --------------------------------
        # Load mask
        # --------------------------------

        if sample["has_mask"]:

            mask = Image.open(
                sample["mask_path"]
            ).convert("L")

            if self.mask_transform:
                mask = self.mask_transform(mask)

            else:
                raise ValueError(
                    "mask_transform must be provided "
                    "for defective samples."
                )

        else:

            # Normal image has no defect.
            # Therefore its ground-truth mask
            # is completely zero.

            height = image.shape[-2]
            width = image.shape[-1]

            mask = torch.zeros(
                (1, height, width),
                dtype=torch.float32,
            )

        # --------------------------------
        # Return sample
        # --------------------------------

        return {
            "image": image,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "mask": mask,
            "has_mask": sample["has_mask"],
            "path": str(sample["image_path"]),
        }