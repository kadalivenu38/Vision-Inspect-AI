from pathlib import Path

import numpy as np
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

        self.category_dir = (
            self.root_dir / category
        )

        self.test_dir = (
            self.category_dir / "test"
        )

        self.ground_truth_dir = (
            self.category_dir / "ground_truth"
        )

        self.samples = []

        self._build_samples()

    # ========================================================
    # Build sample list
    # ========================================================

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

                # --------------------------------------------
                # Normal image
                # --------------------------------------------

                if defect_type == "good":

                    self.samples.append({
                        "image_path": image_path,
                        "label": 0,
                        "defect_type": "good",
                        "mask_path": None,
                    })

                # --------------------------------------------
                # Defective image
                # --------------------------------------------

                else:

                    mask_path = (
                        self.ground_truth_dir
                        / defect_type
                        / f"{image_path.stem}_mask.png"
                    )

                    self.samples.append({
                        "image_path": image_path,
                        "label": 1,
                        "defect_type": defect_type,
                        "mask_path": mask_path,
                    })

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = None

        if sample["mask_path"] is not None:

            mask = Image.open(
                sample["mask_path"]
            ).convert("L")

            # If mask transform exists, use it
            if self.mask_transform:

                mask = self.mask_transform(
                    mask
                )

            # Otherwise convert directly to Tensor
            else:

                mask = torch.from_numpy(
                    np.array(mask)
                ).float()

                mask = mask / 255.0

                # Add channel dimension
                #
                # [H, W]
                #
                # becomes
                #
                # [1, H, W]

                if mask.ndim == 2:

                    mask = mask.unsqueeze(0)

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "image": image,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "mask": mask,
            "path": str(
                sample["image_path"]
            ),
        }