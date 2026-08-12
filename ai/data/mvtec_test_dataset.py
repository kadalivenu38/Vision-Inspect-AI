from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torch


class MVTecTestDataset(Dataset):

    def __init__(
        self,
        root_dir,
        category,
        transform=None,
        mask_transform=None,
        load_masks=True,
    ):
        self.root_dir = Path(root_dir)
        self.category = category

        self.transform = transform
        self.mask_transform = mask_transform

        # --------------------------------------------------
        # If False:
        #
        # Dataset will NOT require ground-truth masks.
        #
        # Useful for:
        # - threshold calculation
        # - image-level anomaly detection
        # - inference
        #
        # If True:
        #
        # Real masks are loaded for defective images.
        #
        # Useful for:
        # - pixel-level evaluation
        # - anomaly-map evaluation
        # --------------------------------------------------

        self.load_masks = load_masks

        self.samples = []

        self._build_samples()

    # ==========================================================
    # Build test samples
    # ==========================================================

    def _build_samples(self):

        test_dir = (
            self.root_dir
            / self.category
            / "test"
        )

        ground_truth_dir = (
            self.root_dir
            / self.category
            / "ground_truth"
        )

        if not test_dir.exists():

            raise FileNotFoundError(
                f"Test directory not found: {test_dir}"
            )

        # ------------------------------------------------------
        # Iterate through defect categories
        # ------------------------------------------------------

        for defect_dir in sorted(test_dir.iterdir()):

            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name

            # ==================================================
            # Good / normal images
            # ==================================================

            if defect_type == "good":

                for image_path in sorted(
                    defect_dir.glob("*.png")
                ):

                    self.samples.append({

                        "image_path": image_path,

                        "label": 0,

                        "defect_type": "good",

                        "mask_path": None,

                    })

            # ==================================================
            # Defective images
            # ==================================================

            else:

                mask_dir = (
                    ground_truth_dir
                    / defect_type
                )

                for image_path in sorted(
                    defect_dir.glob("*.png")
                ):

                    mask_path = (
                        mask_dir
                        / f"{image_path.stem}_mask.png"
                    )

                    # ------------------------------------------------
                    # Only require the mask when masks are actually
                    # requested.
                    # ------------------------------------------------

                    if self.load_masks:

                        if not mask_path.exists():

                            raise FileNotFoundError(
                                f"Mask not found: {mask_path}"
                            )

                    self.samples.append({

                        "image_path": image_path,

                        "label": 1,

                        "defect_type": defect_type,

                        "mask_path": (
                            mask_path
                            if mask_path.exists()
                            else None
                        ),

                    })

    # ==========================================================
    # Dataset length
    # ==========================================================

    def __len__(self):

        return len(self.samples)

    # ==========================================================
    # Get sample
    # ==========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["image_path"]

        label = sample["label"]

        defect_type = sample["defect_type"]

        mask_path = sample["mask_path"]

        # ======================================================
        # Load image
        # ======================================================

        image = Image.open(
            image_path
        ).convert("RGB")

        # ======================================================
        # Apply image transform
        # ======================================================

        if self.transform is not None:

            image = self.transform(image)

        # ======================================================
        # Image must be Tensor
        # ======================================================

        if not torch.is_tensor(image):

            raise TypeError(
                "Image must be a Tensor after "
                "applying transform."
            )

        height = image.shape[-2]

        width = image.shape[-1]

        # ======================================================
        # NORMAL IMAGE
        # ======================================================

        if label == 0:

            mask = torch.zeros(
                (1, height, width),
                dtype=torch.float32,
            )

        # ======================================================
        # DEFECTIVE IMAGE
        # ======================================================

        else:

            # --------------------------------------------------
            # If masks are disabled:
            #
            # We don't need the ground-truth mask.
            #
            # Return a dummy zero mask so the dataset structure
            # remains consistent.
            # --------------------------------------------------

            if not self.load_masks:

                mask = torch.zeros(
                    (1, height, width),
                    dtype=torch.float32,
                )

            else:

                if self.mask_transform is None:

                    raise ValueError(
                        "mask_transform must be provided "
                        "when load_masks=True."
                    )

                if mask_path is None:

                    raise FileNotFoundError(
                        f"Mask path is missing for: "
                        f"{image_path}"
                    )

                mask = Image.open(
                    mask_path
                ).convert("L")

                mask = self.mask_transform(
                    mask
                )

                # ------------------------------------------------
                # Convert mask into binary values.
                #
                # 0 = normal
                # 1 = defective
                # ------------------------------------------------

                mask = (
                    mask > 0.5
                ).float()

        # ======================================================
        # Final safety check
        # ======================================================

        if image.shape[-2:] != mask.shape[-2:]:

            raise ValueError(
                f"Image and mask size mismatch: "
                f"image={image.shape}, "
                f"mask={mask.shape}, "
                f"path={image_path}"
            )

        # ======================================================
        # Return sample
        # ======================================================

        return {

            "image": image,

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "mask": mask,

            "defect_type": defect_type,

            "path": str(image_path),
        }