import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

import matplotlib.pyplot as plt


class PatchCoreLocalizer:

    def __init__(
        self,
        pixel_threshold=None,
    ):

        self.pixel_threshold = (
            pixel_threshold
        )

    # ========================================================
    # Generate Binary Mask
    # ========================================================

    def create_binary_mask(
        self,
        anomaly_map,
    ):

        if self.pixel_threshold is None:

            raise ValueError(
                "pixel_threshold has not been "
                "validated yet."
            )

        if torch.is_tensor(
            anomaly_map
        ):

            anomaly_map = (
                anomaly_map
                .detach()
                .cpu()
                .numpy()
            )

        binary_mask = (
            anomaly_map
            >= self.pixel_threshold
        )

        return binary_mask.astype(
            np.uint8
        )

    # ========================================================
    # Bounding Box
    # ========================================================

    def get_bounding_box(
        self,
        binary_mask,
    ):

        ys, xs = np.where(
            binary_mask > 0
        )

        if len(xs) == 0:

            return None

        x_min = int(
            xs.min()
        )

        y_min = int(
            ys.min()
        )

        x_max = int(
            xs.max()
        )

        y_max = int(
            ys.max()
        )

        return {

            "x": x_min,

            "y": y_min,

            "width": (
                x_max - x_min + 1
            ),

            "height": (
                y_max - y_min + 1
            ),

        }

    # ========================================================
    # Generate Overlay
    # ========================================================

    def create_overlay(
        self,
        image,
        anomaly_map,
        output_path=None,
    ):

        if isinstance(
            image,
            Image.Image,
        ):

            image_np = np.asarray(
                image.convert("RGB")
            ) / 255.0

        elif torch.is_tensor(
            image
        ):

            image_np = (
                image
                .detach()
                .cpu()
                .permute(
                    1,
                    2,
                    0,
                )
                .numpy()
            )

        else:

            image_np = np.asarray(
                image
            )

            if image_np.max() > 1:

                image_np = (
                    image_np / 255.0
                )

        if torch.is_tensor(
            anomaly_map
        ):

            anomaly_map = (
                anomaly_map
                .detach()
                .cpu()
                .numpy()
            )

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            image_np
        )

        plt.imshow(
            anomaly_map,
            alpha=0.5,
        )

        plt.axis("off")

        plt.tight_layout(
            pad=0
        )

        if output_path is not None:

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
                pad_inches=0,
            )

        plt.close()

    # ========================================================
    # Complete Localization
    # ========================================================

    def localize(
        self,
        anomaly_map,
    ):

        binary_mask = (
            self.create_binary_mask(
                anomaly_map
            )
        )

        bounding_box = (
            self.get_bounding_box(
                binary_mask
            )
        )

        return {

            "pixel_threshold": (
                self.pixel_threshold
            ),

            "binary_mask": binary_mask,

            "bounding_box": bounding_box,

        }