import torch
import torch.nn.functional as F
from PIL import Image

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)

from models.patch_embedding import (
    extract_patch_embeddings,
)

from preprocessing.transforms import (
    test_transform,
)


class PatchCorePredictor:

    def __init__(
        self,
        coreset_path,
        threshold=0.124763,
        device=None,
    ):

        # ====================================================
        # Device
        # ====================================================

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        self.threshold = float(
            threshold
        )

        # ====================================================
        # Feature Extractor
        # ====================================================

        self.feature_extractor = (
            ResNet18FeatureExtractor()
            .to(self.device)
        )

        self.feature_extractor.eval()

        # ====================================================
        # Load Coreset
        # ====================================================

        self.coreset = torch.load(
            coreset_path,
            map_location=self.device,
        )

        self.coreset = (
            self.coreset
            .float()
            .to(self.device)
        )

        self.coreset = F.normalize(
            self.coreset,
            p=2,
            dim=1,
        )

        print()
        print("=" * 60)
        print("PatchCore Predictor Initialized")
        print("=" * 60)

        print(
            "Device:",
            self.device,
        )

        print(
            "Coreset:",
            self.coreset.shape,
        )

        print(
            "Threshold:",
            self.threshold,
        )

    # ========================================================
    # Preprocess Image
    # ========================================================

    def preprocess(
        self,
        image,
    ):

        if isinstance(
            image,
            str,
        ):

            image = Image.open(
                image
            ).convert("RGB")

        elif not isinstance(
            image,
            Image.Image,
        ):

            raise TypeError(
                "Input must be a PIL Image "
                "or image path."
            )

        image_tensor = test_transform(
            image
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        return image_tensor

    # ========================================================
    # Predict
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        image,
    ):

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        input_tensor = self.preprocess(
            image
        )

        # ----------------------------------------------------
        # Feature Extraction
        # ----------------------------------------------------

        features = (
            self.feature_extractor(
                input_tensor
            )
        )

        # ----------------------------------------------------
        # Patch Embeddings
        # ----------------------------------------------------

        embeddings = (
            extract_patch_embeddings(
                features
            )
        )

        # Expected:
        #
        # [1, 1024, 384]

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=2,
        )

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        image_embeddings = (
            embeddings[0]
        )

        # ----------------------------------------------------
        # Compare with Coreset
        # ----------------------------------------------------

        similarity = torch.matmul(
            image_embeddings,
            self.coreset.T,
        )

        # ----------------------------------------------------
        # Nearest normal patch
        # ----------------------------------------------------

        max_similarity = (
            similarity
            .max(dim=1)
            .values
        )

        # ----------------------------------------------------
        # Patch anomaly scores
        # ----------------------------------------------------

        patch_scores = (
            1.0 - max_similarity
        )

        # ----------------------------------------------------
        # Image anomaly score
        # ----------------------------------------------------

        anomaly_score = (
            patch_scores.max()
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        is_anomaly = (
            anomaly_score
            >= self.threshold
        )

        prediction = (
            "DEFECT"
            if is_anomaly
            else "NORMAL"
        )

        # ----------------------------------------------------
        # Patch anomaly map
        # ----------------------------------------------------

        anomaly_map = (
            patch_scores
            .reshape(
                32,
                32,
            )
        )

        # ----------------------------------------------------
        # Resize to image resolution
        # ----------------------------------------------------

        anomaly_map_256 = F.interpolate(
            anomaly_map
            .unsqueeze(0)
            .unsqueeze(0),

            size=(256, 256),

            mode="bilinear",

            align_corners=False,
        )

        anomaly_map_256 = (
            anomaly_map_256[
                0,
                0
            ]
        )

        return {

            "prediction": prediction,

            "is_anomaly": bool(
                is_anomaly.item()
            ),

            "anomaly_score": float(
                anomaly_score.item()
            ),

            "threshold": self.threshold,

            "patch_scores": (
                patch_scores
                .detach()
                .cpu()
            ),

            "anomaly_map": (
                anomaly_map_256
                .detach()
                .cpu()
            ),

        }