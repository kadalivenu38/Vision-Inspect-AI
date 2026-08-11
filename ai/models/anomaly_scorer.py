import torch
import torch.nn.functional as F


class NearestNeighborAnomalyScorer:

    def __init__(
        self,
        memory_bank,
        device,
    ):
        self.device = device

        self.memory_bank = memory_bank.to(
            device
        )

        self.memory_bank = F.normalize(
            self.memory_bank,
            p=2,
            dim=1,
        )

    def score(
        self,
        embeddings,
    ):
        """
        embeddings:
            [B, 1024, 384]

        Returns:
            image_scores:
                [B]

            anomaly_maps:
                [B, 32, 32]
        """

        B, N, D = embeddings.shape

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=2,
        )

        image_scores = []

        anomaly_maps = []

        for b in range(B):

            patches = embeddings[b]

            # --------------------------------------
            # Cosine similarity
            # --------------------------------------

            similarity = torch.matmul(
                patches,
                self.memory_bank.T,
            )

            # --------------------------------------
            # Nearest normal patch
            # --------------------------------------

            max_similarity = similarity.max(
                dim=1
            ).values

            # --------------------------------------
            # Convert similarity to distance
            # --------------------------------------

            distances = 1 - max_similarity

            # --------------------------------------
            # Image-level score
            # --------------------------------------

            image_score = distances.max()

            image_scores.append(
                image_score
            )

            # --------------------------------------
            # Patch-level anomaly map
            # --------------------------------------

            anomaly_map = distances.reshape(
                32,
                32,
            )

            anomaly_maps.append(
                anomaly_map
            )

        image_scores = torch.stack(
            image_scores
        )

        anomaly_maps = torch.stack(
            anomaly_maps
        )

        return (
            image_scores,
            anomaly_maps,
        )