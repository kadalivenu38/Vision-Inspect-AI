import torch
from torch.utils.data import DataLoader


class MemoryBankBuilder:

    def __init__(
        self,
        feature_extractor,
        embedding_function,
        device,
        batch_size=8,
    ):
        self.feature_extractor = feature_extractor
        self.embedding_function = embedding_function
        self.device = device
        self.batch_size = batch_size

    def build(self, dataset):

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        memory_bank = []

        self.feature_extractor.eval()

        with torch.no_grad():

            for batch_index, batch in enumerate(loader):

                images = batch["image"].to(
                    self.device,
                    non_blocking=True,
                )

                features = self.feature_extractor(
                    images
                )

                embeddings = self.embedding_function(
                    features
                )

                # embeddings:
                # [B, 1024, 384]

                B, N, C = embeddings.shape

                # Convert:
                #
                # [B, N, C]
                #
                # into:
                #
                # [B*N, C]

                embeddings = embeddings.reshape(
                    B * N,
                    C,
                )

                memory_bank.append(
                    embeddings.cpu()
                )

                print(
                    f"Processed batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

        memory_bank = torch.cat(
            memory_bank,
            dim=0,
        )

        return memory_bank