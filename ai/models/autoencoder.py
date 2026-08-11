import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        encoded = self.encoder(x)

        reconstructed = self.decoder(encoded)

        return reconstructed