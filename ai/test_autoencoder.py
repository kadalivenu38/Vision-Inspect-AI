import torch

from models.autoencoder import ConvAutoencoder


model = ConvAutoencoder()

x = torch.randn(
    2,
    3,
    256,
    256,
)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)