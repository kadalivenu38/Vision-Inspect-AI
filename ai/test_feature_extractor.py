import torch

from models.feature_extractor import (
    ResNet18FeatureExtractor,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# Fake image
images = torch.randn(
    2,
    3,
    256,
    256,
).to(device)


with torch.no_grad():

    features = model(images)


print("Input:", images.shape)

for name, feature in features.items():

    print(
        f"{name}: {feature.shape}"
    )