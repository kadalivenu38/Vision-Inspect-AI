from torchvision import transforms


# ==================================================
# Image transforms
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==================================================
# Mask transform
# ==================================================

mask_transform = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=transforms.InterpolationMode.NEAREST,
    ),
    transforms.ToTensor(),
])