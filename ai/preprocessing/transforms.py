from torchvision import transforms


# ============================================================
# Image Transform
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ============================================================
# Test Image Transform
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ============================================================
# Mask Transform
# ============================================================

mask_transform = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=transforms.InterpolationMode.NEAREST,
    ),
    transforms.ToTensor(),
])