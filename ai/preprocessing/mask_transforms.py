from torchvision import transforms


IMAGE_SIZE = 256


mask_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.NEAREST,
    ),
    transforms.ToTensor(),
])