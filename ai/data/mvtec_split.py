from pathlib import Path
import random


def split_normal_images(
    root_dir,
    category="bottle",
    validation_ratio=0.20,
    seed=42,
):

    root_dir = Path(root_dir)

    train_good_dir = (
        root_dir
        / category
        / "train"
        / "good"
    )

    image_paths = sorted(
        train_good_dir.glob("*.png")
    )

    print(
        "Total normal images:",
        len(image_paths)
    )

    random.seed(seed)

    random.shuffle(image_paths)

    validation_size = int(
        len(image_paths)
        * validation_ratio
    )

    validation_images = image_paths[
        :validation_size
    ]

    memory_images = image_paths[
        validation_size:
    ]

    print(
        "Memory-bank images:",
        len(memory_images)
    )

    print(
        "Validation images:",
        len(validation_images)
    )

    return (
        memory_images,
        validation_images,
    )