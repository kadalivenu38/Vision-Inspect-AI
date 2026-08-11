import torch
import torch.nn.functional as F


def extract_patch_embeddings(
    features,
    target_size=(32, 32),
):
    """
    Convert layer2 and layer3 feature maps
    into a common patch representation.

    layer2:
        [B, 128, 32, 32]

    layer3:
        [B, 256, 16, 16]

    Output:
        [B, 1024, 384]

    where:
        1024 = 32 * 32 spatial patches
        384  = 128 + 256 feature dimensions
    """

    layer2 = features["layer2"]

    layer3 = features["layer3"]

    # Resize layer3 spatial dimensions
    # from 16x16 -> 32x32
    layer3 = F.interpolate(
        layer3,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    # layer2:
    # [B, 128, 32, 32]
    #
    # layer3:
    # [B, 256, 32, 32]
    #
    # concatenate channels
    combined = torch.cat(
        [layer2, layer3],
        dim=1,
    )

    # [B, 384, 32, 32]
    B, C, H, W = combined.shape

    # Convert spatial locations into patches
    #
    # [B, C, H, W]
    #       ↓
    # [B, H, W, C]
    combined = combined.permute(
        0, 2, 3, 1
    )

    # [B, H*W, C]
    embeddings = combined.reshape(
        B,
        H * W,
        C,
    )

    # Normalize each feature vector
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=2,
    )

    return embeddings