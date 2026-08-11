import torch
import torch.nn.functional as F


def greedy_coreset_sampling(
    embeddings,
    sampling_ratio=0.05,
):
    """
    Greedy farthest-point coreset sampling.

    Input:
        [N, D]

    Output:
        [M, D]
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must have shape [N, D]"
        )

    embeddings = embeddings.float()

    # Normalize feature vectors
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    N = embeddings.shape[0]

    target_size = max(
        1,
        int(N * sampling_ratio),
    )

    print(
        "Total embeddings:",
        N,
    )

    print(
        "Target coreset size:",
        target_size,
    )

    # ------------------------------------------------
    # Initial random point
    # ------------------------------------------------

    first_index = torch.randint(
        0,
        N,
        (1,),
    ).item()

    selected_indices = [
        first_index
    ]

    selected = embeddings[
        first_index
    ]

    # ------------------------------------------------
    # Similarity to first point
    # ------------------------------------------------

    similarity = torch.matmul(
        embeddings,
        selected,
    )

    distances = 1 - similarity

    # ------------------------------------------------
    # Greedy sampling
    # ------------------------------------------------

    for i in range(
        1,
        target_size,
    ):

        next_index = torch.argmax(
            distances
        ).item()

        selected_indices.append(
            next_index
        )

        selected = embeddings[
            next_index
        ]

        similarity = torch.matmul(
            embeddings,
            selected,
        )

        new_distances = (
            1 - similarity
        )

        distances = torch.minimum(
            distances,
            new_distances,
        )

        if (
            i % 1000 == 0
            or i == target_size - 1
        ):
            print(
                f"Selected {i + 1}/"
                f"{target_size}"
            )

    selected_indices = torch.tensor(
        selected_indices,
        dtype=torch.long,
    )

    return embeddings[
        selected_indices
    ]