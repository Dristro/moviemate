"""
Embedding model implementation.

Basic model stats:
    Embedding size: 768
    Max context length: 384
    Model size: ~420MB
"""

import torch
import warnings

from sentence_transformers import SentenceTransformer
from utils import auto_device_map


DEVICE = auto_device_map()

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=DEVICE.type,
)
model.to("cpu")


@torch.inference_mode()
def embed(text: list[str]) -> torch.Tensor:
    """
    Return's embeddings for each text item in list.
    Please provide a list of string.

    Args:
        text (list[str]): list of texts

    Returns:
        torch.Tensor of shape [B, 768]
    """
    model.to(DEVICE)
    output = model.encode(
        text,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    model.to("cpu")

    return output


@torch.inference_mode()
def get_best(
    query_embedding: torch.Tensor,
    embeddings: torch.Tensor,
    k: int,
    thresh: float,
) -> int:
    """
    Get top-k number of top similarities.

    Args:
        query_embedding (Tensor): query embedding
        embeddings (Tensor): all embeddings
        k (int): the number of matches to return
        thresh (float): minimum similarity score to return
    Returns:
        best (int): index of the most similar embedding
        from embeddings
    """
    warnings.warn(
        "This function will be depricated in the next iteration of this "
        "system. Please use EMbeddingController.get_topk()"
    )
    query_embedding = query_embedding.to(DEVICE)
    embeddings = embeddings.to(DEVICE)

    model.to(DEVICE)
    similarities = model.similarity(query_embedding, embeddings).squeeze()
    model.to("cpu")

    best = int(similarities.argmax().item())

    return best
