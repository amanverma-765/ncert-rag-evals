"""Local sentence embeddings for the vector arms.

Runs on CPU, so re-running the eval costs nothing. Vectors are L2-normalized
at encode time, making cosine similarity a plain dot product later.
"""

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims
DIMS = 384

# BGE is trained with an asymmetric prefix: queries get it, passages do not
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@cache
def _model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL)


def encode_documents(texts: list[str], batch: int = 64) -> np.ndarray:
    return (
        _model()
        .encode(
            texts,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        .astype(np.float32)
    )


def encode_query(text: str) -> np.ndarray:
    return (
        _model()
        .encode(
            _QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        .astype(np.float32)
    )
