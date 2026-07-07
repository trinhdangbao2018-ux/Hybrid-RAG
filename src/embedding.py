import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy-load the model once and reuse it — loading is slow, encoding is fast.
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into an (N, 384) matrix."""
    return get_model().encode(texts, normalize_embeddings=True)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query into a (384,) vector."""
    return get_model().encode(text, normalize_embeddings=True)