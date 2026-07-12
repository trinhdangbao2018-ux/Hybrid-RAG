import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CONFIG

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:                       # lazy: load once, reuse forever
        _model = SentenceTransformer(CONFIG.embed_model)
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed chunk texts into an (N, 384) unit-length matrix."""
    return get_model().encode_document(texts, normalize_embeddings=True,
                                       show_progress_bar=True)


def embed_query(text: str) -> np.ndarray:
    """Embed one query into a (384,) unit vector — WITH the BGE query prompt."""
    return get_model().encode_query(text, prompt=CONFIG.query_prefix,
                                    normalize_embeddings=True)