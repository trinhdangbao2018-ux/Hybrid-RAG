import numpy as np

from src.embedding import embed_query
from src.types import Chunk


class VectorRetriever:
    """Exact cosine search over a normalized (N, 384) matrix."""

    def __init__(self, chunks: list[Chunk], matrix: np.ndarray):
        self.chunks = chunks
        self.matrix = matrix

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        q = embed_query(query)
        scores = self.matrix @ q                 # cosine, because normalized
        top_n = min(top_n, len(scores))
        idx = np.argpartition(scores, -top_n)[-top_n:]      # top-n, unordered
        idx = idx[np.argsort(scores[idx])[::-1]]            # order those n
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in idx]