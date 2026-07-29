# Why we need this step?
# RRF function only know the rank agreement, it don't care about the quality of the chunk, it only care about 1 value, there rank in booth keyword search and vector search. We have
# to have rrf function because we cannot encoding (go through transformer block) every chunk because it an (n,384) matrix and it take a lot of time
# to go through every of that. So we only let it go through transformer with only high quality chunks to make the machine faster and get a better result
from sentence_transformers import CrossEncoder   # used for: scores a (query, text) PAIR read together — the only model here that sees question and chunk in one sequence

from src.config import CONFIG                    # used for: rerank_model name, top_k
from src.types import Chunk                      # used for: what we take in and hand back, unmodified


class Reranker:
    _model: CrossEncoder | None = None   # Just mean the CrossEncoder can be a Crossencoder object or none

    def __init__(self, model_name: str = CONFIG.rerank_model) -> None:
        self.model_name = model_name           # kept only as a record of what was requested
        if Reranker._model is None:            # still empty, so we are the first caller and we pay
            Reranker._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]      
        source = Reranker._model.predict(pairs)       # The model take query and text as an input and make them go through transition
        order = sorted(range(len(chunks)), key=lambda i: source[i], reverse=True)
        return [chunks[i] for i in order[:top_k]]

        