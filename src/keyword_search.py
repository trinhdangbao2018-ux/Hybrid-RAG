import bm25s

from src.types import Chunk


class KeywordRetriever:
    """BM25 ranking over chunk texts."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = []

        for chunk in chunks:
            texts.append(chunk.text)
        corpus_tokens = bm25s.tokenize(texts, stopwords="en")       # Tokenize every text to token form, the stopwords="en" mean that the machine skip every popular words that don't bring a lot of context like "is", "are", "am", "to", "in"

        self.bm25 = bm25s.BM25()                    # BM25 is a built in class in python. bm25s.BM25(), if you want to use BM25 method still have to .bm25.method EX: self.bm25.retrieve(...)
        self.bm25.index(corpus_tokens)              # this task analise every words, there position, and how many time it appear

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        top_n = min(top_n, len(self.chunks))                # Take every chunks in case the input is higher for top_n
        q_tokens = bm25s.tokenize(query, stopwords="en")       # Tokenize query to token form, the stopwords="en" mean that the machine skip every popular words that don't bring a lot of context like "is", "are", "am", "to", "in"
        ids, scores = self.bm25.retrieve(q_tokens, k=top_n)     # Retrieve keyword 
        return [(self.chunks[i].chunk_id, float(s)) for i, s in zip(ids[0], scores[0])]