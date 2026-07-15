from typing import Protocol, Callable

from src.config import CONFIG
from src.hybrid_search import metadata_filter, rrf_fuse   
from src.types import Chunk


class Retriever(Protocol):
    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        ...

class HybridRetriever:
    def __init__(self, retrievers, chunks: list[Chunk]):
        self.retrievers = retrievers
        self.chunk_by_id = {} # Create a lib
        for chunk in chunks:
            self.chunk_by_id[chunk.chunk_id] = chunk  # Store chunk_id: chunk in the lib

    def search(self, query: str, top_n: int = CONFIG.retrieve_n, predicate: Callable[[Chunk], bool] | None = None) -> list[tuple[str, float]]:

        scoreboard = list()     # Create a temporary list
        for r in self.retrievers:       # Only vector search and keyword search :0
            ans = r.search(query, top_n)        # Take the output of each type of search, they search for an lib of (chunk_id, score)
            scoreboard.append(ans)              # Append the score board, so the scoreboard look like this ((chunk_id, score_board), (chunk_id_2,score_board_2) ,...)
        if predicate is not None:               # Skip chunk without metadata
            sortboard = []
            for board in scoreboard:
                clean_board = metadata_filter(board, self.chunk_by_id, predicate)  # Take only the chunk with metadata     
                sortboard.append(clean_board)
            scoreboard = sortboard
        return rrf_fuse(scoreboard, k=CONFIG.rrf_k)[:top_n]
