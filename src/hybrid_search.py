from typing import Callable
from src.types import Chunk

def metadata_filter(ranked: list[tuple[str, float]], chunks_by_id: dict[str, Chunk], predicate: Callable[[Chunk], bool]) -> list[tuple[str, float]]:
    ans = []                            
    for cid, s in ranked:                   
        chunk = chunks_by_id[cid]           
        if predicate(chunk):                
            ans.append((cid, s))        
    return ans                        

#Reciprocal Rank Fusion
def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009): score = sum of 1/(k + rank)."""

    chunk_scores = dict()       
    for ranking in rankings:
        for rank, (chunk_id, score) in enumerate(ranking, start = 1):           # start = 1 change the first index from zero to one, it count 1->2->3->4 rather than 0->1->2->3
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 1/ (k + rank)          # Add chunk_id and chunk_score
            else:
                chunk_scores[chunk_id] += 1/ (k +rank)          # Plus chunk_score
    return sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True) 

