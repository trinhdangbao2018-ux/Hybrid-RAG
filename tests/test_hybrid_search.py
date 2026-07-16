from src.hybrid_search import rrf_fuse


def test_rrf_rewards_agreement():
    v = [("c1", .9), ("c2", .8)]        # vector ranking
    kw = [("c2", 5.), ("c3", 4.)]       # keyword ranking
    fused = rrf_fuse([v, kw], k=60)
    assert fused[0][0] == "c2"          # ranked by BOTH lists → wins
    assert len(fused) == 3              # union, no duplicates