import numpy as np

from src.embedding import embed_documents, embed_query


def test_shapes():  # (N, 384) for documents, (384,) for a single query
    docs = embed_documents(["Dogs eat kibble.", "Cats drink milk."])
    q = embed_query("what do dogs eat?")
    assert docs.shape == (2, 384)
    assert q.shape == (384,)


def test_unit_length():  # normalize_embeddings=True must give norm ~ 1.0
    docs = embed_documents(["Dogs eat kibble.", "Cats drink milk."])
    q = embed_query("what do dogs eat?")
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-3)


def test_related_beats_unrelated():  # embeddings must capture meaning, not just shape
    q = embed_query("what do dogs eat?")
    docs = embed_documents(["Dogs eat kibble.",
                            "The stock market fell sharply today."])
    related, unrelated = docs @ q       # cosine scores (unit vectors -> dot = cosine)
    assert related > unrelated
