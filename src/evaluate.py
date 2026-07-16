import json

from src.config import CONFIG
from src.embedding import embed_documents
from src.keyword_search import KeywordRetriever
from src.loader import load_and_chunk
from src.retriever import HybridRetriever
from src.types import Chunk
from src.vector_search import VectorRetriever


def normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces, lowercase."""
    return " ".join(s.split()).lower()

def first_relevant_rank(result_ids: list[str],
                        chunks_by_id: dict[str, Chunk],
                        expect_doc: str,
                        expect_phrase: str) -> int | None:
    """1-based rank of the first hit, or None. Hit = right doc AND phrase in text."""
    needle = normalize(expect_phrase)           # normalize once, not per iteration
    for rank, cid in enumerate(result_ids, start=1):
        chunk = chunks_by_id[cid]               # id -> full Chunk object
        if chunk.doc_id == expect_doc and needle in normalize(chunk.text):
            return rank                         # first hit wins
    return None                                 # whole list walked, no hit


def eval_mode(search_fn, chunks_by_id: dict[str, Chunk], golden: list[dict], k: int = CONFIG.top_k) -> dict:
    """Run every golden question through one search function.

    search_fn: question str -> list of chunk_ids (already cut to k).
    Returns overall + per-type hit@1 / hit@5 / MRR, plus a failures list.
    """
    overall = {"n": 0, "hit1": 0, "hitk": 0, "rr_sum": 0.0}
    by_type = {}                                # one counter dict per question type
    failures = []
    unanswerable_skipped = 0

    for item in golden:
        if not item.get("answerable", True):    # item 5's questions: count, don't grade
            unanswerable_skipped += 1
            continue

        ids = search_fn(item["question"])
        rank = first_relevant_rank(ids, chunks_by_id,
                                   item["expect_doc"], item["expect_phrase"])

        qtype = item.get("type", "untyped")
        if qtype not in by_type:
            by_type[qtype] = {"n": 0, "hit1": 0, "hitk": 0, "rr_sum": 0.0}

        for counters in (overall, by_type[qtype]):   # same update, both scopes
            counters["n"] += 1
            if rank == 1:
                counters["hit1"] += 1
            if rank is not None:                # rank <= k guaranteed: ids already cut
                counters["hitk"] += 1
                counters["rr_sum"] += 1.0 / rank

        if rank is None:
            failures.append({"question": item["question"], "type": qtype})

    result = {"k": k,
              "overall": _rates(overall),
              "by_type": {},
              "failures": failures,
              "unanswerable_skipped": unanswerable_skipped}
    for qtype, counters in by_type.items():
        result["by_type"][qtype] = _rates(counters)
    return result


def _rates(counters: dict) -> dict:
    """Turn raw counts into hit@1 / hit@k / MRR fractions."""
    n = counters["n"]
    if n == 0:                                  # empty type group: no division by zero
        return {"n": 0, "hit@1": 0.0, "hit@k": 0.0, "mrr": 0.0}
    return {"n": n,
            "hit@1": counters["hit1"] / n,
            "hit@k": counters["hitk"] / n,
            "mrr": counters["rr_sum"] / n}


GOLDEN_PATH = "eval/golden.json"


def main():
    # Job 4a — build everything ONCE, shared by all three modes
    print("loading + chunking ...")
    chunks = load_and_chunk(CONFIG.docs_dir)
    texts = []
    for c in chunks:                            # same heading+text recipe as the demos
        heading = c.metadata.get("heading", "")
        if heading:
            texts.append(f"{heading}\n{c.text}")
        else:
            texts.append(c.text)
    print("embedding (the ~30s step, happens once) ...")
    vec = VectorRetriever(chunks, embed_documents(texts))
    kw = KeywordRetriever(chunks)
    hybrid = HybridRetriever([vec, kw], chunks)
    chunks_by_id = {}
    for c in chunks:
        chunks_by_id[c.chunk_id] = c

    # Job 4b — three search functions: retrieve DEEP (25), cut to k AFTER
    k = CONFIG.top_k
    depth = CONFIG.retrieve_n

    def cut(pairs):                             # [(cid, score), ...] -> top-k ids only
        ids = []
        for cid, _score in pairs[:k]:
            ids.append(cid)
        return ids

    def vector_mode(question):
        return cut(vec.search(question, depth))

    def bm25_mode(question):
        return cut(kw.search(question, depth))

    def hybrid_mode(question):
        return cut(hybrid.search(question, depth))

    # Job 4c — run all modes over the golden set, print table + failures
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    modes = [("vector", vector_mode), ("bm25", bm25_mode), ("hybrid", hybrid_mode)]
    results = {}
    for name, fn in modes:
        results[name] = eval_mode(fn, chunks_by_id, golden, k)

    print(f"\n{'mode':<8} {'n':>3} {'hit@1':>7} {f'hit@{k}':>7} {'MRR':>7}")
    for name, _fn in modes:
        r = results[name]["overall"]
        print(f"{name:<8} {r['n']:>3} {r['hit@1']:>7.3f} {r['hit@k']:>7.3f} {r['mrr']:>7.3f}")

    print("\nper type:")
    for name, _fn in modes:
        for qtype, r in results[name]["by_type"].items():
            print(f"  {name:<8} {qtype:<12} n={r['n']:<3} "
                  f"hit@1={r['hit@1']:.3f}  hit@{k}={r['hit@k']:.3f}  mrr={r['mrr']:.3f}")

    print("\nfailures (question missed the top-k entirely):")
    any_failure = False
    for name, _fn in modes:
        for f_item in results[name]["failures"]:
            any_failure = True
            print(f"  {name:<8} [{f_item['type']}] {f_item['question']}")
    if not any_failure:
        print("  none 🎉")

    skipped = results["hybrid"]["unanswerable_skipped"]
    if skipped:
        print(f"\nunanswerable questions skipped from metrics: {skipped}")


if __name__ == "__main__":                  # Keep everything here 
    main()