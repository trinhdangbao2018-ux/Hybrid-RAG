"""Benchmark the four retrieval configurations on documents/ with eval/golden.json.

Adds what `cli.py eval` does not measure: the cross-encoder reranker and the time
per query. Run from the repository root:  python -m eval.benchmark
Prints a table and writes eval/benchmark_results.json.
"""
import json
import time

from src.config import CONFIG
from src.embedding import embed_documents
from src.evaluate import GOLDEN_PATH, eval_mode
from src.keyword_search import KeywordRetriever
from src.loader import load_and_chunk
from src.rerank import Reranker
from src.retriever import HybridRetriever
from src.vector_search import VectorRetriever

RESULTS_PATH = "eval/benchmark_results.json"


def main() -> None:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    chunks = load_and_chunk(CONFIG.docs_dir)
    texts = [f"{c.metadata['heading']}\n{c.text}" if c.metadata.get("heading") else c.text
             for c in chunks]
    vec = VectorRetriever(chunks, embed_documents(texts))
    kw = KeywordRetriever(chunks)
    hybrid = HybridRetriever([vec, kw], chunks)
    by_id = {c.chunk_id: c for c in chunks}
    reranker = Reranker(CONFIG.rerank_model)
    k, depth = CONFIG.top_k, CONFIG.retrieve_n

    def fused(q):
        return [cid for cid, _score in hybrid.search(q, depth)]

    modes = {
        "vector": lambda q: [cid for cid, _ in vec.search(q, depth)][:k],
        "bm25": lambda q: [cid for cid, _ in kw.search(q, depth)][:k],
        "hybrid": lambda q: fused(q)[:k],
        "hybrid+rerank": lambda q: [c.chunk_id for c in
                                    reranker.rerank(q, [by_id[i] for i in fused(q)], k)],
    }
    for fn in modes.values():                    # warm up: load models before timing
        fn(golden[0]["question"])

    results = {"documents": len({c.doc_id for c in chunks}), "chunks": len(chunks),
               "questions": len(golden), "top_k": k, "modes": {}}
    print(f"{results['documents']} documents, {results['chunks']} chunks, "
          f"{results['questions']} questions\n")
    print(f"{'mode':<14} {'hit@1':>7} {f'hit@{k}':>7} {'MRR':>7} {'ms/query':>10}")
    for name, fn in modes.items():
        start = time.perf_counter()
        r = eval_mode(fn, by_id, golden, k)
        ms = (time.perf_counter() - start) / max(r["overall"]["n"], 1) * 1000
        o = r["overall"]
        print(f"{name:<14} {o['hit@1']:>7.3f} {o['hit@k']:>7.3f} {o['mrr']:>7.3f} {ms:>10.1f}")
        results["modes"][name] = {"overall": o, "by_type": r["by_type"],
                                  "ms_per_query": round(ms, 1),
                                  "failures": r["failures"]}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nwritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
