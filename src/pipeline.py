import hashlib                                    
import json                                       
from dataclasses import asdict                    
from pathlib import Path                          

import numpy as np                                

from src.config import CONFIG                     
from src.embedding import embed_documents          
from src.generator import generate, verify_citations  
from src.loader import load_and_chunk             
from src.retriever import HybridRetriever         
from src.types import Chunk                       
from src.vector_search import VectorRetriever     
from src.keyword_search import KeywordRetriever
from src.rerank import Reranker

def _doc_hashes(docs_dir: str) -> dict[str, str]:
    # INPUT:  docs_dir str — from CONFIG.docs_dir ("documents")
    # Process: doc_dir -> byte -> hash
    # OUTPUT: {"rag-tech-stack.md": "a3f9c2e871b04d5e"} — 16-char sha256 per file
    #         -> stamped into manifest.json by build_index
    #         -> recomputed & compared against that stamp by load_index
    hash_docs = {}
    for path in Path(docs_dir).rglob("*"): 
        if path.is_file():
            byte_doc = path.read_bytes()    
            hash_doc = hashlib.sha256(byte_doc).hexdigest()[:16]    
            hash_docs[str(path.relative_to(docs_dir))] = hash_doc   
        else:
            continue
    return hash_docs

def build_index(cfg=CONFIG) -> None:
    # INPUT:  documents/ on disk (via load_and_chunk -> list[Chunk],
    #         via embed_documents -> matrix (n_chunks, 384) float32)
    #
    # OUTPUT: nothing returned — writes 3 files into index/:
    #         embeddings.npy (the matrix), chunks.json (asdict per Chunk),
    #         manifest.json (embed_model + chunk params + _doc_hashes)
    #         -> all three are read back ONLY by load_index
    chunks = load_and_chunk(cfg.docs_dir)       
    texts = []
    for c in chunks:                            
        heading = c.metadata.get("heading", "") 
        texts.append(f"{heading}\n{c.text}")    
    matrix = embed_documents(texts)             

    out = Path(cfg.index_dir)                  
    out.mkdir(exist_ok=True)                    
    np.save(out / "embeddings.npy", matrix)     
    
    dicts = []                                   
    for c in chunks:
        dicts.append(asdict(c))                 
    (out / "chunks.json").write_text(
        json.dumps(dicts, indent=1), encoding="utf-8")      

    manifest = {                                                    
        "embed_model": cfg.embed_model,
        "chunk_max_tokens": cfg.chunk_max_tokens,
        "doc_hashes": _doc_hashes(cfg.docs_dir),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")

    n_docs = len(set(c.doc_id for c in chunks))             
    print(f"indexed {len(chunks)} chunks from {n_docs} documents")



def load_index(cfg=CONFIG) -> tuple[list[Chunk], np.ndarray]:
    # INPUT:  the 3 files build_index wrote in index/
    # OUTPUT: (list[Chunk], matrix (n_chunks, 384)) — same pair build_index
    #         had in memory, resurrected in ~1s instead of 30s
    #         -> consumed by make_retriever to assemble the searchers
    # Side effects: RuntimeError on embed_model mismatch; WARNING on changed docs

    out = Path(cfg.index_dir)                       
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))     
    if manifest["embed_model"] != cfg.embed_model: 
        raise RuntimeError(f"index built with {manifest['embed_model']} but "f"config wants {cfg.embed_model}")
    if manifest["doc_hashes"] != _doc_hashes(cfg.docs_dir):      
        print("WARNING: documents/ changed since last index — answers may be stale.")

    dicts = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
    chunks = []
    for d in dicts:
        chunks.append(Chunk(**d))

    matrix = np.load(out / "embeddings.npy")
    return chunks, matrix

def make_retriever(cfg=CONFIG):
    # INPUT:  (chunks, matrix) from load_index
    # OUTPUT: (HybridRetriever, list[Chunk]) — the ready-to-search engine
    #         -> consumed by ask(); also handy for evaluate.py later (skips embed)
    chunks, matrix = load_index(cfg)
    vector_retriever = VectorRetriever(chunks, matrix)  
    keyword_retriever = KeywordRetriever(chunks)
    hybrid = HybridRetriever([vector_retriever, keyword_retriever], chunks)
    return hybrid, chunks

def ask(question: str, cfg=CONFIG, show: bool = False,
        no_llm: bool = False, model: str | None = None) -> str:
    # INPUT:  question str — from the human (CLI later, M8 §2)
    # OUTPUT: answer str with [n] citations — from generate, checked by
    #         verify_citations -> printed to the human; the END of the relay
    # Flow inside: make_retriever -> hybrid.search(retrieve_n) -> [:top_k]
    #              -> generate(model or cfg.ollama_model) -> verify -> return
    
    hybrid, chunks = make_retriever(cfg)                                  

    fused = hybrid.search(question,cfg.retrieve_n)       
    top = []
    for cid, score in fused:
        top.append(hybrid.chunk_by_id[cid])
    top = Reranker(cfg.rerank_model).rerank(question, top, cfg.top_k)

    if show or no_llm:
        for i, c in enumerate(top, 1):
            print(f"[{i}] {c.doc_id} — {c.metadata.get('heading', '')}")
    if no_llm:
        return ""

    answer = generate(question, top, model or cfg.ollama_model)
    valid, invalid = verify_citations(answer, len(top))
    if invalid:
        print(f"WARNING: hallucinated citation(s): {invalid}")
    return answer