from dataclasses import dataclass

#frozen = True turn every vale to a constant. The value will be the same even if you change the value in the latter part
@dataclass(frozen=True)
class Config:
    docs_dir: str = "documents"
    index_dir: str = "index"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_max_tokens: int = 350          # inside bge-small's 512-token limit
    retrieve_n: int = 25                 # per retriever, before fusion
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_k: int = 5                       # chunks that reach the LLM
    ollama_model: str = "llama3.2"

#After this step the only thing that needed is to import CONFIG from here
CONFIG = Config()