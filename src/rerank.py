from src.types import Chunk


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name    # chưa dùng tới, M9 sẽ nạp cross-encoder thật ở đây

    def rerank(self, question: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        return chunks[:k]    # stub: giữ nguyên thứ tự fusion, chỉ cắt bớt — M9 sẽ thay bằng điểm rerank thật
