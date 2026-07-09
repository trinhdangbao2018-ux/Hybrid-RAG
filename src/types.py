import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str                # sha256 of the text — stable content ID
    doc_id: str                  # relative path of the source file
    text: str
    position: int                # 0-based order within its document
    metadata: dict[str, Any] = field(default_factory=dict)


def make_chunk(text: str, doc_id: str, position: int,
               metadata: dict[str, Any] | None = None) -> Chunk:
    chunk_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return Chunk(chunk_id, doc_id, text, position, metadata or {})