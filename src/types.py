from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """A single piece of a document, ready to be embedded and searched.

    Attributes:
        text: The chunk's raw text content.
        source: Where it came from (e.g. the filename).
        chunk_id: Position of this chunk within its source, starting at 0.
        metadata: Optional extra info (page number, headings, etc.).
    """

    text: str
    source: str
    chunk_id: int
    metadata: dict[str, Any] = field(default_factory=dict)