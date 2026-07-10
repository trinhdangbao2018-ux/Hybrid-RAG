from pathlib import Path

from src.chunking import pack_blocks, split_blocks
from src.config import CONFIG
from src.types import Chunk, make_chunk

TEXT_EXTENSIONS = {".md", ".txt"}


def load_documents(directory: str) -> list[tuple[str, str]]:
    """Return (doc_id, text) for every supported file, recursively."""
    root = Path(directory)
    docs: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            print(f"skipped (unsupported): {path.name}")
            continue
        docs.append((str(path.relative_to(root)), text))
    return docs


def load_and_chunk(directory: str, max_tokens: int = CONFIG.chunk_max_tokens) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in load_documents(directory):
        for pos, (heading, body) in enumerate(pack_blocks(split_blocks(text), max_tokens)):
            chunks.append(make_chunk(body, doc_id, pos, {"heading": heading}))
    return chunks


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # lazy: only imported when a PDF appears
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)