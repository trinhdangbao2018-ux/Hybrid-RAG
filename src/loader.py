from pathlib import Path

from src.chunking import chunk_text
from src.types import Chunk

TEXT_EXTENSIONS = {".md", ".txt"}


def load_and_chunk(directory: str) -> list[Chunk]:
    """Load every supported document in a directory and split it into chunks.

    Reads .md/.txt directly and .pdf via pypdf. Each chunk is tagged with the
    filename it came from. Files with unsupported extensions are skipped.
    """
    all_chunks: list[Chunk] = []

    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():          # skip subdirectories
            continue
        suffix = path.suffix.lower()

        if suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue

        for i, piece in enumerate(chunk_text(text)):
            all_chunks.append(Chunk(piece, path.name, i))

    return all_chunks


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF, one page at a time."""
    from pypdf import PdfReader  # lazy import: only needed when a PDF appears

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)