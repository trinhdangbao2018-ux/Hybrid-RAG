def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of ~size characters."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    step = size - overlap  # advance less than a full window so chunks overlap
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i : i + size]
        if piece.strip():
            chunks.append(piece)
    return chunks