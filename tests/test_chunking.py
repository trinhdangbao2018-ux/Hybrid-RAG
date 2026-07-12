from src.chunking import pack_blocks, split_blocks


def test_split_block(): # Using split_block function to find the headline
    text = "# Guide\n\nIntro para.\n\n## Refunds\n\nRefund para."
    blocks = split_blocks(text)
    assert blocks == [("Guide", "Intro para."),
                      ("Guide > Refunds", "Refund para.")]


def test_pack_block(): # Using pack block function to combine or divide block to match the token limit
    blocks = [("A", "one two three")] * 3 + [("B", "four five")]
    chunks = pack_blocks(blocks, max_tokens=10)
    # exact packing may vary; the invariants may not:
    assert all(h in ("A", "B") for h, _ in chunks)      # no cross-heading mix
    assert len(chunks) >= 2                              # budget forced a split


def test_no_duplicate_chunks():
    text = "para one.\n\npara two.\n\npara three."
    chunks = pack_blocks(split_blocks(text), max_tokens=5)
    texts = [c for _, c in chunks]
    assert len(texts) == len(set(texts))                 # every chunk unique
