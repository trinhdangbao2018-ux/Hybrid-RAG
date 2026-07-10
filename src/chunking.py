from collections.abc import Callable
from functools import lru_cache
import re #regular expresion

_HEADING = re.compile(r"^(#{1,6})\s+(.*)")
# re.compile create a finding template
# r = raw text, ^ poing to the begining of the prompt, #{1,6}, match index =0 to 5 that is # (heading have 6 level from # to ######)
# s+ eat up all space between # part and heaing part
# .* match through every thing left

@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    # Lazy import: importing this file should not load transformers.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens with the real tokenizer used by the embedding model."""
    if model_name is None:
        from src.config import CONFIG

        model_name = CONFIG.embed_model

    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=True))

def split_blocks(text: str) -> list[tuple[str, str]]:
      headings: list[str] = []  # remember the heading
      blocks: list[tuple[str, str]] = []    
      """Using stack to remember every path"""

      for raw in re.split(r"\n\s*\n", text): # re.split(r"\n\s*\n", text) delete space at the start of the paragraph
          block = raw.strip()
          if not block:
              continue

          m = _HEADING.match(block) # check if heading match block
          if m:
              level = len(m.group(1))
              headings = headings[: level - 1] # Take the Heading level before this heading
              headings.append(m.group(2).strip()) # Add this Heading to the block, heading exp: [big topic 1, medium topic 1]

              rest = block[m.end():].strip()
              if rest:
                  blocks.append((" > ".join(headings), rest))  # Join the heading Example [big topic 1, medium topic 1] -> [big topic 1 -> medium topic 1], rest
          else:
              blocks.append((" > ".join(headings), block))

      return blocks

def split_sentences(text: str) -> list[str]:        # Split paragraph into token
    """Naive sentence split — good enough for oversized blocks."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def pack_blocks(
    blocks: list[tuple[str, str]],
    max_tokens: int = 350,
    token_counter: Callable[[str], int] | None = None,
) -> list[tuple[str, str]]: # This is a buffer function
    """Pack consecutive blocks under the same heading into max_tokens chunks.

    Returns (heading_path, chunk_text) pairs. Oversized single blocks are
    split by sentence.
    """
    if token_counter is None:
        token_counter = count_tokens

    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_heading = ""
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if buf:
            chunks.append((buf_heading, "\n\n".join(buf)))
            buf, buf_tokens = [], 0

    for heading, block in blocks:
        text_for_count = f"{heading}\n{block}" if heading else block
        btok = token_counter(text_for_count)
        if btok > max_tokens:                      # single huge block
            flush()
            piece, ptok = [], 0
            for sent in split_sentences(block):
                sent_for_count = f"{heading}\n{sent}" if heading else sent
                stok = token_counter(sent_for_count)
                if piece and ptok + stok > max_tokens:
                    chunks.append((heading, " ".join(piece)))
                    piece, ptok = [], 0
                piece.append(sent)
                ptok += stok
            if piece:
                chunks.append((heading, " ".join(piece)))
            continue
        if buf and (heading != buf_heading or buf_tokens + btok > max_tokens):
            flush()
        if not buf:
            buf_heading = heading
        buf.append(block)
        buf_tokens += btok
    flush()
    return chunks
