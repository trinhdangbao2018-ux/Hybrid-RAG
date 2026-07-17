import re

import ollama

from src.config import CONFIG
from src.types import Chunk

_PROMPT = """Answer the question using ONLY the numbered context below.
Cite the context you use with bracketed numbers like [1] or [2].
If the answer is not in the context, say "I don't know based on the provided
documents." — do not guess.

Context:
{context}

Question: {question}

Answer:"""

def build_prompt(question: str, chunks: list[Chunk]) ->str:    # ← chunks -> numbered context block -> full prompt

    blocks = []
    for i, chunk in enumerate(chunks, start = 1):     # go through all chunk
        heading = chunk.metadata.get("heading","")                    # .get("heading","") because metadata in chunk can be an empty dictionary
        blocks.append(f"[{i}] ({chunk.doc_id} — {heading})\n{chunk.text}")   # source tag + body, the output after this loop is: blocks =[card 1, card 2, ...]
    context = "\n\n".join(blocks)               # glue cards with a blank line; question has its own {question} slot
    return _PROMPT.format(context= context, question= question)   
    
def generate(question: str, chunks: list[Chunk],
             model: str = CONFIG.ollama_model) -> str:
    """Send the built prompt to the local Ollama model, return the answer text.

    Raises RuntimeError with a human message when the server/model is absent.
    """
    prompt = build_prompt(question, chunks)                              
    try:
            resp= ollama.chat (model=model,                        # Take the model from Config
            messages=[{"role": "user", "content": prompt}],       # Role of the person who write prompt
            options={"temperature": CONFIG.temperature},          # creativity knob lives in config, not hardcoded
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed — is Ollama running, and did you "
            f"`ollama pull {model}`? ({e})"
        ) from e                               
    return resp["message"]["content"]                               
    

def verify_citations(answer, n_chunks):  # ← answer text -> (valid, invalid) citation numbers
    """Return (valid, invalid) citation numbers found in the answer."""
    # n_chunks is the amout of chunk you feed the model
    found = re.findall(r"\[(\d+)\]", answer)      
    ints = []
    for s in found:                      
        ints.append(int(s))              
    cited = sorted(set(ints))            
    valid = []
    invalid = []
    for c in cited:
        if 1 <= c <= n_chunks:                          
            valid.append(c)
        else:
            invalid.append(c)
    return valid, invalid
