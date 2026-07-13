# Hybrid RAG — How This System Works

This is a small question-answering system that runs entirely on a local
machine. You give it a folder of documents and ask questions in plain
English. It finds the most relevant passages and has a language model
write an answer based on them.

The interesting part is the word *hybrid*: the system searches in two
different ways at the same time — by meaning (vector search) and by exact
words (BM25 keyword search) — and then merges the two result lists. Each
method catches things the other one misses.

## Loading Documents

Everything starts in the `documents/` folder. The loader walks through it
and picks up three kinds of files: Markdown (`.md`), plain text (`.txt`),
and PDF. Markdown and text files are read directly. PDFs go through the
**pypdf** library, which pulls the text out page by page. Anything else —
images, spreadsheets, whatever — is skipped with a warning.

Every document keeps its relative file path as its ID (the `doc_id`), so
later steps can always tell you *which file* an answer came from.

## Cutting Documents into Chunks

Search doesn't work well on whole documents — they're too long and mix
too many topics. So each document is cut into smaller pieces called
chunks.

The chunker is heading-aware. It splits the text at blank lines, and
remembers which markdown headings each piece sits under (for example
`Setup > Wi-Fi Configuration`). Pieces that belong to the same heading
are packed together until the chunk reaches **350 tokens**. Why 350? The
embedding model can only read 512 tokens at once, so 350 leaves a
comfortable safety margin. If one single paragraph is already too big, it
gets split at sentence boundaries instead.

Two small but important details: token counts come from the embedding
model's own tokenizer (not from counting words), and every chunk gets a
stable ID — the first 16 hex characters of the SHA-256 hash of its text.
Same text, same ID, every time.

## Turning Text into Vectors

For search-by-meaning, every chunk is converted into a list of numbers —
an embedding. The model doing this is **BAAI/bge-small-en-v1.5**, a small
English model that turns any text into a **384-dimensional** vector.
Texts with similar meaning end up with similar vectors, even when they
share no words.

One quirk of the BGE model family: questions need a special prefix —
"Represent this sentence for searching relevant passages:" — glued in
front before embedding. Documents don't get the prefix, only queries. All
vectors are also L2-normalized, which makes the next step much cheaper.
The model runs locally through the `sentence-transformers` library.

## Vector Search

When a question comes in, it gets embedded too, and the system looks for
the chunk vectors closest to the question vector. Because everything was
normalized, cosine similarity turns into a single matrix–vector
multiplication in NumPy — one line of code, no approximation, every chunk
checked.

Picking the best N results uses `argpartition`, a NumPy trick that finds
the top scores in linear time without sorting the entire array.

Vector search is the strong half for **paraphrase questions** — when you
ask "How do I get my money back?" and the document says "refund policy",
vector search still connects the two.

## Keyword Search with BM25

The other half is plain old exact-word matching, ranked by **BM25** (the
full name is Okapi BM25). This project uses the **bm25s** library,
version 0.3.9.

BM25 scores a chunk using three common-sense ideas. First, the more often
a query word appears in a chunk, the better — but with diminishing
returns, so ten repeats don't beat five by much (the `k1` parameter,
default 1.5, controls this saturation). Second, rare words count for more
than common ones — matching "argpartition" means a lot more than matching
"the". Third, long chunks get a small penalty so they can't win just by
containing more words (the `b` parameter, default 0.75, controls that).

BM25 is the strong half for **exact-match questions**: error codes, model
numbers, product names — the kind of tokens that embedding models tend to
blur together.

## Merging the Two Lists

Each searcher hands back its top **25 candidates**. Now the two ranked
lists have to become one, and the method used is **Reciprocal Rank Fusion
(RRF)**: a chunk at rank `r` in either list earns a score of
`1 / (60 + r)`, and the scores are added up. The constant 60 is the
standard choice from the original RRF paper.

Why RRF instead of just averaging the scores? Because BM25 scores and
cosine similarities live on completely different scales — comparing them
directly is meaningless. RRF sidesteps the problem entirely: it only
looks at ranks, never at raw scores, so nothing needs to be calibrated.

## Reranking

Fusion gives a decent shortlist, but there's one more quality pass: a
cross-encoder reranker, **cross-encoder/ms-marco-MiniLM-L6-v2**.

The difference from the embedding model is how it reads. The embedding
model encodes the question and each chunk *separately* and never sees
them together. The cross-encoder reads the question and a chunk *side by
side* in one pass and outputs a relevance score. That's much more
accurate but also much slower — which is exactly why it only runs on the
small fused shortlist instead of the whole corpus. The **top 5 chunks**
after reranking move on to the final step.

## Writing the Answer

The last step is generation. The question and the top 5 chunks are put
into a prompt, and **llama3.2**, running locally on **Ollama**, writes
the answer. The model is told to only use the provided context — if the
answer isn't in the chunks, it should say so rather than make things up.

Worth repeating: no cloud API is involved anywhere. Loading, embedding,
retrieval, reranking, and generation all happen on the local machine.

## Measuring Quality

How do you know any of this actually works? With a golden set — a
hand-written list of test questions in `eval/golden.json`. Each entry has
a question, the document that should answer it, and an exact phrase the
retrieved chunk must contain.

Questions come in two flavors, matching the two search methods:
`paraphrase` questions reword the document's content (vector search
should win those), and `keyword` questions use exact identifiers (BM25
should win those). The headline number is the hit rate: out of all test
questions, how many had their expected evidence show up in the top-k
retrieved chunks.
