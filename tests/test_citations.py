from src.generator import build_prompt, verify_citations

from src.types import make_chunk


# ---------- verify_citations ----------

def test_valid_and_invalid_split():
    assert verify_citations("uses [1] and [3]", 5) == ([1, 3], [])
    assert verify_citations("[9]", 5) == ([], [9])
    assert verify_citations("real [2] fake [7]", 5) == ([2], [7])


def test_boundaries_are_valid():
    # the <= bug: card [1] (first) and card [n] (last) EXIST — both must pass
    assert verify_citations("[1] and [5]", 5) == ([1, 5], [])
    # just outside both ends must fail
    assert verify_citations("[0] and [6]", 5) == ([], [0, 6])


def test_duplicates_counted_once():
    assert verify_citations("[2] and [2] again", 5) == ([2], [])


def test_no_citations_is_safe():
    assert verify_citations("no citations here", 5) == ([], [])


def test_plain_numbers_ignored():
    # digits without brackets (versions, formulas) must not be citations
    assert verify_citations("version 0.3.9 has 384 dimensions", 5) == ([], [])


# ---------- build_prompt ----------

def _two_chunks():
    a = make_chunk("bm25s version 0.3.9", "doc.md", 0, {"heading": "BM25"})
    b = make_chunk("384-dimensional vectors", "doc.md", 1)  # no heading
    return [a, b]


def test_cards_numbered_from_one():
    p = build_prompt("Which version?", _two_chunks())
    assert "[1] (doc.md — BM25)" in p
    assert "[2] (doc.md — )" in p          # heading-less chunk still renders


def test_question_appears_exactly_once():
    p = build_prompt("Which version?", _two_chunks())
    assert p.count("Which version?") == 1


def test_refusal_instruction_present():
    # the anti-hallucination line must survive any template edit
    p = build_prompt("Which version?", _two_chunks())
    assert "I don't know" in p


def test_cards_separated_by_blank_line():
    p = build_prompt("q", _two_chunks())
    assert "bm25s version 0.3.9\n\n[2]" in p
