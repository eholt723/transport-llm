import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub heavy deps not needed by the functions under test so pytest can collect
# this file without requiring a full sentence-transformers install.
sys.modules.setdefault("sentence_transformers", MagicMock())

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rag_prep import clean_text, chunk_text, infer_domain


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("  hello  ") == "hello"


def test_clean_text_collapses_multiple_newlines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_clean_text_collapses_inline_spaces():
    assert clean_text("a  b   c") == "a b c"


def test_clean_text_normalizes_crlf():
    assert "\r" not in clean_text("line1\r\nline2")


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_chunk_text_all_chunks_meet_min_length():
    text = "First paragraph.\n\n" + "x" * 200
    chunks = chunk_text(text, chunk_size=600, chunk_overlap=0)
    assert all(len(c) >= 80 for c in chunks)


def test_chunk_text_drops_paragraphs_under_min_length():
    text = "Short.\n\n" + "y" * 200
    chunks = chunk_text(text, chunk_size=600, chunk_overlap=0)
    assert not any(c.strip() == "Short." for c in chunks)


def test_chunk_text_empty_input_returns_empty():
    assert chunk_text("", 600, 0) == []


def test_chunk_text_overlap_prepends_tail_of_previous():
    para_a = "a" * 300
    para_b = "b" * 300
    chunks = chunk_text(para_a + "\n\n" + para_b, chunk_size=600, chunk_overlap=60)
    if len(chunks) >= 2:
        tail = chunks[0][-60:].strip()
        assert chunks[1].startswith(tail[:20])


def test_chunk_text_long_paragraph_is_split():
    long_para = "x" * 1200
    chunks = chunk_text(long_para, chunk_size=400, chunk_overlap=0)
    assert len(chunks) >= 3


# ── infer_domain ──────────────────────────────────────────────────────────────

def test_infer_domain_from_parent_directory(tmp_path):
    known = {"rail", "auto"}
    p = tmp_path / "rail" / "signaling.md"
    p.parent.mkdir()
    assert infer_domain(p, known, "other") == "rail"


def test_infer_domain_from_filename_prefix(tmp_path):
    known = {"rail", "auto"}
    p = tmp_path / "docs" / "auto_powertrains.md"
    p.parent.mkdir()
    assert infer_domain(p, known, "other") == "auto"


def test_infer_domain_falls_back_to_default(tmp_path):
    known = {"rail", "auto"}
    p = tmp_path / "misc" / "document.md"
    p.parent.mkdir()
    assert infer_domain(p, known, "other") == "other"


def test_infer_domain_parent_takes_priority_over_prefix(tmp_path):
    known = {"rail", "auto"}
    p = tmp_path / "rail" / "auto_document.md"
    p.parent.mkdir()
    assert infer_domain(p, known, "other") == "rail"
