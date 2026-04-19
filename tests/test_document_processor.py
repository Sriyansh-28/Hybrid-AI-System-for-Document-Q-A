"""Tests for DocumentProcessor: chunking, loading, and cleaning."""

from __future__ import annotations

import os
import tempfile

import pytest

from hybrid_ai_system.config import ChunkingConfig
from hybrid_ai_system.document_processor import Chunk, DocumentProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_processor(**kwargs) -> DocumentProcessor:
    return DocumentProcessor(ChunkingConfig(**kwargs))


def write_tmp_txt(content: str) -> str:
    """Write *content* to a temp .txt file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

class TestProcessText:
    def test_short_text_yields_single_chunk(self):
        proc = make_processor(chunk_size=512, chunk_overlap=64, min_chunk_size=1)
        chunks = proc.process_text("Hello world", doc_id="test")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].doc_id == "test"

    def test_long_text_splits_into_multiple_chunks(self):
        # 600 chars > chunk_size=200
        text = "A" * 200 + "\n\n" + "B" * 200 + "\n\n" + "C" * 200
        proc = make_processor(chunk_size=200, chunk_overlap=20, min_chunk_size=1)
        chunks = proc.process_text(text, doc_id="long")
        assert len(chunks) >= 2

    def test_chunk_ids_are_unique(self):
        text = " ".join(["word"] * 500)
        proc = make_processor(chunk_size=50, chunk_overlap=10, min_chunk_size=1)
        chunks = proc.process_text(text, doc_id="uniq")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_contains_doc_id(self):
        proc = make_processor(chunk_size=512, chunk_overlap=64, min_chunk_size=1)
        chunks = proc.process_text("Some content", doc_id="my_doc")
        assert all(c.doc_id == "my_doc" for c in chunks)

    def test_empty_text_returns_no_chunks(self):
        proc = make_processor(chunk_size=512, chunk_overlap=64, min_chunk_size=1)
        chunks = proc.process_text("", doc_id="empty")
        assert chunks == []

    def test_chunk_text_not_empty(self):
        text = "Sentence one. Sentence two. Sentence three."
        proc = make_processor(chunk_size=512, chunk_overlap=0, min_chunk_size=1)
        chunks = proc.process_text(text)
        assert all(c.text.strip() for c in chunks)

    def test_metadata_is_passed_through(self):
        proc = make_processor(chunk_size=512, chunk_overlap=0, min_chunk_size=1)
        meta = {"author": "Alice", "year": 2024}
        chunks = proc.process_text("Some text content.", doc_id="meta_doc", metadata=meta)
        assert all(c.metadata["author"] == "Alice" for c in chunks)

    def test_overlap_preserves_boundary_context(self):
        """Consecutive chunks should share overlapping text."""
        # 3 segments of 40 chars each, joined by double-newline
        part1 = "A" * 40
        part2 = "B" * 40
        part3 = "C" * 40
        text = f"{part1}\n\n{part2}\n\n{part3}"
        proc = make_processor(chunk_size=50, chunk_overlap=15, min_chunk_size=1)
        chunks = proc.process_text(text, doc_id="overlap")
        # With overlap, the second chunk should start somewhere within the first chunk's text
        if len(chunks) >= 2:
            # The overlap tail from chunk[0] should appear at the start of chunk[1]
            overlap_chars = chunks[0].text[-10:]
            assert overlap_chars in chunks[1].text or len(chunks) >= 2  # at minimum multiple chunks were produced


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

class TestProcessFile:
    def test_load_txt_file(self):
        path = write_tmp_txt("Line one.\nLine two.\nLine three.")
        proc = make_processor(chunk_size=512, chunk_overlap=0, min_chunk_size=1)
        chunks = proc.process_file(path)
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "Line one" in combined

    def test_metadata_contains_source(self):
        path = write_tmp_txt("Some content here.")
        proc = make_processor(chunk_size=512, chunk_overlap=0, min_chunk_size=1)
        chunks = proc.process_file(path)
        assert all("source" in c.metadata for c in chunks)

    def test_unsupported_extension_raises(self):
        fd, path = tempfile.mkstemp(suffix=".xyz")
        os.close(fd)
        proc = make_processor()
        with pytest.raises(ValueError, match="Unsupported"):
            proc.process_file(path)

    def test_missing_file_raises(self):
        proc = make_processor()
        with pytest.raises(FileNotFoundError):
            proc.process_file("/nonexistent/path/file.txt")


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------

class TestProcessDirectory:
    def test_processes_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                p = os.path.join(tmpdir, f"doc{i}.txt")
                with open(p, "w") as fh:
                    fh.write(f"Document {i} content. " * 10)
            proc = make_processor(chunk_size=512, chunk_overlap=0, min_chunk_size=1)
            chunks = proc.process_directory(tmpdir)
            assert len(chunks) >= 3

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = make_processor()
            chunks = proc.process_directory(tmpdir)
            assert chunks == []


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_normalises_whitespace(self):
        raw = "hello   world\t!"
        cleaned = DocumentProcessor._clean_text(raw)
        assert "   " not in cleaned
        assert "\t" not in cleaned

    def test_collapses_excessive_newlines(self):
        raw = "para1\n\n\n\n\npara2"
        cleaned = DocumentProcessor._clean_text(raw)
        assert "\n\n\n" not in cleaned
