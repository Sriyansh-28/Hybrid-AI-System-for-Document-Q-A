"""Document loading and smart chunking with overlap for the Hybrid AI System."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

from .config import ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk produced by the document processor."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    # Character-level start / end within the source document
    start_char: int = 0
    end_char: int = 0


class DocumentProcessor:
    """
    Loads documents from disk (TXT, PDF, DOCX) and splits them into
    overlapping chunks with configurable size/overlap.

    Chunking strategy
    -----------------
    1. Try to split on natural boundaries (double newline, single newline,
       sentence-ending punctuation) so that chunks are semantically coherent.
    2. Fall back to hard character-level splitting only when a segment
       exceeds `chunk_size`.
    3. Consecutive chunks share `chunk_overlap` characters to preserve
       cross-boundary context.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_directory(self, directory: str) -> List[Chunk]:
        """Load and chunk every supported document in *directory*."""
        chunks: List[Chunk] = []
        doc_paths = self._find_documents(directory)
        logger.info("Found %d document(s) in '%s'", len(doc_paths), directory)
        for path in doc_paths:
            try:
                chunks.extend(self.process_file(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping '%s': %s", path, exc)
        logger.info("Produced %d chunk(s) from %d document(s)", len(chunks), len(doc_paths))
        return chunks

    def process_file(self, file_path: str) -> List[Chunk]:
        """Load a single file and return its chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        text = self._load_text(path)
        text = self._clean_text(text)
        doc_id = path.stem
        return list(self._chunk_text(text, doc_id, metadata={"source": str(path), "filename": path.name}))

    def process_text(self, text: str, doc_id: str = "inline", metadata: Optional[Dict] = None) -> List[Chunk]:
        """Chunk an arbitrary text string directly."""
        cleaned = self._clean_text(text)
        return list(self._chunk_text(cleaned, doc_id, metadata=metadata or {}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_documents(self, directory: str) -> List[str]:
        root = Path(directory)
        paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            paths.extend(str(p) for p in root.rglob(f"*{ext}"))
        return sorted(paths)

    def _load_text(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if ext == ".pdf":
            return self._load_pdf(path)
        if ext == ".docx":
            return self._load_docx(path)
        raise ValueError(f"No loader for extension '{ext}'")

    @staticmethod
    def _load_pdf(path: Path) -> str:
        try:
            import pdfplumber  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install pdfplumber to load PDF files: pip install pdfplumber") from exc
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)

    @staticmethod
    def _load_docx(path: Path) -> str:
        try:
            import docx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install python-docx to load DOCX files: pip install python-docx") from exc
        doc = docx.Document(str(path))
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())

    @staticmethod
    def _clean_text(text: str) -> str:
        # Collapse runs of whitespace (but keep paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        # Normalise newlines
        text = re.sub(r"\r\n?", "\n", text)
        # Collapse more than two consecutive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
    ) -> Generator[Chunk, None, None]:
        """Split *text* into overlapping chunks and yield each one."""
        cfg = self.config
        segments = self._split_on_boundaries(text)

        buffer = ""
        buffer_start = 0  # char offset of buffer start within *text*
        global_offset = 0  # running char offset for each segment
        chunk_index = 0

        for segment in segments:
            seg_len = len(segment)

            # Would adding this segment push us over the limit?
            if buffer and len(buffer) + len(segment) > cfg.chunk_size:
                # Emit the current buffer as a chunk
                if len(buffer.strip()) >= cfg.min_chunk_size:
                    yield Chunk(
                        chunk_id=f"{doc_id}_{chunk_index}",
                        doc_id=doc_id,
                        text=buffer.strip(),
                        metadata=dict(metadata or {}),
                        start_char=buffer_start,
                        end_char=buffer_start + len(buffer),
                    )
                    chunk_index += 1

                # Start the next buffer with an overlapping tail of the previous one
                overlap_text = self._overlap_tail(buffer, cfg.chunk_overlap)
                overlap_start = buffer_start + len(buffer) - len(overlap_text)
                buffer = overlap_text + segment
                buffer_start = overlap_start
            else:
                if not buffer:
                    buffer_start = global_offset
                buffer += segment

            global_offset += seg_len

        # Emit any remaining text
        if buffer.strip() and len(buffer.strip()) >= cfg.min_chunk_size:
            yield Chunk(
                chunk_id=f"{doc_id}_{chunk_index}",
                doc_id=doc_id,
                text=buffer.strip(),
                metadata=dict(metadata or {}),
                start_char=buffer_start,
                end_char=buffer_start + len(buffer),
            )
        elif buffer.strip() and chunk_index > 0:
            # Append tiny tail to the last chunk rather than discarding it
            pass  # already yielded above; tail below min_chunk_size is dropped

    def _split_on_boundaries(self, text: str) -> List[str]:
        """
        Split text on natural boundaries while preserving the separators
        (they are kept attached to the preceding segment).
        Returns a list of non-empty string segments.
        """
        separators = self.config.sentence_separators
        # Try separators in order of preference
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                # Re-attach separator to end of each part (except last)
                segments = [p + sep for p in parts[:-1]] + [parts[-1]]
                # Now hard-split any segment still above chunk_size
                result = []
                for seg in segments:
                    if len(seg) > self.config.chunk_size:
                        result.extend(self._hard_split(seg))
                    elif seg:
                        result.append(seg)
                return result
        # No boundary found at all → hard split
        return self._hard_split(text)

    def _hard_split(self, text: str) -> List[str]:
        """Split text into fixed-size pieces as a last resort."""
        size = self.config.chunk_size
        return [text[i : i + size] for i in range(0, len(text), size) if text[i : i + size]]

    @staticmethod
    def _overlap_tail(text: str, overlap: int) -> str:
        """Return the last *overlap* characters of *text*, preferring a word boundary."""
        if overlap <= 0 or len(text) <= overlap:
            return text
        tail = text[-overlap:]
        # Try to start at a word boundary
        space_idx = tail.find(" ")
        if space_idx != -1:
            return tail[space_idx + 1 :]
        return tail
