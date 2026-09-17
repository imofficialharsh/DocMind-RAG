"""Structure and header-aware recursive chunker preserving page numbers and table integrity."""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.config import get_settings
from src.parser import PageData, ParsedDocument


@dataclass
class DocumentChunk:
    """A granular chunk of document text enriched with provenance metadata."""

    chunk_id: str
    text: str
    page_num: int
    file_hash: str
    source_filename: str
    chunk_index: int
    is_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return asdict(self)


class RecursiveDocumentChunker:
    """
    Recursively splits page content on semantic separators while keeping tables intact
    and attaching precise page provenance metadata.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of separators until pieces are <= chunk_size."""
        final_chunks: List[str] = []
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Determine which separator to use
        separator = separators[-1]
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        # Merge splits with overlap
        current_doc: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split) + (len(separator) if current_doc else 0)

            if current_len + split_len > self.chunk_size:
                if current_doc:
                    merged = separator.join(current_doc).strip()
                    if merged:
                        if len(merged) > self.chunk_size and new_separators:
                            # Recurse further down with remaining separators
                            final_chunks.extend(self._split_text_recursively(merged, new_separators))
                        else:
                            final_chunks.append(merged)

                    # Compute overlap from previous doc
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for piece in reversed(current_doc):
                        if overlap_len + len(piece) <= self.chunk_overlap:
                            overlap_doc.insert(0, piece)
                            overlap_len += len(piece) + len(separator)
                        else:
                            break
                    current_doc = overlap_doc
                    current_len = sum(len(p) for p in current_doc) + (
                        len(separator) * (len(current_doc) - 1) if current_doc else 0
                    )

            current_doc.append(split)
            current_len += len(split) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            merged = separator.join(current_doc).strip()
            if merged:
                if len(merged) > self.chunk_size and new_separators:
                    final_chunks.extend(self._split_text_recursively(merged, new_separators))
                else:
                    final_chunks.append(merged)

        return final_chunks

    def _chunk_large_table(self, table_md: str, max_table_size: int = 3500) -> List[str]:
        """Split a large Markdown table row-by-row while preserving header and keeping multi-row sections cohesive."""
        lines = [line.strip() for line in table_md.split("\n") if line.strip()]
        if len(lines) <= 2:
            return [table_md]

        # Extract header and delimiter row
        header = lines[0]
        delimiter = lines[1] if len(lines) > 1 and "---" in lines[1] else "|---|---|"
        body_rows = lines[2:] if "---" in delimiter else lines[1:]

        chunks: List[str] = []
        current_rows: List[str] = []
        base_len = len(header) + len(delimiter) + 2
        target_size = max(self.chunk_size, max_table_size)

        for row in body_rows:
            current_len = base_len + sum(len(r) + 1 for r in current_rows) + len(row) + 1
            if current_len > target_size and current_rows:
                chunk_content = f"{header}\n{delimiter}\n" + "\n".join(current_rows)
                chunks.append(chunk_content)
                current_rows = []

            current_rows.append(row)

        if current_rows:
            chunk_content = f"{header}\n{delimiter}\n" + "\n".join(current_rows)
            chunks.append(chunk_content)

        return chunks if chunks else [table_md]

    def chunk_page(
        self,
        page: PageData,
        file_hash: str,
        source_filename: str,
        start_index: int = 0,
    ) -> List[DocumentChunk]:
        """Chunk a single page and produce DocumentChunk objects with page metadata."""
        chunks: List[DocumentChunk] = []
        idx = start_index

        # Extract contextual lead-in summary from page text to give tables semantic context
        page_summary = ""
        if page.text:
            cleaned_page_text = " ".join(page.text.split())
            if cleaned_page_text:
                page_summary = cleaned_page_text[:280] + ("..." if len(cleaned_page_text) > 280 else "")

        # 1. Chunk markdown tables preserving section cohesion (up to 3500 chars per table piece)
        for table_idx, table_md in enumerate(page.tables, start=1):
            if len(table_md) <= 3500:
                table_pieces = [table_md]
            else:
                table_pieces = self._chunk_large_table(table_md, max_table_size=3500)

            for part_idx, piece in enumerate(table_pieces, start=1):
                chunk_id = f"{file_hash[:8]}_p{page.page_num}_c{idx}"
                header_line = f"[Page {page.page_num} Table {table_idx} (Part {part_idx})]"
                if page_summary:
                    chunk_text = f"{header_line} | Context: {page_summary}\n{piece}"
                else:
                    chunk_text = f"{header_line}\n{piece}"

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        page_num=page.page_num,
                        file_hash=file_hash,
                        source_filename=source_filename,
                        chunk_index=idx,
                        is_table=True,
                        metadata={
                            "page_num": page.page_num,
                            "table_index": table_idx,
                            "is_table": True,
                        },
                    )
                )
                idx += 1

        # 2. Chunk plain text on the page
        if page.text:
            text_splits = self._split_text_recursively(page.text, self.separators)
            for split in text_splits:
                if not split.strip():
                    continue
                chunk_id = f"{file_hash[:8]}_p{page.page_num}_c{idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=split.strip(),
                        page_num=page.page_num,
                        file_hash=file_hash,
                        source_filename=source_filename,
                        chunk_index=idx,
                        is_table=False,
                        metadata={
                            "page_num": page.page_num,
                            "is_table": False,
                        },
                    )
                )
                idx += 1

        return chunks

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunk all pages in a ParsedDocument preserving document-wide ordering."""
        all_chunks: List[DocumentChunk] = []
        current_idx = 0

        for page in doc.pages:
            page_chunks = self.chunk_page(
                page=page,
                file_hash=doc.file_hash,
                source_filename=doc.filename,
                start_index=current_idx,
            )
            all_chunks.extend(page_chunks)
            current_idx += len(page_chunks)

        return all_chunks

