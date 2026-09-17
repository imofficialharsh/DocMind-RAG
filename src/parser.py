"""PDF Parser utilizing pdfplumber with explicit Markdown table extraction and disk caching."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pdfplumber
from tabulate import tabulate

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Structured representation of extracted page content."""

    page_num: int
    text: str
    tables: List[str] = field(default_factory=list)
    combined_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete representation of a parsed document."""

    file_hash: str
    filename: str
    total_pages: int
    pages: List[PageData]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to JSON-serializable dictionary."""
        return {
            "file_hash": self.file_hash,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "pages": [asdict(p) for p in self.pages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedDocument":
        """Reconstruct ParsedDocument from dictionary."""
        pages = [
            PageData(
                page_num=p["page_num"],
                text=p["text"],
                tables=p.get("tables", []),
                combined_content=p.get("combined_content", ""),
                metadata=p.get("metadata", {}),
            )
            for p in data.get("pages", [])
        ]
        return cls(
            file_hash=data["file_hash"],
            filename=data["filename"],
            total_pages=data["total_pages"],
            pages=pages,
            metadata=data.get("metadata", {}),
        )


def compute_file_hash(file_source: Union[str, Path, bytes]) -> str:
    """Compute SHA-256 hash of a file path or in-memory byte buffer."""
    sha256 = hashlib.sha256()
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
    elif isinstance(file_source, bytes):
        sha256.update(file_source)
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")
    return sha256.hexdigest()


def format_table_as_markdown(raw_table: List[List[Optional[str]]]) -> Optional[str]:
    """Convert a raw pdfplumber table into a clean GitHub-Flavored Markdown table."""
    if not raw_table or not any(raw_table):
        return None

    # Clean rows: strip whitespaces and convert None to empty string
    cleaned_rows: List[List[str]] = []
    for row in raw_table:
        cleaned_row = [
            "" if cell is None else str(cell).replace("\n", " ").strip()
            for cell in row
        ]
        # Ignore rows that are completely empty
        if any(cell != "" for cell in cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return None

    # Handle single-row table or multiple rows
    if len(cleaned_rows) == 1:
        headers = [f"Col {i+1}" for i in range(len(cleaned_rows[0]))]
        data = cleaned_rows
        return tabulate(data, headers=headers, tablefmt="github")

    # If first row has content, treat it as headers
    headers = cleaned_rows[0]
    data = cleaned_rows[1:]
    return tabulate(data, headers=headers, tablefmt="github")


def _is_inside_any_bbox(
    obj: Dict[str, Any], bboxes: List[tuple[float, float, float, float]]
) -> bool:
    """Check if a layout object falls inside any bounding box (x0, top, x1, bottom)."""
    obj_x0 = obj.get("x0", 0)
    obj_top = obj.get("top", 0)
    obj_x1 = obj.get("x1", 0)
    obj_bottom = obj.get("bottom", 0)

    # Center point check
    mid_x = (obj_x0 + obj_x1) / 2
    mid_y = (obj_top + obj_bottom) / 2

    for x0, top, x1, bottom in bboxes:
        if x0 <= mid_x <= x1 and top <= mid_y <= bottom:
            return True
    return False


def extract_page_content(page: pdfplumber.page.Page, page_num: int) -> PageData:
    """Extract non-table text and Markdown tables from a pdfplumber Page."""
    table_markdowns: List[str] = []
    table_bboxes: List[tuple[float, float, float, float]] = []

    try:
        found_tables = page.find_tables()
        for t in found_tables:
            table_bboxes.append(t.bbox)
            raw_data = t.extract()
            if raw_data:
                md_table = format_table_as_markdown(raw_data)
                if md_table:
                    table_markdowns.append(md_table)
    except Exception as e:
        logger.warning(f"Error finding tables on page {page_num}: {e}")

    # Extract text excluding table areas to avoid duplication
    text = ""
    try:
        if table_bboxes:
            filtered_page = page.filter(
                lambda obj: not _is_inside_any_bbox(obj, table_bboxes)
            )
            text = filtered_page.extract_text(layout=False) or ""
        else:
            text = page.extract_text(layout=False) or ""
    except Exception as e:
        logger.warning(f"Error extracting filtered text on page {page_num}: {e}. Falling back to default.")
        text = page.extract_text() or ""

    text = text.strip()

    # Combine text and formatted tables into unified page content
    combined_parts = []
    if text:
        combined_parts.append(text)
    for idx, table_md in enumerate(table_markdowns, start=1):
        combined_parts.append(f"\n[Table {idx} on Page {page_num}]\n{table_md}\n")

    combined_content = "\n\n".join(combined_parts)

    return PageData(
        page_num=page_num,
        text=text,
        tables=table_markdowns,
        combined_content=combined_content,
        metadata={"table_count": len(table_markdowns), "char_count": len(combined_content)},
    )


def parse_pdf(
    file_source: Union[str, Path, bytes],
    filename: str = "document.pdf",
    use_cache: bool = True,
    progress_callback: Optional[Any] = None,
) -> ParsedDocument:
    """
    Parse a PDF file (supporting 300+ pages) with table extraction and disk caching.

    Args:
        file_source: Path to PDF file or raw bytes.
        filename: Name of the PDF file.
        use_cache: Whether to use disk caching by SHA-256 checksum.
        progress_callback: Optional callable (current_page, total_pages) -> None.

    Returns:
        ParsedDocument containing structured page data.
    """
    settings = get_settings()

    # Enforce maximum document size constraint (50 MB)
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if isinstance(file_source, bytes) and len(file_source) > max_bytes:
        raise ValueError(
            f"File size ({len(file_source) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {settings.max_file_size_mb} MB."
        )
    elif isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if p.exists() and p.stat().st_size > max_bytes:
            raise ValueError(
                f"File '{p.name}' ({p.stat().st_size / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {settings.max_file_size_mb} MB."
            )

    file_hash = compute_file_hash(file_source)
    cache_file = settings.cache_dir / f"{file_hash}_parsed.json"

    if use_cache and cache_file.exists():
        try:
            logger.info(f"Loading cached parsed PDF for hash: {file_hash}")
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return ParsedDocument.from_dict(cached_data)
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_file}: {e}. Reparsing...")

    logger.info(f"Parsing PDF '{filename}' (hash: {file_hash})...")
    pages: List[PageData] = []

    # If bytes passed, write to a temp or open directly via pdfplumber
    import io
    pdf_obj = io.BytesIO(file_source) if isinstance(file_source, bytes) else file_source

    with pdfplumber.open(pdf_obj) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            page_data = extract_page_content(page, page_num=i)
            pages.append(page_data)
            try:
                page.flush_cache()
            except Exception:
                pass
            if progress_callback:
                progress_callback(i, total_pages)

    parsed_doc = ParsedDocument(
        file_hash=file_hash,
        filename=filename,
        total_pages=len(pages),
        pages=pages,
        metadata={"source_hash": file_hash},
    )

    if use_cache:
        try:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(parsed_doc.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Cached parsed document to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to write cache {cache_file}: {e}")

    return parsed_doc

