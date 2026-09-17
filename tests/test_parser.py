"""Tests for PDF parsing, Markdown table conversion, and disk caching."""

import io
from pathlib import Path
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from src.parser import (
    compute_file_hash,
    format_table_as_markdown,
    parse_pdf,
    ParsedDocument,
)


def create_sample_pdf_bytes() -> bytes:
    """Generate a multi-page PDF with text and structured tables in memory."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Page 1: Heading and narrative text
    story.append(Paragraph("DocMind Enterprise Evaluation Report", styles["Heading1"]))
    story.append(
        Paragraph(
            "This document presents the quarterly performance benchmarks for DocMind RAG. "
            "Dense embeddings combined with BM25 keyword matching provide superior retrieval precision.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 20))

    # Page 1 Table
    table_data = [
        ["Metric", "BM25 Baseline", "Dense Vector", "Hybrid + RRF"],
        ["Precision@5", "0.64", "0.72", "0.89"],
        ["Recall@5", "0.58", "0.68", "0.85"],
        ["MRR", "0.61", "0.74", "0.91"],
    ]
    t = Table(table_data)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(t)

    # Page 2: Second page
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    story.append(Paragraph("Financial Summary and Cost Structure", styles["Heading2"]))
    story.append(
        Paragraph(
            "Local embedding inference via all-MiniLM-L6-v2 eliminates embedding API costs completely. "
            "Re-ranking with ms-marco-MiniLM-L-6-v2 adds negligible CPU latency while boosting top-1 accuracy.",
            styles["Normal"],
        )
    )

    doc.build(story)
    return buf.getvalue()


def test_format_table_as_markdown():
    """Verify raw tables are converted to GitHub Flavored Markdown."""
    raw_table = [
        ["Quarter", "Revenue", "Margin"],
        ["Q1", "$1.2M", "24%"],
        ["Q2", "$1.5M", "28%"],
    ]
    md = format_table_as_markdown(raw_table)
    assert md is not None
    assert "| Quarter" in md
    assert "| Revenue" in md
    assert "|---" in md
    assert "| Q1" in md
    assert "| $1.2M" in md


def test_format_table_empty():
    """Verify empty or None table returns None."""
    assert format_table_as_markdown([]) is None
    assert format_table_as_markdown([[], []]) is None


def test_compute_file_hash():
    """Verify deterministic SHA-256 hash generation."""
    sample_data = b"DocMind Test PDF Content"
    hash1 = compute_file_hash(sample_data)
    hash2 = compute_file_hash(sample_data)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_parse_pdf_and_caching(tmp_path: Path, monkeypatch):
    """Verify multi-page parsing, table extraction, and disk caching."""
    pdf_bytes = create_sample_pdf_bytes()
    test_cache_dir = tmp_path / "cache"

    from src import config
    settings = config.get_settings()
    monkeypatch.setattr(settings, "cache_dir", test_cache_dir)

    # First parse: should parse and write cache
    doc1 = parse_pdf(pdf_bytes, filename="test_report.pdf", use_cache=True)
    assert doc1.total_pages == 2
    assert len(doc1.pages) == 2
    assert doc1.pages[0].page_num == 1
    assert doc1.pages[1].page_num == 2

    # Check that text and table content are present
    assert "DocMind Enterprise Evaluation Report" in doc1.pages[0].text or any(
        "DocMind" in p.text for p in doc1.pages
    )
    assert len(doc1.pages[0].tables) >= 1
    assert "Precision@5" in doc1.pages[0].tables[0]

    # Verify cache file was created
    cache_file = test_cache_dir / f"{doc1.file_hash}_parsed.json"
    assert cache_file.exists()

    # Second parse: should load directly from cache
    doc2 = parse_pdf(pdf_bytes, filename="test_report.pdf", use_cache=True)
    assert doc2.file_hash == doc1.file_hash
    assert doc2.total_pages == doc1.total_pages
    assert len(doc2.pages[0].tables) == len(doc1.pages[0].tables)

