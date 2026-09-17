"""Tests for chunking, BM25 indexing, RRF fusion, and hybrid retrieval."""

from pathlib import Path
import pytest

from src.chunking import DocumentChunk, RecursiveDocumentChunker
from src.config import AppSettings
from src.indexer import IndexManager, tokenize_text
from src.parser import PageData, ParsedDocument
from src.retriever import HybridRetriever, compute_rrf_scores


def test_tokenize_text():
    """Verify BM25 tokenizer strips punctuation and lowercases."""
    text = "DocMind: Enterprise-grade Hybrid RAG, 2026!"
    tokens = tokenize_text(text)
    assert "docmind" in tokens
    assert "enterprise" in tokens
    assert "grade" in tokens
    assert "rag" in tokens
    assert "2026" in tokens
    assert ":" not in tokens
    assert "!" not in tokens


def test_compute_rrf_scores_math():
    """Verify Reciprocal Rank Fusion calculation with exact math."""
    dense_ids = ["chunk_a", "chunk_b"]
    bm25_ids = ["chunk_b", "chunk_c"]
    k = 60

    rrf = compute_rrf_scores(dense_ids, bm25_ids, k=k)

    # chunk_b is in both rankings (rank 2 dense, rank 1 bm25)
    expected_b = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
    score_b, dense_rank_b, bm25_rank_b = rrf["chunk_b"]
    assert pytest.approx(score_b, rel=1e-5) == expected_b
    assert dense_rank_b == 2
    assert bm25_rank_b == 1

    # chunk_a is only in dense (rank 1)
    expected_a = 1.0 / (60 + 1)
    score_a, dense_rank_a, bm25_rank_a = rrf["chunk_a"]
    assert pytest.approx(score_a, rel=1e-5) == expected_a
    assert dense_rank_a == 1
    assert bm25_rank_a is None

    # chunk_c is only in bm25 (rank 2)
    expected_c = 1.0 / (60 + 2)
    score_c, dense_rank_c, bm25_rank_c = rrf["chunk_c"]
    assert pytest.approx(score_c, rel=1e-5) == expected_c
    assert dense_rank_c is None
    assert bm25_rank_c == 2

    # Verify ranking order: chunk_b > chunk_a > chunk_c
    assert score_b > score_a > score_c


def test_recursive_document_chunker():
    """Verify chunker respects chunk size and preserves provenance metadata."""
    chunker = RecursiveDocumentChunker(chunk_size=200, chunk_overlap=40)

    long_text = (
        "DocMind is a modern RAG platform. " * 15  # approx 500 chars
    )
    table_md = "| Metric | Score |\n|---|---|\n| Accuracy | 99% |"

    page = PageData(
        page_num=3,
        text=long_text,
        tables=[table_md],
    )

    doc = ParsedDocument(
        file_hash="test_hash_123",
        filename="test_guide.pdf",
        total_pages=1,
        pages=[page],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1

    # Check table chunk
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) >= 1
    assert "Accuracy" in table_chunks[0].text
    assert table_chunks[0].page_num == 3
    assert table_chunks[0].file_hash == "test_hash_123"

    # Check text chunks
    text_chunks = [c for c in chunks if not c.is_table]
    assert len(text_chunks) >= 1
    for c in text_chunks:
        assert c.page_num == 3
        assert len(c.text) <= 250  # comfortably near chunk_size
        assert c.source_filename == "test_guide.pdf"
        assert c.chunk_id.startswith("test_has")


def test_index_and_hybrid_retrieval(tmp_path: Path):
    """End-to-end integration test: Indexing, Hybrid Search, and Re-ranking."""
    test_settings = AppSettings(
        chroma_persist_dir=tmp_path / "chroma",
        bm25_persist_dir=tmp_path / "bm25",
        cache_dir=tmp_path / "cache",
        chunk_size=300,
        chunk_overlap=50,
        top_k_dense=5,
        top_k_bm25=5,
        top_k_final=2,
    )
    test_settings.ensure_directories()

    index_mgr = IndexManager(settings=test_settings)
    retriever = HybridRetriever(index_manager=index_mgr, settings=test_settings)

    file_hash = "doc_eval_hash_999"
    chunks = [
        DocumentChunk(
            chunk_id=f"{file_hash[:8]}_p1_c0",
            text="The company generated $45 million in Q3 with net margin expanding to 22 percent.",
            page_num=1,
            file_hash=file_hash,
            source_filename="quarterly_report.pdf",
            chunk_index=0,
            is_table=False,
        ),
        DocumentChunk(
            chunk_id=f"{file_hash[:8]}_p2_c1",
            text="Our cloud infrastructure migrated entirely to Google Cloud Platform to reduce latency.",
            page_num=2,
            file_hash=file_hash,
            source_filename="quarterly_report.pdf",
            chunk_index=1,
            is_table=False,
        ),
        DocumentChunk(
            chunk_id=f"{file_hash[:8]}_p3_c2",
            text="Safety protocols require mandatory two-factor authentication for all remote employees.",
            page_num=3,
            file_hash=file_hash,
            source_filename="quarterly_report.pdf",
            chunk_index=2,
            is_table=False,
        ),
    ]

    # Index chunks
    assert not index_mgr.is_document_indexed(file_hash)
    index_mgr.index_document(chunks=chunks, file_hash=file_hash)
    assert index_mgr.is_document_indexed(file_hash)

    # Query hybrid retrieval for financial performance
    query = "What was the revenue and margin in Q3?"
    top_results = retriever.retrieve_and_rerank(
        query=query,
        file_hash=file_hash,
        top_k_final=2,
    )

    assert len(top_results) > 0
    # Top chunk should be the financial one from Page 1
    assert top_results[0].page_num == 1
    assert "45 million" in top_results[0].text
    assert top_results[0].rerank_score != 0.0
    assert top_results[0].rrf_score > 0.0

