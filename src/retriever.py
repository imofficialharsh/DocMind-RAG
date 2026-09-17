"""Hybrid retrieval pipeline combining Dense Vector Search and BM25 with Reciprocal Rank Fusion and CrossEncoder re-ranking."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from src.chunking import DocumentChunk
from src.config import AppSettings, get_settings
from src.indexer import IndexManager, tokenize_text

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk enriched with retrieval metrics and provenance."""

    chunk_id: str
    text: str
    page_num: int
    file_hash: str
    source_filename: str
    is_table: bool
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compute_rrf_scores(
    dense_ranked_ids: List[str],
    bm25_ranked_ids: List[str],
    k: int = 60,
) -> Dict[str, Tuple[float, Optional[int], Optional[int]]]:
    """
    Compute Reciprocal Rank Fusion (RRF) scores across two ranked lists.

    Returns:
        Dict mapping chunk_id -> (rrf_score, dense_rank_1indexed, bm25_rank_1indexed)
    """
    rrf_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}

    for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
        dense_ranks[chunk_id] = rank
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        bm25_ranks[chunk_id] = rank
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    results: Dict[str, Tuple[float, Optional[int], Optional[int]]] = {}
    for chunk_id, score in rrf_scores.items():
        results[chunk_id] = (
            score,
            dense_ranks.get(chunk_id),
            bm25_ranks.get(chunk_id),
        )

    return results


class HybridRetriever:
    """
    Performs hybrid retrieval using ChromaDB dense embeddings and BM25 keyword matching,
    followed by cross-encoder re-ranking.
    """

    def __init__(
        self,
        index_manager: Optional[IndexManager] = None,
        settings: Optional[AppSettings] = None,
    ):
        self.settings = settings or get_settings()
        self.index_manager = index_manager or IndexManager(self.settings)
        self._reranker: Optional[CrossEncoder] = None

    @property
    def reranker(self) -> CrossEncoder:
        """Lazy-load and return the CrossEncoder re-ranking model."""
        if self._reranker is None:
            logger.info(f"Loading re-ranker model: {self.settings.reranker_model_name}")
            self._reranker = CrossEncoder(self.settings.reranker_model_name)
        return self._reranker

    def _dense_search(
        self,
        query: str,
        file_hash: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Query ChromaDB dense vectors and return list of (id, text, metadata)."""
        k = top_k or self.settings.top_k_dense
        query_embedding = self.index_manager.embedder.encode([query]).tolist()

        where_filter = {"file_hash": file_hash} if file_hash else None

        results = self.index_manager.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas"],
        )

        dense_candidates: List[Tuple[str, str, Dict[str, Any]]] = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            for chunk_id, doc, meta in zip(ids, docs, metas):
                dense_candidates.append((chunk_id, doc, meta or {}))

        return dense_candidates

    def _bm25_search(
        self,
        query: str,
        file_hash: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Query BM25 index and return list of (id, text, metadata)."""
        k = top_k or self.settings.top_k_bm25
        bm25_store = self.index_manager.get_bm25_store(file_hash)
        if not bm25_store or not bm25_store.chunks:
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        # Enhance query tokens with corporate filing indicators if querying financials/performance
        lower_q = query.lower()
        financial_indicators = {"financial", "financials", "finance", "revenue", "profit", "ebitda", "margin", "results", "statement", "balance", "pat", "income", "sales"}
        if any(term in lower_q for term in financial_indicators):
            for exp in ["particulars", "fiscal", "gross", "ebitda", "profit", "restated", "margin"]:
                if exp not in tokenized_query:
                    tokenized_query.append(exp)

        scores = bm25_store.bm25.get_scores(tokenized_query)
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        bm25_candidates: List[Tuple[str, str, Dict[str, Any]]] = []
        for idx in scored_indices[:k]:
            if scores[idx] <= 0:
                continue
            chunk = bm25_store.chunks[idx]
            bm25_candidates.append(
                (
                    chunk.chunk_id,
                    chunk.text,
                    {
                        "page_num": chunk.page_num,
                        "chunk_id": chunk.chunk_id,
                        "file_hash": chunk.file_hash,
                        "source_filename": chunk.source_filename,
                        "chunk_index": chunk.chunk_index,
                        "is_table": chunk.is_table,
                    },
                )
            )

        return bm25_candidates

    def retrieve_hybrid(
        self,
        query: str,
        file_hash: str,
        top_k_dense: Optional[int] = None,
        top_k_bm25: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Perform hybrid retrieval using RRF fusion across dense and BM25 results."""
        dense_results = self._dense_search(query, file_hash=file_hash, top_k=top_k_dense)
        bm25_results = self._bm25_search(query, file_hash=file_hash, top_k=top_k_bm25)

        dense_ids = [c[0] for c in dense_results]
        bm25_ids = [c[0] for c in bm25_results]

        rrf_scores = compute_rrf_scores(
            dense_ranked_ids=dense_ids,
            bm25_ranked_ids=bm25_ids,
            k=rrf_k or self.settings.rrf_k,
        )

        # Store metadata lookup by chunk_id
        chunk_lookup: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for c_id, text, meta in dense_results:
            chunk_lookup[c_id] = (text, meta)
        for c_id, text, meta in bm25_results:
            if c_id not in chunk_lookup:
                chunk_lookup[c_id] = (text, meta)

        fused_chunks: List[RetrievedChunk] = []
        for chunk_id, (rrf_score, d_rank, b_rank) in rrf_scores.items():
            if chunk_id in chunk_lookup:
                text, meta = chunk_lookup[chunk_id]
                fused_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        page_num=meta.get("page_num", 1),
                        file_hash=meta.get("file_hash", file_hash),
                        source_filename=meta.get("source_filename", "document.pdf"),
                        is_table=meta.get("is_table", False),
                        rrf_score=rrf_score,
                        dense_rank=d_rank,
                        bm25_rank=b_rank,
                        metadata=meta,
                    )
                )

        # Sort by RRF score descending
        fused_chunks.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused_chunks

    def retrieve_and_rerank(
        self,
        query: str,
        file_hash: str,
        top_k_final: Optional[int] = None,
        candidate_pool_limit: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Execute full hybrid retrieval and CrossEncoder re-ranking with table preservation.

        Args:
            query: User's question or search query.
            file_hash: Target document hash to restrict search context.
            top_k_final: Number of final high-precision chunks to return.
            candidate_pool_limit: Max candidates to pass to the cross-encoder.

        Returns:
            List of top-ranked RetrievedChunk objects with rerank_score.
        """
        k_final = top_k_final or self.settings.top_k_final

        candidates = self.retrieve_hybrid(query=query, file_hash=file_hash)
        if not candidates:
            logger.warning(f"No candidates retrieved for query: '{query}'")
            return []

        # Restrict to candidate pool limit for efficient re-ranking
        candidates = candidates[:candidate_pool_limit]

        # Prepare pairs for cross-encoder
        sentence_pairs = [[query, c.text] for c in candidates]
        rerank_scores = self.reranker.predict(sentence_pairs)

        for chunk, score in zip(candidates, rerank_scores):
            chunk.rerank_score = float(score)

        # Sort all candidates by re-rank score descending
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)

        # Table-Diversity Preservation:
        # Cross-encoders naturally favor fluent text sentences over tabular cells.
        # Ensure that if relevant table chunks exist in the candidate pool, at least 2-3
        # top table chunks are preserved in final_chunks to prevent data starvation.
        table_candidates = [c for c in candidates if c.is_table and c.rerank_score > -5.0]
        final_top = candidates[:k_final]
        has_tables_in_top = any(c.is_table for c in final_top)

        if not has_tables_in_top and table_candidates:
            # Sort table candidates by rerank score
            table_candidates.sort(key=lambda x: x.rerank_score, reverse=True)
            tables_to_inject = table_candidates[: min(3, len(table_candidates))]

            # Keep the best text candidates and combine with top table chunks
            num_text_to_keep = max(1, k_final - len(tables_to_inject))
            final_chunks = candidates[:num_text_to_keep] + tables_to_inject
            final_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        else:
            final_chunks = final_top

        logger.info(
            f"Re-ranked {len(candidates)} candidates. Selected top {len(final_chunks)} chunks "
            f"with top score: {final_chunks[0].rerank_score:.4f} (Page {final_chunks[0].page_num})"
        )
        return final_chunks

