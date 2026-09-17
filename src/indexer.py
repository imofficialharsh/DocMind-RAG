"""Indexing pipeline managing ChromaDB vector persistence and serialized BM25 keyword indices."""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunking import DocumentChunk
from src.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 indexing."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25IndexStore:
    """Wrapper for BM25 index and corresponding DocumentChunks."""

    def __init__(self, bm25: BM25Okapi, chunks: List[DocumentChunk]):
        self.bm25 = bm25
        self.chunks = chunks

    def save(self, filepath: Path) -> None:
        """Serialize BM25 index and chunk references to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "BM25IndexStore":
        """Deserialize BM25 index from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class IndexManager:
    """Orchestrates persistent dense vector indexing in ChromaDB and sparse BM25 indexing."""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.settings.chroma_persist_dir)
        )
        self.collection_name = "docmind_chunks"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "DocMind Hybrid RAG persistent chunk repository"},
        )

        # Lazy-loaded SentenceTransformer model
        self._embedder: Optional[SentenceTransformer] = None
        self._bm25_cache: Dict[str, BM25IndexStore] = {}

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load and return the SentenceTransformer model."""
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model_name}")
            self._embedder = SentenceTransformer(self.settings.embedding_model_name)
        return self._embedder

    def is_document_indexed(self, file_hash: str) -> bool:
        """Check if a document's chunks are already present in ChromaDB and BM25 storage."""
        bm25_path = self.settings.bm25_persist_dir / f"{file_hash}_bm25.pkl"
        if not bm25_path.exists():
            return False

        # Query ChromaDB to see if any chunk with this file_hash exists
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1,
        )
        return bool(results and results.get("ids"))

    def index_document(
        self,
        chunks: List[DocumentChunk],
        file_hash: str,
        batch_size: int = 64,
    ) -> None:
        """
        Index document chunks into both ChromaDB and BM25.
        Idempotent: Skips re-indexing if document is already cached.
        """
        if not chunks:
            logger.warning(f"No chunks provided to index for file_hash {file_hash}")
            return

        if self.is_document_indexed(file_hash):
            logger.info(f"Document {file_hash} is already fully indexed. Skipping indexing.")
            return

        logger.info(f"Indexing {len(chunks)} chunks for document {file_hash}...")

        # 1. BM25 Indexing
        logger.info("Building BM25 keyword index...")
        tokenized_corpus = [tokenize_text(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_store = BM25IndexStore(bm25=bm25, chunks=chunks)
        bm25_path = self.settings.bm25_persist_dir / f"{file_hash}_bm25.pkl"
        bm25_store.save(bm25_path)
        self._bm25_cache[file_hash] = bm25_store
        logger.info(f"BM25 index saved to {bm25_path}")

        # 2. ChromaDB Dense Vector Indexing
        logger.info("Generating dense embeddings and upserting into ChromaDB...")
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "page_num": c.page_num,
                "chunk_id": c.chunk_id,
                "file_hash": c.file_hash,
                "source_filename": c.source_filename,
                "chunk_index": c.chunk_index,
                "is_table": c.is_table,
            }
            for c in chunks
        ]

        # Compute embeddings in batches to conserve memory
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        ).tolist()

        # Upsert into Chroma in batches
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            self.collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        logger.info(f"Successfully indexed {total_chunks} chunks into ChromaDB.")

    def get_bm25_store(self, file_hash: str) -> Optional[BM25IndexStore]:
        """Retrieve the BM25 index store for a document from memory cache or disk."""
        if file_hash in self._bm25_cache:
            return self._bm25_cache[file_hash]

        bm25_path = self.settings.bm25_persist_dir / f"{file_hash}_bm25.pkl"
        if bm25_path.exists():
            try:
                store = BM25IndexStore.load(bm25_path)
                self._bm25_cache[file_hash] = store
                return store
            except Exception as e:
                logger.error(f"Error loading BM25 index {bm25_path}: {e}")
                return None
        return None

    def get_all_indexed_hashes(self) -> List[str]:
        """Return list of all unique file hashes indexed in the collection."""
        res = self.collection.get(include=["metadatas"])
        if not res or not res.get("metadatas"):
            return []
        hashes = set()
        for meta in res["metadatas"]:
            if meta and "file_hash" in meta:
                hashes.add(meta["file_hash"])
        return list(hashes)

