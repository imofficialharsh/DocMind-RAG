# DocMind: Enterprise Document Intelligence RAG

DocMind is a production-ready, modular Document Intelligence Retrieval-Augmented Generation (RAG) platform designed to ingest complex, multi-page PDFs (up to 300+ pages), extract high-fidelity text and tables (formatted as Markdown), index them via Hybrid Search (BM25 + Dense Vectors), re-rank using Cross-Encoders, and generate strictly grounded answers with verifiable page-level citations.

---

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Parsing"]
        PDF["Input PDF (up to 300+ pages)"] --> Hash["SHA-256 Checksum Verification"]
        Hash --> CacheCheck{"Cached in disk?"}
        CacheCheck -- Yes --> LoadCache["Load Cached Parsed Document"]
        CacheCheck -- No --> Extractor["pdfplumber Engine"]
        Extractor --> TableExt["Extract Tables -> Markdown Tables"]
        Extractor --> TextExt["Extract Text (Filtered by Table Bounding Boxes)"]
        TableExt & TextExt --> StructuredDoc["Structured PageData Objects"]
        StructuredDoc --> SaveCache["Save to Cache (data/cache/)"]
    end

    subgraph Chunking ["2. Structure-Aware Chunking"]
        LoadCache & SaveCache --> Chunker["Recursive Chunker"]
        Chunker --> TableChunks["Table Chunks (Header Preserving)"]
        Chunker --> TextChunks["Paragraph & Semantic Text Chunks"]
        TableChunks & TextChunks --> ChunkProvenance["Enriched Chunks (chunk_id, page_num, file_hash)"]
    end

    subgraph Indexing ["3. Dual Hybrid Indexing"]
        ChunkProvenance --> ChromaDB["ChromaDB Vector Store (all-MiniLM-L6-v2)"]
        ChunkProvenance --> BM25["BM25 Index Builder (BM25Okapi)"]
        ChromaDB --> ChromaPersist[("./data/chroma_db")]
        BM25 --> BM25Persist[("./data/bm25/{hash}_bm25.pkl")]
    end

    subgraph Security ["4. Security & Request Throttling"]
        UserQuery["User Query"] --> RateLimit["30s Cooldown Rate Limiter\n(Server Load & API Budget Defense)"]
    end

    subgraph Retrieval ["5. Hybrid Retrieval & Re-ranking"]
        RateLimit --> DenseRetriever["ChromaDB Dense Search (Top-30)"]
        RateLimit --> BM25Retriever["BM25 Keyword Search + Expansion (Top-30)"]
        DenseRetriever & BM25Retriever --> RRF["Reciprocal Rank Fusion (RRF: k=60)"]
        RRF --> CandidatePool["Candidate Chunks (Top-60 Pool)"]
        CandidatePool --> CrossEncoder["Cross-Encoder (ms-marco-MiniLM-L-6-v2)"]
        CrossEncoder --> FinalTopK["High-Precision Top-10 Chunks"]
    end

    subgraph Generation ["6. Server-Managed 120B Generation & UI"]
        FinalTopK --> GroundingPrompt["Strict Grounding Prompt Template"]
        RateLimit --> GroundingPrompt
        GroundingPrompt --> GroqCloud["Groq Cloud LPU Inference\n(openai/gpt-oss-120b - Backend Managed)"]
        GroqCloud --> CitationExtractor["Citation & Provenance Mapper"]
        CitationExtractor --> StreamlitUI["Streamlit Interactive UI"]
    end
```

---

## Key Features

1. **High-Fidelity PDF & Table Ingestion**:
   - Uses `pdfplumber` to extract page layouts.
   - Converts tabular regions directly into GitHub-flavored Markdown tables.
   - Filters out table bounding boxes during text extraction to prevent duplicate or disjoint cell fragments.
   - Streaming page-by-page extraction handles 300+ page documents without memory exhaustion.

2. **Idempotent Caching**:
   - Computes SHA-256 digests of document byte content.
   - Parsed documents, BM25 indices, and vector embeddings are stored persistently on disk. Re-uploading the same PDF takes < 0.05 seconds.

3. **Hybrid Search via Reciprocal Rank Fusion (RRF)**:
   - Dense semantic vector search via HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (runs 100% locally on CPU or GPU with zero API cost).
   - Sparse keyword matching via `rank-bm25` (BM25Okapi).
   - Merges disparate ranking distributions using Reciprocal Rank Fusion:
     $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$

4. **Cross-Encoder Re-ranking**:
   - Evaluates fused candidate pairs `(query, chunk_text)` using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Computes cross-attention between question and document context, eliminating false-positive semantic matches.

5. **Enterprise Security & Rate Limiting**:
   - Enforces an automated **30-second cooldown per query** on the server.
   - Prevents automated spam/DDoS attempts, protects server CPU/memory, and prevents sudden API quota exhaustion.

6. **Server-Managed 120B Flagship Reasoning**:
   - Uses **Groq Cloud LPU** running the **120B parameter** model (`openai/gpt-oss-120b`).
   - Managed on the backend so HR and technical recruiters can evaluate the system immediately with zero setup or API key entry.

7. **Strict Grounding & Verifiable Citations**:
   - System prompts enforce that responses are derived strictly from retrieved context.
   - Demands inline brackets with exact page provenance: `[Page 14]`, `[Page 22, Table 1]`.
   - Streamlit UI renders interactive citation badges and full source chunk expandable cards.

---

## Latency Metrics & Performance

Evaluated on standard multi-page documents running on an 8-core CPU:

| Stage                                | Benchmark (per 50 pages)   | Optimization Applied                                       |
| :----------------------------------- | :------------------------- | :--------------------------------------------------------- |
| **PDF Text & Table Extraction**      | ~1.8s - 3.2s               | SHA-256 disk cache skips re-parse (0.02s on re-read)       |
| **Dense Vector Embeddings**          | ~2.1s (MiniLM-L6-v2)       | Batch encoding (`batch_size=64`) & PyTorch multi-threading |
| **BM25 Tokenization & Indexing**     | ~0.15s                     | Regex word tokenization & pickle serialization             |
| **Rate Limit Guard & Throttling**    | < 1ms                      | In-memory session tracking & cooldown verification         |
| **Hybrid Search (Dense + BM25)**     | ~45ms                      | Pre-indexed Chroma SQLite + in-memory BM25 index           |
| **Cross-Encoder Re-ranking**         | ~120ms (for 60 candidates) | Fast 6-layer MiniLM architecture                           |
| **LLM Inference (Groq 120B LPU)**    | ~300ms - 650ms             | Ultra-fast Groq LPU acceleration (`openai/gpt-oss-120b`)   |

---

## Project Structure

```
DocMind/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Pydantic BaseSettings & environment configs
│   ├── parser.py            # pdfplumber extractor & Markdown table converter
│   ├── chunking.py          # Recursive structure & table-aware chunker
│   ├── indexer.py           # ChromaDB & BM25 persistence manager
│   ├── retriever.py         # Hybrid search (Dense + BM25) + Cross-Encoder re-ranker
│   └── generator.py         # Grounded prompt engineering & multi-LLM client
├── tests/
│   ├── __init__.py
│   ├── test_parser.py       # Parser & caching tests
│   └── test_retrieval.py    # Chunking, RRF math & hybrid retrieval tests
├── app.py                   # Streamlit frontend application
├── requirements.txt         # Core dependencies
├── .env.example             # Configuration environment template
└── README.md                # Project documentation
```

---

## Installation & Setup

### 1. Clone & Set Up Virtual Environment

```bash
# Clone or navigate to the repository
cd DocMind

# Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# LLM Provider: "groq" (default) or "gemini"
LLM_PROVIDER=groq

# Groq Configuration (Server-managed backend inference)
# HR / evaluators can test immediately without providing an API key
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Query Rate Limiting (seconds cooldown between queries)
RATE_LIMIT_SECONDS=30

# Optional Google Gemini fallback:
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

---

## Running Tests

Run the full automated test suite:

```bash
pytest -v tests/
```

---

## Launching the Streamlit Application

Start the web interface:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to start analyzing documents!
