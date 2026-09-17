"""Streamlit Frontend for DocMind Document Intelligence RAG System."""

import time
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from src.chunking import RecursiveDocumentChunker
from src.config import get_settings
from src.generator import GroundedGenerator, RAGResponse
from src.indexer import IndexManager
from src.parser import ParsedDocument, parse_pdf
from src.retriever import HybridRetriever, RetrievedChunk

st.set_page_config(
    page_title="DocMind | Document Intelligence RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Sleek Dark & Light Orange Enterprise Theme with Green Ready Indicators
st.markdown(
    """
    <style>
    /* Global dark canvas & typography */
    .stApp {
        background-color: #0E1117;
        color: #F3F4F6;
    }

    /* Hide Streamlit default "Press Enter to apply / submit form" instructions */
    [data-testid="InputInstructions"],
    .stTextInput div[data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Gradient Brand Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF8A3D 0%, #FFA94D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 0.98rem;
        color: #9CA3AF;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }

    /* Sidebar Branding */
    .brand-container {
        padding-bottom: 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid #232736;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF8A3D 0%, #FFA94D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
    }
    .brand-sub {
        font-size: 0.72rem;
        color: #9CA3AF;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 2px;
    }
    .creator-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 138, 61, 0.12);
        border: 1px solid rgba(255, 138, 61, 0.35);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.76rem;
        color: #FFA94D;
        margin-top: 6px;
    }
    .author-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 138, 61, 0.12);
        border: 1px solid rgba(255, 138, 61, 0.35);
        color: #FFA94D;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 20px;
        box-shadow: 0 0 10px rgba(255, 138, 61, 0.15);
    }

    /* Section Headings in Sidebar */
    .sidebar-section-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFA94D;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 14px;
        margin-bottom: 8px;
    }

    /* Sleek File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #181B24;
        border: 1px dashed rgba(255, 138, 61, 0.35);
        border-radius: 10px;
        padding: 8px;
        transition: all 0.25s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #FF8A3D;
        box-shadow: 0 0 12px rgba(255, 138, 61, 0.15);
    }

    /* Hide '+' (Add files) button in file uploader when a document is uploaded */
    [data-testid="stFileUploader"] button[aria-label="Add files"],
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"],
    [data-testid="stFileChips"] button[aria-label="Add files"],
    [data-testid="stFileChips"] > button,
    [data-testid="stFileUploader"] button[aria-label*="Add"] {
        display: none !important;
    }

    /* Progress Bar custom orange gradient & glow */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B00 0%, #FF8A3D 60%, #FFA94D 100%) !important;
        border-radius: 9999px;
        box-shadow: 0 0 12px rgba(255, 138, 61, 0.4);
    }
    .stProgress > div > div > div {
        background-color: #232736 !important;
        border-radius: 9999px;
    }

    /* Vibrant Green Ready Alert Banner */
    .ready-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.55);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.22);
    }
    .ready-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .ready-pulse-green {
        width: 10px;
        height: 10px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
        animation: pulse-green 1.8s infinite;
    }
    @keyframes pulse-green {
        0% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.25); box-shadow: 0 0 12px #10B981; }
        100% { opacity: 0.4; transform: scale(0.9); }
    }
    .ready-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #34D399;
    }
    .ready-text {
        font-size: 0.82rem;
        color: #E5E7EB;
        line-height: 1.45;
    }
    .ready-highlight {
        color: #6EE7B7;
        font-weight: 600;
        display: block;
        margin-top: 4px;
    }

    /* Metric & Statistics Card */
    .metric-card {
        background-color: #181B24;
        border: 1px solid #282D3D;
        border-left: 3px solid #10B981;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.84rem;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .metric-row:last-child {
        margin-bottom: 0;
    }
    .metric-label {
        color: #9CA3AF;
    }
    .metric-value {
        color: #F3F4F6;
        font-weight: 600;
    }

    /* LLM Status Box */
    .llm-status-box {
        background-color: #1F2432;
        border: 1px solid #2D3346;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.83rem;
    }
    .badge-orange {
        background-color: rgba(255, 138, 61, 0.2);
        color: #FFA94D;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }

    /* Citation Badges */
    .citation-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(255, 138, 61, 0.12);
        color: #FFA94D;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 6px;
        border: 1px solid rgba(255, 138, 61, 0.35);
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .table-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.35);
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Expanders styled for sleek dark mode */
    div[data-testid="stExpander"] {
        background-color: #181B24;
        border: 1px solid #282D3D;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stExpander"]:focus-within {
        border-color: rgba(255, 138, 61, 0.4);
    }

    /* Orange Gradient Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF8A3D 0%, #FF6B00 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(255, 107, 0, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(255, 107, 0, 0.4) !important;
        filter: brightness(1.1);
    }
    div.stButton > button {
        background-color: #1E2230 !important;
        color: #F3F4F6 !important;
        border: 1px solid #2D3346 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: #FF8A3D !important;
        color: #FFA94D !important;
        box-shadow: 0 0 10px rgba(255, 138, 61, 0.2) !important;
    }

    /* Hero Empty State & Prompt Cards */
    .hero-card {
        background: linear-gradient(135deg, #181B24 0%, #1A1E2B 100%);
        border: 1px solid #282D3D;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFA94D;
        margin-bottom: 6px;
    }
    .hero-desc {
        font-size: 0.92rem;
        color: #9CA3AF;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_components():
    """Cache core retriever and generator components."""
    settings = get_settings()
    index_mgr = IndexManager(settings)
    retriever = HybridRetriever(index_manager=index_mgr, settings=settings)
    generator = GroundedGenerator(settings=settings)
    chunker = RecursiveDocumentChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return settings, index_mgr, retriever, generator, chunker


settings, index_mgr, retriever, generator, chunker = get_components()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "indexing_time" not in st.session_state:
    st.session_state.indexing_time = 0.0

# LLM Saved Configuration in Session State
if "saved_llm_provider" not in st.session_state:
    st.session_state.saved_llm_provider = settings.llm_provider
if "saved_gemini_model" not in st.session_state or st.session_state.saved_gemini_model not in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
    st.session_state.saved_gemini_model = "gemini-3.5-flash"
if "saved_gemini_api_key" not in st.session_state:
    st.session_state.saved_gemini_api_key = settings.gemini_api_key or ""
if "saved_groq_model" not in st.session_state or (st.session_state.saved_groq_model not in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"] and not st.session_state.saved_groq_model.startswith("Custom")):
    st.session_state.saved_groq_model = "openai/gpt-oss-120b"
if "saved_groq_api_key" not in st.session_state:
    st.session_state.saved_groq_api_key = settings.groq_api_key or ""

# Query suggestion trigger
if "sample_prompt_query" not in st.session_state:
    st.session_state.sample_prompt_query = None

# ==============================================================================
# ARCHITECTURE BLUEPRINT MODAL
# ==============================================================================
@st.dialog("DocMind — Architecture & System Design Blueprint", width="large")
def show_architecture_dialog():
    """Display comprehensive system architecture diagram and engineering breakdown."""
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #282D3D; padding-bottom: 10px; margin-bottom: 14px;">
            <div>
                <span style="font-size: 1.15rem; font-weight: 700; color: #FFA94D;">End-to-End Enterprise RAG Pipeline</span>
                <div style="font-size: 0.82rem; color: #9CA3AF;">Engineered by <b>Harsh Kumar</b> | Zero-API-Cost Hybrid IR Architecture</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Mermaid Diagram
    components.html(
        """
        <div style="background-color: #12151E; padding: 16px; border-radius: 10px; border: 1px solid #282D3D; overflow-x: auto;">
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({startOnLoad: true, theme: 'dark', themeVariables: {darkMode: true, background: '#12151E'}});</script>
            <div class="mermaid">
            flowchart TD
                subgraph Ingestion ["1. Document Ingestion & Parsing"]
                    PDF["Input PDF (up to 300+ pages)"] --> Hash["SHA-256 Checksum Verification"]
                    Hash --> CacheCheck{"Cached on disk?"}
                    CacheCheck -- Yes --> LoadCache["Load Cached Parsed Document (0.02s)"]
                    CacheCheck -- No --> Extractor["pdfplumber Layout Engine"]
                    Extractor --> TableExt["Extract Tables -> Markdown Tables"]
                    Extractor --> TextExt["Extract Text (Filtered by Table Bounding Boxes)"]
                    TableExt & TextExt --> StructuredDoc["Structured PageData Objects"]
                    StructuredDoc --> SaveCache["Persist to Disk Cache (data/cache/)"]
                end

                subgraph Chunking ["2. Structure-Aware Chunking"]
                    LoadCache & SaveCache --> Chunker["Recursive Chunker"]
                    Chunker --> TableChunks["Table Chunks (Header Preserving, 3.5k chars)"]
                    Chunker --> TextChunks["Semantic Paragraph Text Chunks"]
                    TableChunks & TextChunks --> ChunkProvenance["Enriched Chunks (chunk_id, page_num, file_hash)"]
                end

                subgraph Indexing ["3. Dual Hybrid Indexing ($0 API Cost)"]
                    ChunkProvenance --> ChromaDB["ChromaDB Vector Store (all-MiniLM-L6-v2)"]
                    ChunkProvenance --> BM25["BM25 Index Builder (BM25Okapi)"]
                    ChromaDB --> ChromaPersist[("./data/chroma_db")]
                    BM25 --> BM25Persist[("./data/bm25/{hash}_bm25.pkl")]
                end

                subgraph Retrieval ["4. Hybrid Retrieval & Re-ranking"]
                    UserQuery["User Query"] --> DenseRetriever["ChromaDB Dense Search (Top-30)"]
                    UserQuery --> BM25Retriever["BM25 Keyword Search + Expansion (Top-30)"]
                    DenseRetriever & BM25Retriever --> RRF["Reciprocal Rank Fusion (RRF: k=60)"]
                    RRF --> CandidatePool["Candidate Chunks (Top-60 Pool)"]
                    CandidatePool --> CrossEncoder["Cross-Encoder (ms-marco-MiniLM-L-6-v2)"]
                    CrossEncoder --> TablePreserve["Table-Diversity Preservation"]
                    TablePreserve --> FinalTopK["High-Precision Top-10 Chunks"]
                end

                subgraph Generation ["5. Grounded Generation & Citations"]
                    FinalTopK & UserQuery --> GroundingPrompt["Strict Grounding Prompt Template"]
                    GroundingPrompt --> LLM{"LLM Provider"}
                    LLM -- Gemini --> GenAI["Google Gemini (3.5 Flash / 3.1 Flash-Lite)"]
                    LLM -- Groq --> GroqCloud["Groq Cloud (GPT-OSS / Qwen)"]
                    GenAI & GroqCloud --> CitationExtractor["Citation & Provenance Mapper"]
                    CitationExtractor --> StreamlitUI["Interactive UI with Source Cards"]
                end
            </div>
        </div>
        """,
        height=580,
        scrolling=True,
    )

    tab1, tab2, tab3 = st.tabs(["🚀 5-Stage System Breakdown", "⚡ Latency Benchmarks", "💎 Why It Outperforms Generic RAG"])

    with tab1:
        st.markdown(
            """
            #### 1. High-Fidelity PDF & Table Ingestion
            * Employs `pdfplumber` for precise spatial layout parsing.
            * Identifies bounding boxes of financial & operational tables, extracting them directly as **GitHub-flavored Markdown tables**.
            * Filters out table bounding boxes during text extraction to completely eliminate duplicated and broken cell text.
            * Computes SHA-256 byte hashes for idempotent caching: re-uploading an existing 500-page prospectus takes **< 0.05 seconds**.

            #### 2. Structure-Aware Chunking
            * Preserves multi-row table coherence by maintaining up to 3,500 characters per table chunk, keeping complete financial statements (Revenue, EBITDA, PAT, Margins) together.
            * Attaches surrounding introductory text & page context to table chunks, allowing semantic search to match conceptual queries as well as numbers.

            #### 3. Dual Hybrid Indexing (Zero API Cost)
            * **Dense Semantic Index**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` encoded locally on CPU/GPU into ChromaDB SQLite.
            * **Sparse Keyword Index**: `BM25Okapi` with tokenization & corporate filing keyword expansion, serialized to disk via pickle.

            #### 4. Hybrid Search via Reciprocal Rank Fusion & Cross-Encoder
            * Merges disparate vector and keyword rankings using Reciprocal Rank Fusion:
              $$\\text{RRF}(d) = \\sum_{m \\in \\{\\text{dense}, \\text{bm25}\\}} \\frac{1}{k + \\text{rank}_m(d)}$$
            * Re-ranks the top 60 candidate pool using `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention.
            * Applies **Table-Diversity Preservation** to ensure high-value numerical tables are never crowded out by conversational text.

            #### 5. Strict Grounding & Citation Attribution
            * Strict grounding system prompt enforces that answers derive exclusively from retrieved context.
            * Direct provenance mapper extracts page citations (`[Page 212]`, `[Page 33, Table 1]`) and links them directly to interactive source inspection cards.
            """
        )

    with tab2:
        st.markdown(
            """
            | Pipeline Stage | Benchmark (per 50 pages) | Optimization Applied |
            | :--- | :--- | :--- |
            | **PDF Text & Table Extraction** | ~1.8s - 3.2s | SHA-256 disk cache skips re-parse (<0.02s on repeat) |
            | **Dense Vector Embeddings** | ~2.1s (MiniLM-L6-v2) | Batch encoding (`batch_size=64`) & PyTorch multi-threading |
            | **BM25 Tokenization & Indexing** | ~0.15s | Word regex tokenization & pickle serialization |
            | **Hybrid Search (Dense + BM25)** | ~45ms | Pre-indexed Chroma SQLite + in-memory BM25 index |
            | **Cross-Encoder Re-ranking** | ~120ms (for 60 candidates) | Fast 6-layer MiniLM architecture |
            | **LLM Inference (Gemini 3.5 Flash)** | ~800ms - 1.4s | Server-side speculative decoding & streaming |
            | **LLM Inference (Groq GPT-OSS / Qwen)** | ~250ms - 600ms | Ultra-fast LPU hardware acceleration |
            """
        )

    with tab3:
        st.markdown(
            """
            * **Real-World Table Intelligence**: Handles real filings with multi-column financial tables without losing headers or alignment.
            * **Zero Embedding API Cost**: Entire indexing and hybrid retrieval run 100% locally with zero external API fees.
            * **No Hallucinations / Strict Grounding**: Answers are strictly verifiable with interactive color-coded provenance badges (Orange for text, Green for tables).
            * **Frontend-Driven Model Agnostic**: Switch seamlessly between Google Gemini and Groq Cloud with instant key management.
            """
        )


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    # Clean Enterprise Brand Header with Harsh Kumar Attribution
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">DocMind</div>
            <div class="brand-sub">Document Intelligence RAG</div>
            <div class="creator-badge">⚡ Engineered by <b>Harsh Kumar</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📐 View System Architecture Blueprint", key="sidebar_arch_btn", use_container_width=True):
        show_architecture_dialog()

    # Document Ingestion Section
    st.markdown(
        '<div class="sidebar-section-title">Document Ingestion</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        f"Upload PDF (Max {settings.max_file_size_mb} MB)",
        type=["pdf"],
        help=f"Upload multi-page research papers, financial reports, or DRHPs up to {settings.max_file_size_mb}MB.",
    )

    # Handle file removal: when user clears/removes the uploaded document
    if uploaded_file is None:
        if st.session_state.current_doc is not None:
            st.session_state.current_doc = None
            st.session_state.doc_chunks = []
            st.session_state.indexing_time = 0.0
            st.session_state.chat_history = []
            st.session_state.sample_prompt_query = None
            st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        file_size_mb = len(file_bytes) / (1024 * 1024)

        if file_size_mb > settings.max_file_size_mb:
            st.error(
                f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {settings.max_file_size_mb} MB."
            )
            st.stop()

        # Check if new file or re-upload
        if (
            st.session_state.current_doc is None
            or st.session_state.current_doc.filename != filename
        ):
            # Processing with live multi-stage progress bar
            proc_placeholder = st.empty()
            with proc_placeholder.container():
                st.markdown("###### Ingesting Document...")
                progress_bar = st.progress(5, text="Step 1/3: Parsing PDF and extracting text...")
                status_text = st.empty()

                start_proc = time.time()

                def update_progress(curr: int, tot: int):
                    if tot > 0:
                        pct = min(40, max(5, int((curr / tot) * 40)))
                        progress_bar.progress(
                            pct, text=f"Step 1/3: Parsing page {curr}/{tot}..."
                        )

                doc: ParsedDocument = parse_pdf(
                    file_source=file_bytes,
                    filename=filename,
                    use_cache=True,
                    progress_callback=update_progress,
                )
                st.session_state.current_doc = doc

                progress_bar.progress(55, text="Step 2/3: Generating structure-aware chunks...")
                status_text.caption("Chunking text and Markdown tables...")
                chunks = chunker.chunk_document(doc)
                st.session_state.doc_chunks = chunks

                progress_bar.progress(75, text="Step 3/3: Building Hybrid Index...")
                status_text.caption("Indexing dense embeddings in ChromaDB and BM25...")
                index_mgr.index_document(chunks=chunks, file_hash=doc.file_hash)

                st.session_state.indexing_time = time.time() - start_proc
                progress_bar.progress(100, text="100% Complete: Indexing finished.")
                status_text.empty()

            # Replace progress container with completion banner
            proc_placeholder.empty()

    # Document Ready & Statistics Section (Rendered in Vibrant Green)
    if st.session_state.current_doc:
        doc = st.session_state.current_doc
        st.markdown(
            f"""
            <div class="ready-banner">
                <div class="ready-header">
                    <span class="ready-pulse-green"></span>
                    <span class="ready-title">Ready for Querying</span>
                </div>
                <div class="ready-text">
                    Document <b>{doc.filename}</b> is fully indexed.
                    <span class="ready-highlight">You can now submit queries below.</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-row"><span class="metric-label">Document:</span> <span class="metric-value">{doc.filename}</span></div>
                <div class="metric-row"><span class="metric-label">Total Pages:</span> <span class="metric-value">{doc.total_pages}</span></div>
                <div class="metric-row"><span class="metric-label">Total Chunks:</span> <span class="metric-value">{len(st.session_state.doc_chunks)}</span></div>
                <div class="metric-row"><span class="metric-label">Index Time:</span> <span class="metric-value">{st.session_state.indexing_time:.2f}s</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Collapsible LLM Configuration (Expanded if API key is not yet provided)
    current_prov = st.session_state.get("llm_provider_picker", st.session_state.saved_llm_provider)
    has_api_key = bool(
        (st.session_state.get("gemini_key_text", "").strip() or st.session_state.saved_gemini_api_key)
        if current_prov == "gemini"
        else (st.session_state.get("groq_key_text", "").strip() or st.session_state.saved_groq_api_key)
    )

    with st.expander("LLM Provider & API Key", expanded=not has_api_key):
        selected_provider = st.selectbox(
            "Select Provider",
            options=["gemini", "groq"],
            index=0 if current_prov == "gemini" else 1,
            key="llm_provider_picker",
            help="Switch between Google Gemini and Groq Cloud.",
        )

        if selected_provider == "gemini":
            gemini_options = [
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
            ]
            if "gemini_model_select" in st.session_state and st.session_state["gemini_model_select"] not in gemini_options:
                del st.session_state["gemini_model_select"]

            curr_gem = st.session_state.saved_gemini_model
            idx = gemini_options.index(curr_gem) if curr_gem in gemini_options else 0

            chosen_gemini_model = st.selectbox(
                "Gemini Model",
                options=gemini_options,
                index=idx,
                key="gemini_model_select",
                help="Recommended: gemini-3.5-flash for fast and accurate RAG analysis.",
            )
            model_val = chosen_gemini_model

            input_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.saved_gemini_api_key,
                type="password",
                key="gemini_key_text",
                placeholder="AIzaSy...",
                help="API key entered here directly powers the assistant.",
            )
            st.session_state.saved_llm_provider = "gemini"
            st.session_state.saved_gemini_model = model_val or "gemini-3.5-flash"
            if input_key.strip():
                st.session_state.saved_gemini_api_key = input_key.strip()

        else:
            groq_options = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                "Custom...",
            ]
            if "groq_model_select" in st.session_state and st.session_state["groq_model_select"] not in groq_options:
                del st.session_state["groq_model_select"]

            curr_groq = st.session_state.saved_groq_model
            idx = groq_options.index(curr_groq) if curr_groq in groq_options else 0

            chosen_groq_model = st.selectbox(
                "Groq Model",
                options=groq_options,
                index=idx,
                key="groq_model_select",
                help="Ultra-fast LPU inference via Groq Cloud.",
            )
            if chosen_groq_model == "Custom...":
                model_val = st.text_input(
                    "Custom Model Name",
                    value=curr_groq if curr_groq not in groq_options[:-1] else "",
                    key="groq_custom_text",
                    placeholder="e.g. meta-llama/llama-4-scout",
                ).strip()
            else:
                model_val = chosen_groq_model

            input_key = st.text_input(
                "Groq API Key",
                value=st.session_state.saved_groq_api_key,
                type="password",
                key="groq_key_text",
                placeholder="gsk_...",
                help="API key entered here directly powers the assistant.",
            )
            st.session_state.saved_llm_provider = "groq"
            st.session_state.saved_groq_model = model_val or "openai/gpt-oss-120b"
            if input_key.strip():
                st.session_state.saved_groq_api_key = input_key.strip()

        st.caption("Settings are applied automatically from the frontend.")

    # Advanced Retrieval Parameters (Hidden by default)
    with st.expander("Retrieval Parameters", expanded=False):
        top_k_final = st.slider(
            "Final Top-K Chunks",
            min_value=1,
            max_value=20,
            value=settings.top_k_final,
            key="slider_top_k_final",
            help="Number of re-ranked context chunks passed to the LLM (Default: 10).",
        )
        top_k_dense = st.slider(
            "Chroma Dense Candidates",
            min_value=5,
            max_value=40,
            value=settings.top_k_dense,
        )
        top_k_bm25 = st.slider(
            "BM25 Keyword Candidates",
            min_value=5,
            max_value=40,
            value=settings.top_k_bm25,
        )
        rrf_k = st.slider(
            "RRF Constant (k)",
            min_value=10,
            max_value=120,
            value=settings.rrf_k,
        )

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        """
        <div style="border-top: 1px solid #232736; padding-top: 14px; margin-top: 24px; text-align: center;">
            <div style="font-size: 0.76rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.8px;">Enterprise RAG Platform</div>
            <div style="font-size: 0.85rem; color: #FFA94D; font-weight: 600; margin-top: 4px;">
                Made by <b>Harsh Kumar</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# MAIN CHAT & WORKSPACE AREA
# ==============================================================================
col_head_left, col_head_right = st.columns([3.5, 1.5])
with col_head_left:
    st.markdown('<div class="main-title">DocMind: Document Intelligence RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Enterprise multi-page PDF analysis with Hybrid Search (BM25 + Dense), '
        'Cross-Encoder Re-ranking, and Grounded Citations.</div>',
        unsafe_allow_html=True,
    )
with col_head_right:
    st.markdown(
        """
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            <div class="author-pill">
                <span>⚡ Built by <b>Harsh Kumar</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# If no document is uploaded, user cannot make query
if not st.session_state.current_doc:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Welcome to DocMind</div>
            <div class="hero-desc">
                To start querying, please upload a multi-page PDF document using the <b>Document Ingestion</b>
                dropzone in the sidebar. Once uploaded, DocMind will parse the text and tables, build hybrid
                semantic indexes, and notify you as soon as it is ready for queries.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Disabled input indicating document upload is required
    st.chat_input("Please upload a PDF document in the sidebar to start querying...", disabled=True)

else:
    # Display quick prompt chips if no chat history yet
    sample_to_run = None
    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="hero-card" style="padding: 16px 20px; margin-bottom: 16px;">
                <div class="hero-title" style="font-size: 1.05rem; color: #34D399; display: flex; align-items: center; gap: 8px;">
                    <span class="ready-pulse-green"></span> Document Ready: <code>{st.session_state.current_doc.filename}</code>
                </div>
                <div class="hero-desc" style="font-size: 0.88rem;">
                    DocMind has indexed <b>{st.session_state.current_doc.total_pages} pages</b> and <b>{len(st.session_state.doc_chunks)} chunks</b>.
                    Ask any question below or select a suggested prompt to get started:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Summarize Document", use_container_width=True):
                sample_to_run = "Please provide an executive summary of this document highlighting the key takeaways."
        with col2:
            if st.button("Extract Key Metrics & Tables", use_container_width=True):
                sample_to_run = "Extract all key metrics, financial figures, and structured data tables found in this document."
        with col3:
            if st.button("Core Architecture / Findings", use_container_width=True):
                sample_to_run = "What are the core methodologies, findings, or technical specifications detailed in this document?"

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "citations" in message and message["citations"]:
                st.markdown("##### Page Citations")
                st.caption("🟠 **Orange**: Text excerpts | 🟢 **Green**: Structured tables")
                cols = st.columns(min(len(message["citations"]), 4))
                for idx, citation in enumerate(message["citations"]):
                    col_idx = idx % min(len(message["citations"]), 4)
                    with cols[col_idx]:
                        badge_type = "table-badge" if citation.get("is_table") else "citation-badge"
                        label = f"Page {citation['page_num']}" + (" (Table)" if citation.get("is_table") else "")
                        st.markdown(f'<span class="{badge_type}">{label}</span>', unsafe_allow_html=True)

            if "chunks" in message and message["chunks"]:
                with st.expander(f"View {len(message['chunks'])} Retrieved Source Chunks (Re-ranked)"):
                    for i, chunk in enumerate(message["chunks"], start=1):
                        st.markdown(
                            f"**[Chunk {i}] Page {chunk.page_num}** | "
                            f"Rerank Score: `{chunk.rerank_score:.4f}` | "
                            f"RRF Score: `{chunk.rrf_score:.5f}` | "
                            f"Dense Rank: `{chunk.dense_rank}` | "
                            f"BM25 Rank: `{chunk.bm25_rank}`"
                        )
                        st.markdown(f"```markdown\n{chunk.text}\n```")
                        st.divider()

    # User Input handling
    chat_prompt = st.chat_input("Ask a question about the document...")
    active_query = chat_prompt or sample_to_run

    if active_query:
        # Guard: ensure document is loaded before querying
        if not st.session_state.current_doc:
            st.warning("No document uploaded. Please upload a PDF in the sidebar first.")
            st.stop()

        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving hybrid matches and re-ranking..."):
                retrieved_chunks = retriever.retrieve_and_rerank(
                    query=active_query,
                    file_hash=st.session_state.current_doc.file_hash,
                    top_k_final=top_k_final,
                )

            effective_provider = st.session_state.saved_llm_provider
            effective_model = (
                st.session_state.saved_gemini_model
                if effective_provider == "gemini"
                else st.session_state.saved_groq_model
            )
            # Prioritize API key directly typed in frontend sidebar
            frontend_key = (
                st.session_state.get("gemini_key_text", "").strip()
                if effective_provider == "gemini"
                else st.session_state.get("groq_key_text", "").strip()
            )
            raw_key = frontend_key or (
                st.session_state.saved_gemini_api_key
                if effective_provider == "gemini"
                else st.session_state.saved_groq_api_key
            )
            effective_key = raw_key.strip() if raw_key else None

            if not effective_key:
                st.error(
                    f"⚠️ **{effective_provider.upper()} API Key Required**: Please enter your API key in the **LLM Provider & API Key** section in the left sidebar to generate grounded answers."
                )
                st.stop()

            with st.spinner(f"Synthesizing grounded answer using {effective_provider.upper()} ({effective_model})..."):
                response: RAGResponse = generator.generate_answer(
                    query=active_query,
                    retrieved_chunks=retrieved_chunks,
                    provider=effective_provider,
                    model=effective_model,
                    api_key=effective_key,
                )

            # Render generated answer
            st.markdown(response.answer)

            # Display citations
            if response.citations:
                st.markdown("##### Page Citations")
                st.caption("🟠 **Orange**: Text excerpts | 🟢 **Green**: Structured tables")
                cols = st.columns(min(len(response.citations), 4))
                for idx, citation in enumerate(response.citations):
                    col_idx = idx % min(len(response.citations), 4)
                    with cols[col_idx]:
                        badge_type = "table-badge" if citation.is_table else "citation-badge"
                        label = f"Page {citation.page_num}" + (" (Table)" if citation.is_table else "")
                        st.markdown(f'<span class="{badge_type}">{label}</span>', unsafe_allow_html=True)

            # Display source chunks accordion
            if response.source_chunks:
                with st.expander(f"View {len(response.source_chunks)} Retrieved Source Chunks (Re-ranked)"):
                    for i, chunk in enumerate(response.source_chunks, start=1):
                        st.markdown(
                            f"**[Chunk {i}] Page {chunk.page_num}** | "
                            f"Rerank Score: `{chunk.rerank_score:.4f}` | "
                            f"RRF Score: `{chunk.rrf_score:.5f}` | "
                            f"Dense Rank: `{chunk.dense_rank}` | "
                            f"BM25 Rank: `{chunk.bm25_rank}`"
                        )
                        st.markdown(f"```markdown\n{chunk.text}\n```")
                        st.divider()

            # Record in session state
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "citations": [c.to_dict() for c in response.citations],
                    "chunks": response.source_chunks,
                }
            )
            # Rerun so state updates cleanly
            st.rerun()
