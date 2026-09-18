"""Generation module with strict grounding prompts and multi-provider (Gemini / Groq) support."""

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.config import AppSettings, get_settings
from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

STRICT_GROUNDING_PROMPT = """You are DocMind, an enterprise AI Document Intelligence assistant.
Your task is to provide an accurate, strictly grounded, and comprehensive answer to the user's question based EXCLUSIVELY on the retrieved document excerpts below.

=== GROUNDING & RELEVANCE RULES ===
1. EXCLUSIVE SOURCE MATERIAL: Base your answer STRICTLY and EXCLUSIVELY on the provided Context excerpts. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond what is directly stated in the text and tables.
2. STRICT RELEVANCE: Address ONLY what is directly relevant to the user's question. Focus on extracting all relevant figures, metrics, percentages, and financial items.
3. TABLE & METRIC EXTRACTION:
   - Carefully inspect all Markdown table rows, columns, and metric labels.
   - If the user asks for financial metrics, margins, ratios, or tables, you MUST extract ALL metrics, margins, and operational figures present across the excerpts (such as EBITDA, Operating Profit, PAT, Gross Profit, Margins, etc.) and synthesize them into a clean, well-formatted GitHub-flavored Markdown table.
   - NEVER refuse to answer if ANY metric, number, or data point relating to the question is present in the excerpts.
4. EXACT PROVENANCE CITATIONS: For EVERY factual statement, figure, or table row, you MUST cite the exact page number in brackets, e.g., "[Page 212, Table 1]" or "[Page 33]".
5. PARTIAL COVERAGE: If some requested items are present but others are not explicitly listed in the excerpts, provide all available metrics with citations and briefly note which specific item was not found.
6. REFUSAL: You may state "The provided document does not contain information regarding this topic." ONLY if the excerpts contain literally zero mentions, tables, or data related to the question.

=== RETRIEVED CONTEXT ===
{context_blocks}

=== USER QUESTION ===
{question}

=== GROUNDED ANSWER (WITH EXACT PAGE CITATIONS) ===
"""


@dataclass
class Citation:
    """Document citation extracted from response or source chunks."""

    page_num: int
    source_filename: str
    chunk_id: str
    snippet: str
    is_table: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGResponse:
    """Complete RAG response with generated text, provenance citations, and execution telemetry."""

    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    source_chunks: List[RetrievedChunk] = field(default_factory=list)
    provider: str = "unknown"
    model: str = "unknown"
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "source_chunks": [c.to_dict() for c in self.source_chunks],
            "provider": self.provider,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
        }


def format_context_blocks(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a standardized context block for LLM prompts."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        type_tag = "Table" if c.is_table else "Text"
        header = f"--- Excerpt {i} | Page {c.page_num} ({type_tag}) | Source: {c.source_filename} ---"
        blocks.append(f"{header}\n{c.text}\n")
    return "\n".join(blocks)


def extract_cited_pages(answer_text: str) -> List[int]:
    """Parse out page citations in format [Page X] or [Page X, ...] from generated text."""
    matches = re.findall(r"\[Page\s+(\d+)(?:[^\]]*)\]", answer_text, re.IGNORECASE)
    return sorted(list(set(int(m) for m in matches)))


class GroundedGenerator:
    """Generates strictly grounded answers with citations using Google Gemini or Groq."""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()

    def _generate_gemini(
        self, prompt: str, api_key: Optional[str] = None, model: Optional[str] = None
    ) -> str:
        """Execute inference using Google Gemini via google-genai SDK."""
        key = api_key or self.settings.gemini_api_key
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in the server environment (.env)."
            )

        from google import genai
        from google.genai import types

        model_to_use = model or self.settings.gemini_model
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model_to_use,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return response.text or ""

    def _generate_groq(
        self, prompt: str, api_key: Optional[str] = None, model: Optional[str] = None
    ) -> str:
        """Execute inference using Groq Cloud API."""
        key = api_key or self.settings.groq_api_key
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please configure GROQ_API_KEY in the server environment (.env)."
            )

        from groq import Groq

        model_to_use = model or self.settings.groq_model
        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return completion.choices[0].message.content or ""

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> RAGResponse:
        """
        Generate a strictly grounded answer with citations from retrieved chunks.
        """
        start_time = time.time()
        chosen_provider = (provider or self.settings.llm_provider).lower()
        model_name = model or (
            self.settings.gemini_model
            if chosen_provider == "gemini"
            else self.settings.groq_model
        )

        if not retrieved_chunks:
            return RAGResponse(
                query=query,
                answer="No relevant excerpts found in the document to answer your question.",
                citations=[],
                source_chunks=[],
                provider=chosen_provider,
                model=model_name,
                latency_seconds=time.time() - start_time,
            )

        context_str = format_context_blocks(retrieved_chunks)
        prompt = STRICT_GROUNDING_PROMPT.format(
            context_blocks=context_str,
            question=query,
        )

        answer_text = ""
        try:
            if chosen_provider == "gemini":
                answer_text = self._generate_gemini(
                    prompt, api_key=api_key, model=model_name
                )
            elif chosen_provider == "groq":
                answer_text = self._generate_groq(
                    prompt, api_key=api_key, model=model_name
                )
            else:
                raise ValueError(f"Unknown LLM provider: {chosen_provider}")
        except Exception as e:
            logger.error(f"Error during LLM inference ({chosen_provider}): {e}")
            answer_text = (
                f"**Inference Error**: Could not complete generation using {chosen_provider}.\n"
                f"Details: {str(e)}\n\n"
                f"*Please verify the backend API key configuration in .env.*"
            )

        latency = time.time() - start_time

        # Extract citations
        cited_pages = extract_cited_pages(answer_text)
        citations: List[Citation] = []

        is_refusal = (
            "cannot answer" in answer_text.lower()
            or "do not contain information" in answer_text.lower()
            or "no relevant excerpts" in answer_text.lower()
            or "does not contain information" in answer_text.lower()
        )

        # Map cited pages to corresponding chunks ONLY when pages are cited and it's not a refusal
        if cited_pages and not is_refusal:
            seen_badges = set()
            for chunk in retrieved_chunks:
                if chunk.page_num in cited_pages:
                    badge_key = (chunk.page_num, chunk.is_table)
                    if badge_key not in seen_badges:
                        seen_badges.add(badge_key)
                        snippet = chunk.text[:200].strip() + ("..." if len(chunk.text) > 200 else "")
                        citations.append(
                            Citation(
                                page_num=chunk.page_num,
                                source_filename=chunk.source_filename,
                                chunk_id=chunk.chunk_id,
                                snippet=snippet,
                                is_table=chunk.is_table,
                            )
                        )

        return RAGResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            source_chunks=retrieved_chunks,
            provider=chosen_provider,
            model=model_name,
            latency_seconds=latency,
        )

