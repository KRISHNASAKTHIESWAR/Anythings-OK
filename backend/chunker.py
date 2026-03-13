"""
backend/chunker.py
Semantic-aware text chunking.
Splits on paragraph/sentence boundaries, not mid-sentence.
"""

import re
import logging
from typing import List, Dict
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── Sentence splitter ───────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex-based)."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _approx_tokens(text: str) -> int:
    """Rough token count — words * 1.3 is close enough for English."""
    return int(len(text.split()) * 1.3)


# ── Chunker ─────────────────────────────────────────────────────────

def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:
    """
    Split extracted documents into overlapping chunks.
    
    Strategy:
    1. Split into paragraphs first (preserves structure)
    2. Within paragraphs, accumulate sentences up to chunk_size
    3. Overlap by pulling back overlap tokens worth of sentences
    
    Returns list of dicts with keys: chunk_id, text, source, type, doc_id
    """
    all_chunks = []

    for doc in docs:
        doc_id = str(uuid4())
        raw_text = doc["text"]
        source = doc.get("source", "unknown")
        doc_type = doc.get("type", "unknown")

        if not raw_text.strip():
            logger.warning(f"[CHUNK] Empty document from {source}, skipping")
            continue

        # Split into paragraphs
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text]

        # Collect all sentences across paragraphs, keeping paragraph breaks
        sentences = []
        for para in paragraphs:
            para_sents = _split_sentences(para)
            if not para_sents:
                # Paragraph didn't split well — treat as one sentence
                para_sents = [para]
            sentences.extend(para_sents)

        # Build chunks by accumulating sentences
        chunks = []
        current_sents = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _approx_tokens(sent)

            if current_tokens + sent_tokens > chunk_size and current_sents:
                # Flush current chunk
                chunk_text = " ".join(current_sents)
                chunks.append(chunk_text)

                # Overlap: keep last N tokens worth of sentences
                overlap_sents = []
                overlap_tokens = 0
                for s in reversed(current_sents):
                    st = _approx_tokens(s)
                    if overlap_tokens + st > chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += st

                current_sents = overlap_sents
                current_tokens = overlap_tokens

            current_sents.append(sent)
            current_tokens += sent_tokens

        # Last chunk
        if current_sents:
            chunks.append(" ".join(current_sents))

        # Package
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "text": chunk_text,
                "source": source,
                "type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        logger.info(
            f"[CHUNK] {source}: {len(sentences)} sentences → {len(chunks)} chunks "
            f"(target {chunk_size} tokens, {chunk_overlap} overlap)"
        )

    logger.info(f"[CHUNK] Total: {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks