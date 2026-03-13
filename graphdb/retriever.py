"""
graphdb/retriever.py
GraphRAG-style retrieval: local search (entity neighborhood) + 
global search (community summaries).
"""

import logging
from typing import List, Dict, Optional

import requests

from graphdb.model import graphdb

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"


def _call_ollama(model: str, prompt: str, timeout: int = 300) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 2048},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_ollama_stream(model: str, prompt: str, timeout: int = 300):
    """Streaming version for chat responses."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.1, "num_ctx": 2048},
        },
        timeout=timeout,
        stream=True,
    )
    resp.raise_for_status()
    import json
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if "message" in data and "content" in data["message"]:
                yield data["message"]["content"]
            if data.get("done", False):
                break


# ── Entity extraction from query ────────────────────────────────────

ENTITY_EXTRACT_PROMPT = """Extract the key entities from this question. Return ONLY a comma-separated list of entity names, nothing else.

Question: {query}

Entities:"""


def extract_query_entities(query: str, model: str = "qwen3-small-ctx") -> List[str]:
    """Extract entity names from a user query."""
    prompt = ENTITY_EXTRACT_PROMPT.format(query=query)
    try:
        raw = _call_ollama(model, prompt)
        # Parse comma-separated list
        entities = [e.strip().lower() for e in raw.split(",") if e.strip()]
        # Also try keyword matching against the graph
        words = query.lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                matches = graphdb.search_entities(word, limit=3)
                for m in matches:
                    if m["name"] not in entities:
                        entities.append(m["name"])

        logger.debug(f"[RETRIEVE] Query entities: {entities}")
        return entities[:10]  # Cap at 10
    except Exception as e:
        logger.warning(f"[RETRIEVE] Entity extraction failed: {e}")
        # Fallback: just use query words
        return [w.lower() for w in query.split() if len(w) > 3][:5]


# ── Context building ────────────────────────────────────────────────

def build_local_context(entities: List[str], hops: int = 2) -> str:
    """
    Local search: gather neighborhoods for matched entities.
    Returns formatted context string.
    """
    context_parts = []

    for entity_name in entities:
        # Try exact match first, then fuzzy
        neighborhood = graphdb.get_entity_neighborhood(entity_name, hops=hops)

        if not neighborhood["neighbors"] and not neighborhood["chunks"]:
            # Try fuzzy search
            matches = graphdb.search_entities(entity_name, limit=1)
            if matches:
                neighborhood = graphdb.get_entity_neighborhood(matches[0]["name"], hops=hops)

        if neighborhood["neighbors"] or neighborhood["chunks"]:
            parts = [f"\n== Entity: {neighborhood['entity']} =="]

            if neighborhood["relationships"]:
                parts.append("Relationships:")
                for rel in neighborhood["relationships"][:10]:
                    parts.append(
                        f"  {rel['source']} --[{rel['relation']}]--> {rel['target']}"
                        f"{': ' + rel.get('description', '') if rel.get('description') else ''}"
                    )

            if neighborhood["chunks"]:
                parts.append("Source text:")
                for chunk in neighborhood["chunks"][:3]:
                    # Truncate chunks to save context window
                    text = chunk["text"][:300]
                    parts.append(f"  [{chunk['source']}] {text}")

            context_parts.append("\n".join(parts))

    return "\n\n".join(context_parts) if context_parts else ""


def build_global_context(entities: Optional[List[str]] = None) -> str:
    """
    Global search: gather community summaries.
    If entities provided, get their communities. Otherwise get all.
    """
    summaries = []

    if entities:
        for ent in entities:
            comms = graphdb.get_community_summaries(entity_name=ent)
            for c in comms:
                if c["summary"] not in [s["summary"] for s in summaries]:
                    summaries.append(c)
    
    if not summaries:
        # Fallback to all communities
        summaries = graphdb.get_community_summaries()

    if not summaries:
        return ""

    parts = ["\n== Knowledge Graph Communities =="]
    for s in summaries[:5]:  # Cap at 5 summaries
        parts.append(f"[Community {s['id']}, {s['member_count']} entities]: {s['summary']}")

    return "\n".join(parts)


# ── Main retrieval + answer ─────────────────────────────────────────

ANSWER_PROMPT = """You are a helpful assistant answering questions using knowledge from a graph database.

Use the following context to answer the question. If the context doesn't contain enough information, say so honestly.

CONTEXT:
{context}

QUESTION: {query}

Answer based on the context above:"""


def retrieve_and_answer(
    query: str,
    model: str = "qwen3-small-ctx",
    stream: bool = True,
    hops: int = 2,
):
    """
    Full GraphRAG retrieval pipeline:
    1. Extract entities from query
    2. Build local context (entity neighborhoods + source chunks)
    3. Build global context (community summaries)
    4. Combine and send to LLM
    """
    logger.info(f"[RETRIEVE] Query: {query}")

    # Step 1: Extract entities
    entities = extract_query_entities(query, model)
    logger.info(f"[RETRIEVE] Extracted entities: {entities}")

    # Step 2: Local context
    local_ctx = build_local_context(entities, hops=hops)
    logger.info(f"[RETRIEVE] Local context: {len(local_ctx)} chars")

    # Step 3: Global context
    global_ctx = build_global_context(entities)
    logger.info(f"[RETRIEVE] Global context: {len(global_ctx)} chars")

    # Step 4: Combine
    full_context = ""
    if local_ctx:
        full_context += "--- Local Knowledge ---\n" + local_ctx + "\n\n"
    if global_ctx:
        full_context += "--- Global Knowledge ---\n" + global_ctx

    if not full_context.strip():
        full_context = "(No relevant information found in the knowledge graph)"
        logger.warning("[RETRIEVE] No context found for query")

    # Truncate if too long for model context window
    if len(full_context) > 4000:
        full_context = full_context[:4000] + "\n... (truncated)"

    prompt = ANSWER_PROMPT.format(context=full_context, query=query)
    logger.debug(f"[RETRIEVE] Final prompt: {len(prompt)} chars")

    # Step 5: Answer
    if stream:
        return _call_ollama_stream(model, prompt)
    else:
        return _call_ollama(model, prompt)