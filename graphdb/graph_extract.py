"""
graphdb/graph_extract.py
Entity and relationship extraction using direct Ollama calls.
Structured prompts designed for small models (phi3, qwen3 etc).
"""

import json
import re
import logging
from typing import List, Dict, Tuple

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

# ── Extraction prompt ───────────────────────────────────────────────
# Key design choices for small models:
# - Very explicit JSON schema
# - Few-shot example in prompt
# - Limited entity types to reduce confusion
# - Short, direct instructions

EXTRACT_PROMPT = """You are an entity and relationship extractor. Given text, extract entities and relationships.

RULES:
- Extract ONLY clearly stated entities and relationships
- Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, DOCUMENT, OTHER
- Each relationship must connect two extracted entities
- Output ONLY valid JSON, no other text

OUTPUT FORMAT (strict JSON):
{{
  "entities": [
    {{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "one line description"}}
  ],
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "relation": "RELATION_TYPE", "description": "one line"}}
  ]
}}

EXAMPLE:
Text: "Microsoft acquired GitHub in 2018. GitHub is a code hosting platform used by developers."
Output:
{{
  "entities": [
    {{"name": "Microsoft", "type": "ORGANIZATION", "description": "Technology company"}},
    {{"name": "GitHub", "type": "ORGANIZATION", "description": "Code hosting platform"}}
  ],
  "relationships": [
    {{"source": "Microsoft", "target": "GitHub", "relation": "ACQUIRED", "description": "Microsoft acquired GitHub in 2018"}}
  ]
}}

Now extract from this text:
---
{text}
---

Output JSON only:"""


def _call_ollama(model: str, prompt: str, timeout: int = 300) -> str:
    """Raw Ollama chat call."""
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


def _parse_json_response(raw: str) -> Dict:
    """
    Robustly parse JSON from LLM output.
    Small models often wrap JSON in markdown or add preamble.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"[EXTRACT] Failed to parse JSON from LLM response: {raw[:200]}")
    return {"entities": [], "relationships": []}


def _normalize_entity_name(name: str) -> str:
    """Normalize entity names for consistent graph nodes."""
    return name.strip().lower().replace("  ", " ")


def extract_graph_from_chunk(
    text: str,
    model: str = "phi3-small-ctx",
    max_retries: int = 2,
) -> Dict:
    """
    Extract entities and relationships from a single chunk.
    Returns normalized, deduplicated results.
    Retries on JSON parse failure with a shorter input.
    """
    # Limit input to ~800 chars to leave room for prompt + output in 2048 ctx
    chunk_text = text[:800]

    for attempt in range(max_retries + 1):
        prompt = EXTRACT_PROMPT.format(text=chunk_text)

        try:
            raw = _call_ollama(model, prompt)
            data = _parse_json_response(raw)

            # If we got empty results and this isn't the last attempt, retry
            if not data.get("entities") and attempt < max_retries:
                logger.warning(f"[EXTRACT] Attempt {attempt+1}: empty extraction, retrying...")
                # Shorten input further on retry
                chunk_text = text[:600]
                continue
            break

        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"[EXTRACT] Attempt {attempt+1} failed: {e}, retrying...")
                chunk_text = text[:600]
                continue
            logger.error(f"[EXTRACT] All attempts failed: {e}")
            return {"entities": [], "relationships": []}

    # Normalize entities
    entities = []
    seen_entities = set()
    for ent in data.get("entities", []):
        name = _normalize_entity_name(ent.get("name", ""))
        if not name or len(name) < 2:
            continue
        if name in seen_entities:
            continue
        seen_entities.add(name)
        entities.append({
            "name": name,
            "type": ent.get("type", "OTHER").upper(),
            "description": ent.get("description", "")[:200],
        })

    # Normalize relationships — only keep those connecting known entities
    relationships = []
    for rel in data.get("relationships", []):
        src = _normalize_entity_name(rel.get("source", ""))
        tgt = _normalize_entity_name(rel.get("target", ""))
        if src not in seen_entities or tgt not in seen_entities:
            continue
        if src == tgt:
            continue
        relationships.append({
            "source": src,
            "target": tgt,
            "relation": rel.get("relation", "RELATED_TO").upper().replace(" ", "_"),
            "description": rel.get("description", "")[:200],
        })

    logger.debug(
        f"[EXTRACT] Chunk → {len(entities)} entities, {len(relationships)} relationships"
    )
    return {"entities": entities, "relationships": relationships}


def extract_graph_from_chunks(
    chunks: List[Dict],
    model: str = "phi3-small-ctx",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract entities and relationships from all chunks.
    Merges and deduplicates across chunks.
    
    Returns (all_entities, all_relationships)
    """
    entity_map = {}  # name -> entity dict (merge descriptions)
    rel_set = set()  # (src, tgt, relation) for dedup
    all_rels = []

    for i, chunk in enumerate(chunks):
        logger.info(f"[EXTRACT] Processing chunk {i+1}/{len(chunks)}")

        result = extract_graph_from_chunk(chunk["text"], model)

        # Merge entities
        for ent in result["entities"]:
            name = ent["name"]
            if name in entity_map:
                # Keep longer description
                existing = entity_map[name]
                if len(ent["description"]) > len(existing["description"]):
                    existing["description"] = ent["description"]
                # Accumulate source chunks
                existing["chunk_ids"].append(chunk["chunk_id"])
            else:
                entity_map[name] = {
                    **ent,
                    "chunk_ids": [chunk["chunk_id"]],
                }

        # Merge relationships
        for rel in result["relationships"]:
            key = (rel["source"], rel["target"], rel["relation"])
            if key not in rel_set:
                rel_set.add(key)
                all_rels.append({
                    **rel,
                    "chunk_id": chunk["chunk_id"],
                })

    all_entities = list(entity_map.values())
    logger.info(
        f"[EXTRACT] Total: {len(all_entities)} unique entities, "
        f"{len(all_rels)} unique relationships"
    )
    return all_entities, all_rels