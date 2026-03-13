"""
graphdb/ingest.py
Full ingestion pipeline:
  extract → chunk → graph extract → store → community detection → store summaries
"""

import logging

from backend.extract import extract
from backend.chunker import chunk_documents
from graphdb.graph_extract import extract_graph_from_chunks
from graphdb.community import run_community_detection
from graphdb.model import graphdb

logger = logging.getLogger(__name__)


def ingest_file(
    file_path: str,
    extract_model: str = "phi3-small-ctx",
    summary_model: str = "qwen3-small-ctx",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> str:
    """
    Full ingestion pipeline for a single file.
    
    Returns: doc_id of the first document chunk
    """
    logger.info("=" * 60)
    logger.info(f"[INGEST] Starting: {file_path}")
    logger.info("=" * 60)

    # 1. Extract raw text
    logger.info("[INGEST] Step 1: Extracting text...")
    docs = extract(file_path)
    if not docs:
        raise ValueError("Extraction returned nothing")
    logger.info(f"[INGEST] Extracted {len(docs)} document(s)")

    # 2. Chunk
    logger.info("[INGEST] Step 2: Chunking...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Chunking returned nothing")
    doc_id = chunks[0]["doc_id"]
    logger.info(f"[INGEST] {len(chunks)} chunks, doc_id: {doc_id}")

    # 3. Store chunks in Neo4j
    logger.info("[INGEST] Step 3: Storing chunks...")
    graphdb.store_chunks(chunks)

    # 4. Extract entities and relationships
    logger.info("[INGEST] Step 4: Extracting graph (this may take a while)...")
    entities, relationships = extract_graph_from_chunks(chunks, model=extract_model)
    logger.info(f"[INGEST] Found {len(entities)} entities, {len(relationships)} relationships")

    if not entities:
        logger.warning("[INGEST] No entities extracted — graph will be empty")
        return doc_id

    # 5. Store entities and relationships
    logger.info("[INGEST] Step 5: Storing graph in Neo4j...")
    graphdb.store_entities(entities)
    graphdb.store_relationships(relationships)

    # 6. Community detection and summarization
    logger.info("[INGEST] Step 6: Community detection...")
    communities = run_community_detection(
        entities, relationships, summary_model=summary_model
    )
    if communities:
        graphdb.store_communities(communities)
        logger.info(f"[INGEST] {len(communities)} communities stored")
    else:
        logger.info("[INGEST] No communities detected (graph may be too sparse)")

    # 7. Stats
    stats = graphdb.stats()
    logger.info(f"[INGEST] Final graph stats: {stats}")
    logger.info(f"[INGEST] Done! doc_id: {doc_id}")
    logger.info("=" * 60)

    return doc_id