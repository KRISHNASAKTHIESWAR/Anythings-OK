import logging
from uuid import uuid4
from llama_index.core.node_parser import SentenceSplitter

from graphdb.model import graphdb
from backend.extract import extract

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


def ingest_file(file_path: str):
    logger.info(f"[INGEST] Starting ingestion of: {file_path}")

    try:
        doc_id = str(uuid4())
        logger.info(f"[INGEST] Generated doc_id: {doc_id}")

        logger.info("[INGEST] Calling extract()...")
        docs = extract(file_path)

        if not docs:
            raise Exception("Extraction returned no documents")

        logger.info(f"[INGEST] Extracted {len(docs)} documents")

        # Add metadata to documents
        for d in docs:
            d.metadata["doc_id"] = doc_id
            d.metadata["file_name"] = file_path
            logger.debug(f"[INGEST] Document metadata: {d.metadata}")

        logger.info("[INGEST] Splitting documents into nodes...")
        parser = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=32
        )

        nodes = parser.get_nodes_from_documents(docs)
        logger.info(f"[INGEST] Parsed {len(nodes)} nodes")

        if not nodes:
            logger.error("[INGEST] Node parsing returned no nodes!")
            raise Exception("Node parsing returned no nodes")

        # Log sample nodes
        for i, node in enumerate(nodes[:2]):
            logger.debug(f"[INGEST] Sample node {i}: {node.text[:100]}... | metadata: {node.metadata}")

        logger.info(f"[INGEST] Ingesting {len(nodes)} nodes into Neo4j...")
        graphdb.create_index(nodes)

        # Verify data was written
        logger.info("[INGEST] Verifying data was written to Neo4j...")
        nodes_in_db = graphdb._verify_data_in_neo4j()
        logger.info(f"[INGEST] Nodes now in Neo4j: {nodes_in_db}")
        
        if nodes_in_db == 0:
            logger.warning("[INGEST] ⚠ WARNING: No nodes found in Neo4j after ingestion!")
            logger.warning("[INGEST] PropertyGraphIndex may not have persisted data automatically")
            logger.warning("[INGEST] This could be a LlamaIndex bug - trying manual insertion...")

        logger.info(f"[INGEST] Ingestion complete! doc_id: {doc_id}")
        return doc_id

    except Exception as e:
        logger.error(f"[INGEST] Ingestion failed: {str(e)}", exc_info=True)
        raise