import asyncio
import sys
import logging
import nest_asyncio

# Fix for Python 3.13 + Windows: patch event loop to never close mid-run
nest_asyncio.apply()

from llama_index.core import Settings, StorageContext
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

Settings.num_workers = 1


class GraphDB:

    def __init__(self):
        logger.info("[GraphDB] Initializing GraphDB connection...")

        try:
            # Neo4j connection
            self.graph_store = Neo4jPropertyGraphStore(
                username="neo4j",
                password="qwerty123456",
                url="neo4j://127.0.0.1:7687",
            )
            logger.info("[GraphDB] Neo4j connection established")


            # LLM for querying
            self.llm = Ollama(
                model="qwen3-small-ctx",
                base_url="http://localhost:11434",
                request_timeout=600,
                temperature=0,
                json_mode=False,
                context_window=2048,
                additional_kwargs={"options": {"num_ctx": 2048}}
            )
            logger.info("[GraphDB] Query LLM initialized")

            # Faster LLM for graph extraction
            self.extract_llm = Ollama(
                model="phi3-small-ctx",
                base_url="http://localhost:11434",
                request_timeout=600,
                temperature=0,
                json_mode=False,
                context_window=2048,
                additional_kwargs={"options": {"num_ctx": 2048}}
            )
            logger.info("[GraphDb] Extraction LLM initialized")

            self.kg_extractors = [
                SimpleLLMPathExtractor(
                    llm=self.extract_llm,
                    max_paths_per_chunk=3,
                    num_workers=1,
                )
            ]
            logger.info("[GraphDB] KG extractors configured")

            # Embedding model
            self.embed_model = OllamaEmbedding(
                model_name="qwen3-embedding:4b",
                base_url="http://localhost:11434",
                ollama_additional_kwargs={"options": {"num_ctx": 512}}
            )
            logger.info("[GraphDB] Embedding model initialized")

            Settings.llm = self.llm
            Settings.embed_model = self.embed_model

            self.index = None
            logger.info("[GraphDB] GraphDB initialization complete")
        except Exception as e:
            logger.error(f"[GraphDB] Initialization failed: {str(e)}", exc_info=True)
            raise

    def create_index(self, nodes):
        """Build index and persist to Neo4j"""
        logger.info(f"[GraphDB] Starting index creation with {len(nodes)} nodes")

        try:
            # Storage context must be created fresh each time for Neo4j writes
            storage_context = StorageContext.from_defaults(
                graph_store=self.graph_store
            )
            logger.debug("[GraphDB] Storage context created")

            logger.info("[GraphDB] Building PropertyGraphIndex (this extracts KG from nodes)...")
            self.index = PropertyGraphIndex(
                nodes=nodes,
                storage_context=storage_context,
                kg_extractors=self.kg_extractors,
                show_progress=True,
            )
            logger.info("[GraphDB] PropertyGraphIndex construction complete")
            
            # Try to force refresh schema to persist
            logger.info("[GraphDB] Attempting to refresh schema...")
            try:
                self.graph_store.refresh_schema()
                logger.info("[GraphDB] Schema refreshed")
            except Exception as e:
                logger.warning(f"[GraphDB] Schema refresh failed: {str(e)}")
            
            # Verify data was written to Neo4j
            nodes_in_db = self._verify_data_in_neo4j()
            
            # If no nodes were persisted, try manual insertion as fallback
            if nodes_in_db == 0:
                logger.warning("[GraphDB] ⚠ PropertyGraphIndex did not persist data!")
                logger.info("[GraphDB] Attempting manual node insertion as fallback...")
                self._manual_insert_nodes(nodes)
                
                # Verify again
                nodes_in_db = self._verify_data_in_neo4j()
                if nodes_in_db > 0:
                    logger.info("[GraphDB] ✓ Manual insertion successful!")
                else:
                    logger.error("[GraphDB] ✗ Manual insertion also failed - no nodes in database")

            return self.index
        except Exception as e:
            logger.error(f"[GraphDB] Index creation failed: {str(e)}", exc_info=True)
            raise

    def _verify_data_in_neo4j(self):
        """Direct query to verify what's actually in Neo4j after write"""
        logger.info("[GraphDB] Verifying data in Neo4j...")
        
        try:
            with self.graph_store._driver.session() as session:
                # Check total nodes
                result = session.run("MATCH (n) RETURN count(n) as total")
                total_nodes = result.single()["total"]
                logger.info(f"[GraphDB] Total nodes in Neo4j: {total_nodes}")

                # Check all labels and counts
                result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as cnt")
                node_types = [(r["labels"], r["cnt"]) for r in result]
                if node_types:
                    logger.info("[GraphDB] Node types:")
                    for labels, cnt in node_types:
                        logger.info(f"  {labels}: {cnt}")
                else:
                    logger.warning("[GraphDB] No node types found!")

                # Check all property keys that exist
                result = session.run("MATCH (n) UNWIND keys(n) as k RETURN DISTINCT k")
                props = [r["k"] for r in result]
                logger.info(f"[GraphDB] Properties in nodes: {props}")

                # Sample a node
                result = session.run("MATCH (n) RETURN n LIMIT 1")
                record = result.single()
                if record:
                    logger.debug(f"[GraphDB] Sample node: {dict(record['n'])}")
                else:
                    logger.warning("[GraphDB] No nodes found in database!")
                    
                return total_nodes
                    
        except Exception as e:
            logger.error(f"[GraphDB] Verification query failed: {str(e)}", exc_info=True)
            return 0

    def _manual_insert_nodes(self, nodes):
        """Fallback: manually insert nodes to Neo4j using raw driver if PropertyGraphIndex didn't persist them"""
        logger.info(f"[GraphDB] Attempting manual insertion of {len(nodes)} nodes using raw driver...")
        
        inserted_count = 0
        try:
            with self.graph_store._driver.session() as session:
                for i, node in enumerate(nodes):
                    try:
                        # Create __Node__ in Neo4j with node properties
                        node_id = node.node_id or f"node_{i}"
                        node_text = node.get_content()[:500] if hasattr(node, 'get_content') else str(node.text)[:500]
                        
                        # Use raw Cypher to create node
                        query = """
                        CREATE (n:__Node__ {
                            id: $node_id,
                            text: $node_text,
                            created_at: datetime()
                        })
                        """
                        session.run(query, node_id=node_id, node_text=node_text)
                        inserted_count += 1
                        
                        if (i + 1) % 5 == 0 or (i + 1) == len(nodes):
                            logger.debug(f"[GraphDB] Raw insert: {i + 1}/{len(nodes)} nodes created in transaction")
                    except Exception as e:
                        logger.warning(f"[GraphDB] Failed to insert node {i} ({node.node_id}): {str(e)}")
            
            logger.info(f"[GraphDB] Manual insertion completed: {inserted_count}/{len(nodes)} nodes created in Neo4j")
            
        except Exception as e:
            logger.error(f"[GraphDB] Manual insertion failed: {str(e)}", exc_info=True)

    def load_index(self):
        """Load existing graph from Neo4j"""
        logger.info("[GraphDB] Loading index from existing Neo4j graph...")
        
        try:
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                llm=self.llm,
                embed_model=self.embed_model,
            )
            logger.info("[GraphDB] Index loaded successfully")
            return self.index
        except Exception as e:
            logger.error(f"[GraphDB] Load index failed: {str(e)}", exc_info=True)
            raise

    def get_retriever(self):
        logger.debug("[GraphDB] Getting retriever...")

        if self.index is None:
            self.load_index()

        return self.index.as_retriever(include_text=True)

    def clear_graph(self):
        logger.warning("[GraphDB] Clearing entire graph from Neo4j...")
        
        try:
            with self.graph_store._driver.session() as session:
                result = session.run("MATCH (n) DETACH DELETE n")
                logger.info("[GraphDB] Graph cleared")
        except Exception as e:
            logger.error(f"[GraphDB] Clear graph failed: {str(e)}", exc_info=True)
            raise

    def delete_document(self, doc_id: str):
        logger.info(f"[GraphDB] Deleting document: {doc_id}")
        
        try:
            with self.graph_store._driver.session() as session:
                result = session.run(
                    "MATCH (n) WHERE n.doc_id = $doc_id DETACH DELETE n",
                    doc_id=doc_id
                )
                logger.info(f"[GraphDB] Document {doc_id} deleted")
        except Exception as e:
            logger.error(f"[GraphDB] Delete document failed: {str(e)}", exc_info=True)
            raise

    def list_documents(self):
        logger.debug("[GraphDB] Listing documents...")
        
        try:
            # Use filename property since that's what Neo4j actually stores
            query = """
            MATCH (n)
            WHERE n.filename IS NOT NULL
            RETURN DISTINCT n.id AS doc_id, n.filename AS file
            LIMIT 100
            """

            with self.graph_store._driver.session() as session:
                result = session.run(query)
                docs = [{"doc_id": r["doc_id"], "file": r["file"]} for r in result]
                logger.debug(f"[GraphDB] Found {len(docs)} documents")
                return docs
        except Exception as e:
            logger.error(f"[GraphDB] List documents failed: {str(e)}", exc_info=True)
            return []


# Singleton instance
graphdb = GraphDB()