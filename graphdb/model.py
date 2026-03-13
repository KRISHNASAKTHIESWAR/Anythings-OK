"""
graphdb/model.py
Neo4j graph database manager.
Stores entities, relationships, chunks, and community summaries.
"""

import logging
from typing import List, Dict, Optional

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from pathlib import Path


logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class GraphDB:

    def __init__(
        self,
        uri: str = os.getenv("NEO4J_URI"),
        user: str = os.getenv("NEO4J_USER"),
        password: str = os.getenv("NEO4J_PASSWORD"),
    ):
        logger.info("[GraphDB] Connecting to Neo4j...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self._ensure_indexes()
        logger.info("[GraphDB] Connected and indexes ensured")

    def _ensure_indexes(self):
        """Create indexes for fast lookups."""
        with self.driver.session() as s:
            s.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            s.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)")
            s.run("CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.community_id)")
            s.run("CREATE INDEX doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)")

    # ── Write operations ────────────────────────────────────────────

    def store_chunks(self, chunks: List[Dict]):
        """Store text chunks as Chunk nodes."""
        logger.info(f"[GraphDB] Storing {len(chunks)} chunks...")
        query = """
        UNWIND $chunks AS c
        MERGE (chunk:Chunk {chunk_id: c.chunk_id})
        SET chunk.text = c.text,
            chunk.source = c.source,
            chunk.type = c.type,
            chunk.doc_id = c.doc_id,
            chunk.chunk_index = c.chunk_index
        """
        with self.driver.session() as s:
            s.run(query, chunks=chunks)
        logger.info(f"[GraphDB] Chunks stored")

    def store_entities(self, entities: List[Dict]):
        """Store entities as Entity nodes."""
        logger.info(f"[GraphDB] Storing {len(entities)} entities...")
        query = """
        UNWIND $entities AS e
        MERGE (ent:Entity {name: e.name})
        SET ent.type = e.type,
            ent.description = e.description
        """
        with self.driver.session() as s:
            s.run(query, entities=entities)

        # Link entities to their source chunks
        link_query = """
        UNWIND $entities AS e
        UNWIND e.chunk_ids AS cid
        MATCH (ent:Entity {name: e.name})
        MATCH (chunk:Chunk {chunk_id: cid})
        MERGE (ent)-[:MENTIONED_IN]->(chunk)
        """
        with self.driver.session() as s:
            s.run(link_query, entities=entities)

        logger.info(f"[GraphDB] Entities stored and linked to chunks")

    def store_relationships(self, relationships: List[Dict]):
        """Store relationships as edges between entities."""
        logger.info(f"[GraphDB] Storing {len(relationships)} relationships...")
        # Neo4j doesn't allow parameterized relationship types in MERGE,
        # so we use APOC or a generic edge with a 'type' property
        query = """
        UNWIND $rels AS r
        MATCH (src:Entity {name: r.source})
        MATCH (tgt:Entity {name: r.target})
        MERGE (src)-[rel:RELATES_TO {relation: r.relation}]->(tgt)
        SET rel.description = r.description,
            rel.chunk_id = r.chunk_id
        """
        with self.driver.session() as s:
            s.run(query, rels=relationships)
        logger.info(f"[GraphDB] Relationships stored")

    def store_communities(self, communities: List[Dict]):
        """
        Store community detection results.
        Each community has: community_id, level, member_entities, summary
        """
        logger.info(f"[GraphDB] Storing {len(communities)} communities...")
        query = """
        UNWIND $comms AS c
        MERGE (comm:Community {community_id: c.community_id})
        SET comm.level = c.level,
            comm.summary = c.summary,
            comm.member_count = c.member_count
        """
        with self.driver.session() as s:
            s.run(query, comms=communities)

        # Link entities to communities
        link_query = """
        UNWIND $comms AS c
        UNWIND c.members AS member_name
        MATCH (comm:Community {community_id: c.community_id})
        MATCH (ent:Entity {name: member_name})
        MERGE (ent)-[:BELONGS_TO]->(comm)
        """
        with self.driver.session() as s:
            s.run(link_query, comms=communities)

        logger.info(f"[GraphDB] Communities stored and linked")

    # ── Read operations ─────────────────────────────────────────────

    def get_entity_neighborhood(self, entity_name: str, hops: int = 2) -> Dict:
        """
        Get an entity and its local neighborhood (entities + relationships).
        This is the 'local search' in GraphRAG.
        """
        # Get neighbors and relationships using simple 1-hop matches
        # then union for multi-hop — avoids variable-length path issues
        rel_query = """
        MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(neighbor:Entity)
        RETURN neighbor.name AS neighbor_name, neighbor.type AS neighbor_type,
               neighbor.description AS neighbor_desc,
               startNode(r).name AS source, endNode(r).name AS target,
               r.relation AS relation, r.description AS rel_desc
        """
        neighbors = []
        rels = []
        seen_neighbors = set()
        seen_rels = set()

        with self.driver.session() as s:
            # First hop
            result = s.run(rel_query, name=entity_name)
            for r in result:
                n_name = r["neighbor_name"]
                if n_name not in seen_neighbors:
                    seen_neighbors.add(n_name)
                    neighbors.append({
                        "name": n_name, "type": r["neighbor_type"],
                        "description": r["neighbor_desc"],
                    })
                rel_key = (r["source"], r["target"], r["relation"])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    rels.append({
                        "source": r["source"], "target": r["target"],
                        "relation": r["relation"], "description": r["rel_desc"],
                    })

            # Second hop (if requested)
            if hops >= 2:
                for n_name in list(seen_neighbors):
                    result = s.run(rel_query, name=n_name)
                    for r in result:
                        nn = r["neighbor_name"]
                        if nn not in seen_neighbors and nn != entity_name:
                            seen_neighbors.add(nn)
                            neighbors.append({
                                "name": nn, "type": r["neighbor_type"],
                                "description": r["neighbor_desc"],
                            })
                        rel_key = (r["source"], r["target"], r["relation"])
                        if rel_key not in seen_rels:
                            seen_rels.add(rel_key)
                            rels.append({
                                "source": r["source"], "target": r["target"],
                                "relation": r["relation"], "description": r["rel_desc"],
                            })

        # Fetch associated chunks
        chunk_query = """
        MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(c:Chunk)
        RETURN c.text AS text, c.source AS source
        LIMIT 5
        """
        with self.driver.session() as s:
            result = s.run(chunk_query, name=entity_name)
            chunks = [{"text": r["text"], "source": r["source"]} for r in result]

        return {
            "entity": entity_name,
            "neighbors": neighbors,
            "relationships": rels,
            "chunks": chunks,
        }

    def get_community_summaries(self, entity_name: Optional[str] = None) -> List[Dict]:
        """
        Get community summaries — either all (global search) or
        those relevant to a specific entity (local+global).
        """
        if entity_name:
            query = """
            MATCH (e:Entity {name: $name})-[:BELONGS_TO]->(c:Community)
            RETURN c.community_id AS id, c.summary AS summary, c.level AS level,
                   c.member_count AS member_count
            ORDER BY c.level
            """
            with self.driver.session() as s:
                result = s.run(query, name=entity_name)
                return [dict(r) for r in result]
        else:
            query = """
            MATCH (c:Community)
            RETURN c.community_id AS id, c.summary AS summary, c.level AS level,
                   c.member_count AS member_count
            ORDER BY c.level, c.member_count DESC
            """
            with self.driver.session() as s:
                result = s.run(query)
                return [dict(r) for r in result]

    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Fuzzy search for entities by name."""
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($term)
        RETURN e.name AS name, e.type AS type, e.description AS description
        ORDER BY size(e.name)
        LIMIT $lim
        """
        with self.driver.session() as s:
            result = s.run(cypher, term=search_term, lim=limit)
            return [dict(r) for r in result]

    def get_all_entities_and_rels(self) -> Dict:
        """Export full graph as NetworkX-compatible edge list."""
        entity_query = "MATCH (e:Entity) RETURN e.name AS name, e.type AS type"
        rel_query = """
        MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
        RETURN a.name AS source, b.name AS target, r.relation AS relation
        """
        with self.driver.session() as s:
            entities = [dict(r) for r in s.run(entity_query)]
            rels = [dict(r) for r in s.run(rel_query)]
        return {"entities": entities, "relationships": rels}

    # ── Management ──────────────────────────────────────────────────

    def list_documents(self) -> List[Dict]:
        query = """
        MATCH (c:Chunk)
        WHERE c.doc_id IS NOT NULL
        RETURN DISTINCT c.doc_id AS doc_id, c.source AS source
        LIMIT 100
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(query)]

    def delete_document(self, doc_id: str):
        """Delete all chunks, orphaned entities, and communities for a document."""
        logger.info(f"[GraphDB] Deleting document: {doc_id}")
        with self.driver.session() as s:
            # Delete chunks
            s.run("MATCH (c:Chunk {doc_id: $doc_id}) DETACH DELETE c", doc_id=doc_id)
            # Clean orphaned entities (no remaining chunk links)
            s.run("""
                MATCH (e:Entity)
                WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk)
                DETACH DELETE e
            """)
            # Clean orphaned communities
            s.run("""
                MATCH (c:Community)
                WHERE NOT (:Entity)-[:BELONGS_TO]->(c)
                DELETE c
            """)
        logger.info(f"[GraphDB] Document {doc_id} and orphans cleaned")

    def clear_graph(self):
        logger.warning("[GraphDB] Clearing entire graph...")
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.info("[GraphDB] Graph cleared")

    def stats(self) -> Dict:
        """Quick stats about the graph."""
        with self.driver.session() as s:
            entities = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            rels = s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
            communities = s.run("MATCH (c:Community) RETURN count(c) AS c").single()["c"]
        return {
            "entities": entities,
            "chunks": chunks,
            "relationships": rels,
            "communities": communities,
        }

    def close(self):
        self.driver.close()


# Singleton
graphdb = GraphDB()