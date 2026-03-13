"""
graphdb/community.py
Community detection using Leiden algorithm on the entity graph.
This is the key GraphRAG innovation — hierarchical communities
with LLM-generated summaries at each level.
"""

import logging
from typing import List, Dict

import networkx as nx
import requests

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


def build_networkx_graph(entities: List[Dict], relationships: List[Dict]) -> nx.Graph:
    """Convert entity/relationship lists to a NetworkX graph."""
    G = nx.Graph()

    for ent in entities:
        G.add_node(ent["name"], type=ent.get("type", "OTHER"), description=ent.get("description", ""))

    for rel in relationships:
        G.add_edge(
            rel["source"], rel["target"],
            relation=rel.get("relation", "RELATED_TO"),
            description=rel.get("description", ""),
        )

    logger.info(f"[COMMUNITY] NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> Dict[int, List[str]]:
    """
    Run community detection.
    Uses Leiden if available (pip install leidenalg igraph), 
    falls back to Louvain (NetworkX built-in).
    
    Returns: {community_id: [entity_names]}
    """
    if G.number_of_nodes() == 0:
        logger.warning("[COMMUNITY] Empty graph, no communities")
        return {}

    try:
        import igraph as ig
        import leidenalg

        # Convert NetworkX → igraph
        ig_graph = ig.Graph.from_networkx(G)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )

        communities = {}
        nx_nodes = list(G.nodes())
        for comm_id, members in enumerate(partition):
            member_names = [nx_nodes[idx] for idx in members]
            if len(member_names) >= 2:  # Skip singleton communities
                communities[comm_id] = member_names

        logger.info(f"[COMMUNITY] Leiden found {len(communities)} communities")
        return communities

    except ImportError:
        logger.warning("[COMMUNITY] leidenalg not found, falling back to Louvain")

        from networkx.algorithms.community import louvain_communities
        partitions = louvain_communities(G, resolution=resolution)

        communities = {}
        for comm_id, members in enumerate(partitions):
            member_names = list(members)
            if len(member_names) >= 2:
                communities[comm_id] = member_names

        logger.info(f"[COMMUNITY] Louvain found {len(communities)} communities")
        return communities


def _build_community_context(
    community_members: List[str],
    G: nx.Graph,
) -> str:
    """
    Build a text description of a community for the LLM to summarize.
    Includes member entities and their relationships.
    """
    lines = []
    lines.append("Entities in this community:")
    for name in community_members:
        data = G.nodes.get(name, {})
        etype = data.get("type", "")
        desc = data.get("description", "")
        lines.append(f"- {name} ({etype}): {desc}")

    lines.append("\nRelationships:")
    for name in community_members:
        for neighbor in G.neighbors(name):
            if neighbor in community_members:
                edge = G.edges[name, neighbor]
                rel = edge.get("relation", "RELATED_TO")
                desc = edge.get("description", "")
                lines.append(f"- {name} --[{rel}]--> {neighbor}: {desc}")

    return "\n".join(lines)


SUMMARY_PROMPT = """You are summarizing a group of related entities and their relationships from a knowledge graph.

Given the entities and relationships below, write a concise summary (2-4 sentences) that captures:
1. What this group is about
2. The key relationships between members
3. Why these entities are connected

Entities and relationships:
{context}

Write a concise summary:"""


def summarize_communities(
    communities: Dict[int, List[str]],
    G: nx.Graph,
    model: str = "qwen3-small-ctx",
) -> List[Dict]:
    """
    Generate LLM summaries for each community.
    Returns list of community dicts ready for Neo4j storage.
    """
    results = []

    for comm_id, members in communities.items():
        logger.info(f"[COMMUNITY] Summarizing community {comm_id} ({len(members)} members)")

        context = _build_community_context(members, G)

        # Truncate context if too long for the model
        if len(context) > 3000:
            context = context[:3000] + "\n... (truncated)"

        try:
            prompt = SUMMARY_PROMPT.format(context=context)
            summary = _call_ollama(model, prompt)
            # Clean up — small models sometimes add preamble
            summary = summary.strip()
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1]
        except Exception as e:
            logger.warning(f"[COMMUNITY] Summary generation failed for {comm_id}: {e}")
            summary = f"Community of {len(members)} entities: {', '.join(members[:5])}"

        results.append({
            "community_id": f"comm_{comm_id}",
            "level": 0,
            "members": members,
            "member_count": len(members),
            "summary": summary[:500],
        })

    logger.info(f"[COMMUNITY] Generated {len(results)} community summaries")
    return results


def run_community_detection(
    entities: List[Dict],
    relationships: List[Dict],
    summary_model: str = "qwen3-small-ctx",
    resolution: float = 1.0,
) -> List[Dict]:
    """
    Full pipeline: build graph → detect communities → summarize.
    Returns community dicts for storage.
    """
    G = build_networkx_graph(entities, relationships)

    if G.number_of_nodes() < 3:
        logger.info("[COMMUNITY] Too few entities for community detection, skipping")
        return []

    communities = detect_communities(G, resolution)

    if not communities:
        logger.info("[COMMUNITY] No communities found")
        return []

    return summarize_communities(communities, G, summary_model)