# Anythings-OK

A fully offline, multimodal GraphRAG system. Feed it documents, images, or audio — it builds a knowledge graph, detects communities, and lets you chat with your data. Everything runs locally on your machine.

## How it works

```
File (pdf/docx/png/mp3/...)
  │
  ▼
Extract text ──────────── LlamaIndex (documents), Ollama vision (images), Moonshine (audio)
  │
  ▼
Chunk ─────────────────── Sentence-aware splitting, 300 tokens, 50 overlap
  │
  ▼
Extract entities ──────── Local LLM via Ollama, structured JSON prompts
  │
  ▼
Store graph ───────────── Neo4j(Desktop)(Entity, Chunk, Relationship nodes)
  │
  ▼
Community detection ───── Leiden algorithm via NetworkX (no LLM)
  │
  ▼
Summarize communities ─── Local LLM generates 2-4 sentence summaries
  │
  ▼
Query ─────────────────── Local search (entity neighborhoods) + Global search (community summaries)
```

### What makes this different from regular RAG

Standard RAG splits documents into chunks, embeds them, and retrieves by vector similarity. This works for specific factual questions but fails on broad queries like "summarize everything about X" because no single chunk contains the full picture.

GraphRAG extracts entities and relationships into a knowledge graph, then runs community detection to find clusters of related entities. Each community gets an LLM-generated summary. At query time, the system combines:

- **Local context**: the entity's direct neighborhood in the graph + source text chunks
- **Global context**: community summaries that synthesize information across many chunks

This means both specific and broad questions get good answers.

## Prerequisites

- **Python 3.13+**
- **Neo4j** — [download](https://neo4j.com/download/) and run locally (default neo4j port 7687)
- **Ollama** — [download](https://ollama.com/) for local LLM inference
- **ffmpeg** — needed for non-wav audio files (`winget install ffmpeg` on Windows)

### Ollama models

Pull the models used by the system:

```bash
# For graph extraction (entity/relationship parsing)
ollama pull phi3

# For querying and community summarization
ollama pull qwen3

# For image description (ingesting images)
ollama pull qwen3-vl:4b
```

> **Note**: The code references `phi3-small-ctx` and `qwen3-small-ctx` — these are custom Modelfile variants with reduced context windows for speed. You can either create them (see below) or edit the model names in `graphdb/graph_extract.py` and `graphdb/model.py` to use the stock model names.

<details>
<summary>Creating custom Modelfiles (optional, for slow hardware)</summary>

```bash
# phi3-small-ctx
echo 'FROM phi3
PARAMETER num_ctx 2048' > Modelfile.phi3
ollama create phi3-small-ctx -f Modelfile.phi3

# qwen3-small-ctx
echo 'FROM qwen3
PARAMETER num_ctx 2048' > Modelfile.qwen3
ollama create qwen3-small-ctx -f Modelfile.qwen3
```

</details>

## Install

```bash
git clone https://github.com/adithyaa-s/Anythings-OK
cd Anythings-OK
uv sync
```

## Usage

### Load a document

```bash
python cli/load.py "path/to/file.pdf"
```

Supported formats:
- **Documents**: PDF, TXT, MD, DOCX, PPTX, EPUB, HTML, CSV, XLSX, JSON, IPYNB (via LlamaIndex)
- **Images**: PNG, JPG, JPEG, WEBP (described by vision model)
- **Audio**: WAV, MP3, M4A, FLAC (transcribed by Moonshine)

### Chat with your data

```bash
python cli/main.py
```

Commands inside the CLI:
- `chat` — ask questions about your documents
- `list` — show loaded documents
- `delete` — remove a document and its orphaned entities
- `stats` — graph statistics
- `clear` — wipe the entire graph
- `exit` — quit

### Example session

![alt text](image-1.png)

## Project structure

```
Anythings-OK/
├── backend/
│   ├── extract.py          # Multimodal file extraction (text, image, audio)
│   └── chunker.py          # Sentence-aware text chunking
├── cli/
│   ├── load.py             # Document ingestion CLI
│   └── main.py             # Interactive chat CLI
├── graphdb/
│   ├── model.py            # Neo4j graph database manager
│   ├── graph_extract.py    # LLM-based entity/relationship extraction
│   ├── community.py        # Leiden community detection + summarization
│   ├── ingest.py           # Full ingestion pipeline
│   └── retriever.py        # GraphRAG retrieval (local + global search)
├── pyproject.toml
└── README.md
```

## Architecture details

### Extraction

Images are described by a vision model (Qwen3-VL 4B) running locally via Ollama — it generates a textual description rather than doing OCR. Audio is transcribed by Moonshine, which runs on CPU with no CUDA dependencies. All other document types go through LlamaIndex's `SimpleDirectoryReader`.

### Chunking

Documents are split into ~300 token chunks with 50-token overlap. The chunker splits on sentence boundaries (not mid-word) by first breaking text into paragraphs, then accumulating sentences until the token budget is hit. Overlap pulls back the last ~64 tokens of sentences to preserve context across chunk boundaries.

### Graph extraction

Each chunk is sent to a local LLM with a structured prompt that forces JSON output containing entities (typed: PERSON, ORGANIZATION, TECHNOLOGY, etc.) and relationships. Entity names are normalized to lowercase and deduplicated across chunks. Relationships are validated — both endpoints must exist as extracted entities, and self-loops are filtered.

### Community detection

After extraction, the entity graph is exported from Neo4j into NetworkX (in-memory). The Leiden algorithm (or Louvain fallback) finds clusters of densely connected entities. Each community gets an LLM-generated summary stored back in Neo4j. This is the core GraphRAG innovation — it enables answering broad questions by providing pre-computed thematic summaries.

### Retrieval

Queries go through four steps:
1. **Entity extraction** — the LLM identifies key entities in the question, supplemented by fuzzy Neo4j search
2. **Local search** — traverse entity neighborhoods (1-2 hops) to gather relationships and source chunks
3. **Global search** — fetch community summaries for matched entities
4. **Answer generation** — combined local + global context is sent to the LLM with the original question

### Document isolation

All documents share a single Neo4j graph — this is intentional. If two documents mention the same entity (e.g., "TechCorp"), the entity node is reused and both documents' chunks link to it. This builds a richer knowledge base over time. Documents remain traceable via `doc_id` and `source` metadata on chunk nodes, and deleting a document cleans up its chunks and any orphaned entities.
