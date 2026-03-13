"""
cli/load.py
Load a file into the GraphRAG system.
Usage: python cli/load.py <filename>
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 50)
    print("  GraphRAG Document Loader")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\nUsage: python cli/load.py <filename>")
        print("Supported: .pdf .txt .md .docx .csv .xlsx .png .jpg .wav .mp3")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"\nError: File not found: {file_path}")
        return

    print(f"\nLoading: {file_path}")
    print(f"Size: {os.path.getsize(file_path) / 1024:.1f} KB")
    print()

    try:
        from graphdb.ingest import ingest_file

        doc_id = ingest_file(file_path)

        print("\n✓ Ingestion complete!")
        print(f"  Document ID: {doc_id}")

        from graphdb.model import graphdb
        stats = graphdb.stats()
        print(f"  Graph now has: {stats['entities']} entities, "
              f"{stats['relationships']} relationships, "
              f"{stats['communities']} communities")

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")

    print("=" * 50)


if __name__ == "__main__":
    main()