import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from graphdb.ingest import ingest_file

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("ANYTHINGS-OK DATA LOADER")
    logger.info("=" * 70)

    if len(sys.argv) < 2:
        print("Usage: python cli/load.py <filename>")
        logger.error("No filename provided")
        return

    file_path = sys.argv[1]

    logger.info(f"Loading document: {file_path}")

    try:
        doc_id = ingest_file(file_path)
        logger.info("✓ Success!")
        print("Done.")
        print("Document ID:", doc_id)
    except Exception as e:
        logger.error(f"✗ Failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()