"""
cli/main.py
Interactive GraphRAG CLI.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
logging.basicConfig(
    level=logging.WARNING,  # Keep chat quiet
    format='%(name)s - %(levelname)s - %(message)s'
)


def main():
    from graphdb.model import graphdb
    from graphdb.retriever import retrieve_and_answer

    print("\n" + "=" * 50)
    print("  GraphRAG Chat")
    print("=" * 50)

    stats = graphdb.stats()
    print(f"  Graph: {stats['entities']} entities, "
          f"{stats['relationships']} rels, "
          f"{stats['communities']} communities, "
          f"{stats['chunks']} chunks")

    print("\nCommands:")
    print("  chat   → ask questions about your documents")
    print("  list   → list loaded documents")
    print("  delete → delete a document")
    print("  stats  → graph statistics")
    print("  clear  → clear entire graph")
    print("  exit   → quit\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "exit":
            break

        elif cmd == "list":
            docs = graphdb.list_documents()
            if not docs:
                print("No documents loaded.\n")
                continue
            print(f"\nLoaded documents ({len(docs)}):")
            for d in docs:
                print(f"  {d['doc_id']}  |  {d['source']}")
            print()

        elif cmd == "delete":
            doc_id = input("  doc_id: ").strip()
            if doc_id:
                graphdb.delete_document(doc_id)
                print("  Deleted.\n")

        elif cmd == "stats":
            s = graphdb.stats()
            print(f"\n  Entities:      {s['entities']}")
            print(f"  Relationships: {s['relationships']}")
            print(f"  Communities:   {s['communities']}")
            print(f"  Chunks:        {s['chunks']}\n")

        elif cmd == "clear":
            confirm = input("  Are you sure? (yes/no): ").strip()
            if confirm == "yes":
                graphdb.clear_graph()
                print("  Graph cleared.\n")

        elif cmd == "chat":
            print("\nChat mode (type 'back' to return)\n")
            while True:
                try:
                    q = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() == "back" or not q:
                    break

                print("\nLLM: ", end="", flush=True)
                try:
                    for token in retrieve_and_answer(q, stream=True):
                        print(token, end="", flush=True)
                except Exception as e:
                    print(f"\n  Error: {e}")
                print("\n")

        else:
            print("Unknown command. Try: chat, list, delete, stats, clear, exit\n")

    print("\nBye!")
    graphdb.close()


if __name__ == "__main__":
    main()