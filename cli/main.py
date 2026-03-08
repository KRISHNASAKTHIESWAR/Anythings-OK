import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from graphdb.model import graphdb
from llama_index.core.query_engine import RetrieverQueryEngine


def start_chat():

    print("\nGraphRAG CLI")
    print("----------------------")
    print("Commands:")
    print("  chat      → talk to the model")
    print("  list      → list documents")
    print("  delete    → delete document")
    print("  exit\n")

    # Load graph once
    graphdb.load_index()

    retriever = graphdb.get_retriever()

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=graphdb.llm,
        streaming=True
    )

    while True:

        cmd = input("\n> ").strip()

        if cmd == "exit":
            break

        elif cmd == "list":

            docs = graphdb.list_documents()

            if not docs:
                print("No documents stored.")
                continue

            print("\nStored Documents\n")

            for d in docs:
                print(f"{d['doc_id']}  |  {d['file']}")

        elif cmd == "delete":

            doc_id = input("Enter doc_id: ")
            graphdb.delete_document(doc_id)
            print("Deleted.")

        elif cmd == "chat":

            print("\nChat mode (type 'back' to return)\n")

            while True:

                q = input("You: ")

                if q == "back":
                    break

                stream = query_engine.query(q)

                print("\nLLM: ", end="", flush=True)

                for token in stream.response_gen:
                    print(token, end="", flush=True)

                print("\n")

        else:
            print("Unknown command.")


if __name__ == "__main__":
    start_chat()