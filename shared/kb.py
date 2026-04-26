"""
Knowledge Base: ChromaDB + local sentence-transformers embeddings.
No API key needed for embedding -- everything runs locally.
"""

import sys
import os

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console(highlight=True)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "technova_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, local, no API key


def _get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """Create (or reuse) the ChromaDB collection from corpus.py."""
    from data.corpus import DOCUMENTS

    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = _get_embed_fn()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        if not force_rebuild:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
            )
            console.print(
                Panel(
                    f"[green][OK] Loaded existing KB[/green] -- "
                    f"[bold]{collection.count()}[/bold] documents in ChromaDB",
                    title="Knowledge Base",
                    border_style="green",
                )
            )
            return collection
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding & indexing knowledge corpus...", total=len(DOCUMENTS))
        for doc in DOCUMENTS:
            collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                metadatas=[{**doc["metadata"], "title": doc["title"], "category": doc["category"]}],
            )
            progress.advance(task)

    console.print(
        Panel(
            f"[bold green][OK] Knowledge Base built[/bold green] -- "
            f"[bold]{collection.count()}[/bold] documents indexed\n"
            f"Embedding model: [cyan]{EMBED_MODEL}[/cyan] (local, no API key)",
            title="Knowledge Base",
            border_style="green",
        )
    )
    return collection


def semantic_search(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 3,
) -> list[dict]:
    """Pure vector-similarity search. Returns list of result dicts."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": round(1 - results["distances"][0][i], 4),
            }
        )
    return output
