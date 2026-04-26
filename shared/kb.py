"""
Knowledge Base: ChromaDB + local sentence-transformers embeddings.
No API key needed for embedding -- everything runs locally.

Auto-rebuild: a SHA-256 hash of corpus.py is stored inside ChromaDB as
collection metadata. On every load, the hash is compared against the
current corpus.py. If they differ (corpus was edited), the collection is
automatically deleted and rebuilt so the index is never stale.
"""

import sys
import os
import hashlib

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console(highlight=True)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "technova_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, local, no API key

# Path to corpus.py -- used to compute the change-detection hash
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.py")


def _get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def _corpus_hash() -> str:
    """SHA-256 hash of corpus.py content. Changes whenever the corpus is edited."""
    with open(CORPUS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Create (or reuse) the ChromaDB collection from corpus.py.

    Auto-rebuild logic:
      - On first run: builds and stores corpus hash in collection metadata.
      - On subsequent runs: compares stored hash vs current corpus.py hash.
      - If hashes differ (corpus was edited): auto-rebuilds without needing --rebuild-kb.
      - force_rebuild=True always rebuilds regardless of hash.
    """
    from data.corpus import DOCUMENTS

    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = _get_embed_fn()
    current_hash = _corpus_hash()

    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing and not force_rebuild:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        stored_hash = collection.metadata.get("corpus_hash", "")

        if stored_hash == current_hash:
            # Corpus unchanged -- reuse the existing index
            console.print(Panel(
                f"[green][OK] Loaded existing KB[/green] -- "
                f"[bold]{collection.count()}[/bold] documents | corpus unchanged",
                title="Knowledge Base",
                border_style="green",
            ))
            return collection
        else:
            # Corpus was edited since last build -- must rebuild
            console.print(Panel(
                "[yellow]Corpus changed since last build -- rebuilding index...[/yellow]\n"
                "[dim]This happens automatically whenever data/corpus.py is edited.[/dim]",
                title="Knowledge Base",
                border_style="yellow",
            ))
            client.delete_collection(COLLECTION_NAME)

    elif COLLECTION_NAME in existing and force_rebuild:
        client.delete_collection(COLLECTION_NAME)

    # Build fresh collection, storing corpus hash in metadata for future change detection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={
            "hnsw:space": "cosine",
            "corpus_hash": current_hash,   # stored so next run can detect changes
        },
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Embedding & indexing knowledge corpus...", total=len(DOCUMENTS)
        )
        for doc in DOCUMENTS:
            collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                metadatas=[{
                    **doc["metadata"],
                    "title": doc["title"],
                    "category": doc["category"],
                }],
            )
            progress.advance(task)

    console.print(Panel(
        f"[bold green][OK] Knowledge Base built[/bold green] -- "
        f"[bold]{collection.count()}[/bold] documents indexed\n"
        f"Embedding model: [cyan]{EMBED_MODEL}[/cyan] (local, no API key)",
        title="Knowledge Base",
        border_style="green",
    ))
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
        output.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "similarity": round(1 - results["distances"][0][i], 4),
        })
    return output
