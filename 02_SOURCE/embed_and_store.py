"""
embed_and_store.py

Embeds chunks.jsonl (output of chunker.py) using sentence-transformers
and stores them in a persistent ChromaDB collection for retrieval.

Usage:
    python3 embed_and_store.py

Produces:
    ./chroma_db/   (persistent ChromaDB storage directory)
"""

import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.jsonl"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_guard_python_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough for this scale
BATCH_SIZE = 64


def load_chunks(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_collection(records, model):
    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh collection each run so re-embedding is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [r["chunk_id"] for r in records]
    documents = [r["text"] for r in records]
    metadatas = [
        {
            "version": r["version"],
            "source": r["source"],
            "title": r["title"],
            "chunk_index": r["chunk_index"],
            "word_count": r["word_count"],
        }
        for r in records
    ]

    print(f"Embedding {len(documents)} chunks with {EMBEDDING_MODEL} ...")
    t0 = time.time()
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity works cleanly on normalized vectors
    )
    print(f"Embedding took {time.time() - t0:.1f}s")

    # Chroma has a per-call batch limit; insert in chunks to be safe.
    INSERT_BATCH = 500
    for i in range(0, len(ids), INSERT_BATCH):
        collection.add(
            ids=ids[i : i + INSERT_BATCH],
            documents=documents[i : i + INSERT_BATCH],
            metadatas=metadatas[i : i + INSERT_BATCH],
            embeddings=embeddings[i : i + INSERT_BATCH].tolist(),
        )

    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'")
    return collection


def retrieve(collection, model, query, k=3, version_filter=None):
    """
    Retrieve top-k chunks for a query.
    version_filter: None, "py311", or "py312" -- restricts search to one doc version.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    where = {"version": version_filter} if version_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "chunk_id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )
    return hits


if __name__ == "__main__":
    records = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(records)} chunks from {CHUNKS_PATH}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = build_collection(records, model)

    # Quick sanity-check retrieval
    test_queries = [
        "Was distutils removed in Python 3.12?",
        "What is the new syntax for generic type parameters introduced in PEP 695?",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        hits = retrieve(collection, model, q, k=3)
        for h in hits:
            print(f"  [{h['distance']:.4f}] {h['chunk_id']} ({h['metadata']['version']})")
            print(f"    {h['text'][:150].strip()}...")
