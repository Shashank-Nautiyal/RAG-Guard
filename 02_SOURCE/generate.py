"""
generate.py

For each question in data/questions.json, produces two answers:
  1. no_rag  -> LLM answers directly from its own knowledge (no context)
  2. rag     -> LLM answers using top-K retrieved chunks from ChromaDB

Requires:
  pip install groq sentence-transformers chromadb
  export GROQ_API_KEY="your_key_here"

Usage:
    python3 generate.py --questions data/questions.json --out results/baseline_results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # falls back to whatever is already in the environment

SCRIPT_DIR = Path(__file__).parent
DB_DIR = str(SCRIPT_DIR / "chroma_db")
COLLECTION_NAME = "rag_guard_python_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 3

NO_RAG_SYSTEM_PROMPT = (
    "Answer the user's question directly and concisely, using your own "
    "knowledge. Do not mention that you lack access to external documents."
)

RAG_SYSTEM_PROMPT = (
    "You are answering questions using ONLY the provided context chunks from "
    "Python documentation. Follow these rules strictly:\n"
    "1. Base your answer only on the provided context, not on outside knowledge, "
    "even if you are confident you know the correct answer from general Python "
    "knowledge. Confidence from training data is NOT a valid basis for an answer here.\n"
    "2. Before writing your final answer, first quote the exact sentence(s) from "
    "the context that support each claim you are about to make, under a heading "
    "'Supporting evidence:'. If you cannot find an exact or closely paraphrased "
    "supporting sentence for a claim, you may not include that claim.\n"
    "3. If, after checking, the context does not contain an explicit supporting "
    "sentence for the core of the question, respond with exactly: \"I don't have "
    "enough information in the provided context to answer this.\" Do not fill the "
    "gap with outside knowledge, even partially.\n"
    "4. When you do answer, write a final 'Answer:' section citing the chunk_id(s) "
    "you used in square brackets, e.g. [py312_whatsnew_3.12_002].\n"
    "5. Do not state facts that are not directly supported by the quoted evidence."
)


def build_context_block(hits):
    parts = []
    for h in hits:
        parts.append(f"[{h['chunk_id']}] ({h['metadata']['version']})\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def retrieve(collection, embed_model, query, k=TOP_K, version_filter=None):
    t0 = time.time()
    query_embedding = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
    where = {"version": version_filter} if version_filter else None
    results = collection.query(query_embeddings=[query_embedding], n_results=k, where=where)
    retrieval_time = time.time() - t0

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return hits, retrieval_time


def call_llm(client, system_prompt, user_prompt):
    t0 = time.time()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # deterministic-ish, appropriate for factual QA eval
    )
    elapsed = time.time() - t0
    answer = completion.choices[0].message.content
    return answer, elapsed


def answer_no_retrieval(client, question):
    return call_llm(client, NO_RAG_SYSTEM_PROMPT, question)


def answer_rag(client, collection, embed_model, question, k=TOP_K, version_filter=None):
    hits, retrieval_time = retrieve(collection, embed_model, question, k=k, version_filter=version_filter)
    context_block = build_context_block(hits)
    user_prompt = f"Context:\n\n{context_block}\n\nQuestion: {question}"
    answer, generation_time = call_llm(client, RAG_SYSTEM_PROMPT, user_prompt)
    return {
        "answer": answer,
        "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
        "retrieved_distances": [round(h["distance"], 4) for h in hits],
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(generation_time, 3),
    }


def run(questions_path, out_path, version_filter=None, only_ids=None):
    with open(questions_path) as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) and "questions" in data else data

    if only_ids:
        questions = [q for q in questions if q["id"] in only_ids]

    client = Groq()  # reads GROQ_API_KEY from environment
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    fieldnames = [
        "id", "category", "question", "answerable",
        "expected_answer", "expected_evidence",
        "no_rag_answer", "no_rag_time",
        "rag_answer", "retrieved_chunk_ids", "retrieved_distances",
        "retrieval_time", "generation_time", "total_time",
    ]

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            print(f"Running {q['id']}: {q['question'][:60]}...")

            no_rag_answer, no_rag_time = answer_no_retrieval(client, q["question"])

            rag_result = answer_rag(
                client, collection, embed_model, q["question"],
                version_filter=version_filter,
            )

            row = {
                "id": q["id"],
                "category": q.get("category", ""),
                "question": q["question"],
                "answerable": q.get("answerable", ""),
                "expected_answer": q.get("expected_answer", ""),
                "expected_evidence": json.dumps(q.get("expected_evidence", [])),
                "no_rag_answer": no_rag_answer,
                "no_rag_time": round(no_rag_time, 3),
                "rag_answer": rag_result["answer"],
                "retrieved_chunk_ids": json.dumps(rag_result["retrieved_chunk_ids"]),
                "retrieved_distances": json.dumps(rag_result["retrieved_distances"]),
                "retrieval_time": rag_result["retrieval_time"],
                "generation_time": rag_result["generation_time"],
                "total_time": round(
                    no_rag_time + rag_result["retrieval_time"] + rag_result["generation_time"], 3
                ),
            }
            writer.writerow(row)

    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--out", default="results/baseline_results.csv")
    parser.add_argument("--version_filter", default=None, help="py311 or py312, optional")
    parser.add_argument("--ids", default=None, help="Comma-separated question ids to run, e.g. q04,q07,q09")
    args = parser.parse_args()
    only_ids = set(args.ids.split(",")) if args.ids else None
    run(args.questions, args.out, version_filter=args.version_filter, only_ids=only_ids)
