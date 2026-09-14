# Pipeline Reproduction Instructions (End-to-End Demo)

This guide describes how to reproduce the RAG-Guard pipeline end to end from the submission root (`Shashank_Nautiyal_AI_ML_LLM`).

---

## 1. Environment Setup & Prerequisites

Ensure you have Python 3.10+ installed. From the submission root, install all required dependencies:

```bash
pip install -r 02_SOURCE/requirements.txt
```

*(Alternatively, if running from within `02_SOURCE/`, run `pip install -r requirements.txt`).*

---

## 2. API Key Configuration

Create a `.env` file inside `02_SOURCE/` by copying the provided `.env.example`:

```bash
cp 02_SOURCE/.env.example 02_SOURCE/.env
```

Open `02_SOURCE/.env` and replace `your_key_here` with your actual Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_groq_key_here
```

*(Note: You may also export `GROQ_API_KEY` directly in your shell environment: `export GROQ_API_KEY="your_key_here"`).*

---

## 3. Step 1 — Document Chunking

Run `chunker.py` to process the raw documentation files (`py311.jsonl` and `py312.jsonl`) into chunked records:

```bash
python 02_SOURCE/chunker.py
```

- **Input:** `03_DATA_EVALUATION/documents/py311.jsonl` and `03_DATA_EVALUATION/documents/py312.jsonl`
- **Output:** `03_DATA_EVALUATION/chunks.jsonl`
- **Behavior:** Respects markdown code fences, merges small paragraphs (<15 words), splits oversized bullet lists, and enforces a target chunk size of ~300 words with 1-paragraph overlap.

---

## 4. Step 2 — Vector Indexing & Embedding

Run `embed_and_store.py` to embed the chunks with `sentence-transformers` and build the persistent ChromaDB collection:

```bash
python 02_SOURCE/embed_and_store.py
```

- **Input:** `03_DATA_EVALUATION/chunks.jsonl`
- **Embedding Model:** `all-MiniLM-L6-v2` (cosine similarity space)
- **Output:** Persistent ChromaDB vector database directory `chroma_db/`
- **Behavior:** Idempotently recreates the `rag_guard_python_docs` collection and populates it in safe batch inserts.

---

## 5. Step 3 — Evaluation & Generation

Run `generate.py` to evaluate the dataset questions with both non-RAG and RAG inference:

```bash
python 02_SOURCE/generate.py --questions 03_DATA_EVALUATION/questions.json --out 04_EVIDENCE/baseline_results.csv
```

### CLI Arguments for `generate.py`:
- `--questions` (str, default: `data/questions.json`): Path to the evaluation questions JSON dataset.
- `--out` (str, default: `results/baseline_results.csv`): Path where output evaluation results CSV will be written.
- `--version_filter` (str, optional, default: `None`): Restrict retrieval to a specific Python documentation version (`py311` or `py312`).
- `--ids` (str, optional, default: `None`): Comma-separated question IDs to evaluate (e.g., `--ids q04,q07,q09`), enabling rapid verification on targeted questions.

---

## 6. Step 4 — Visualizing Results (Optional)

To regenerate the comparison bar charts from the graded evidence:

```bash
python 04_EVIDENCE/generate_plots.py
```

Generated charts will be written to `04_EVIDENCE/plots/`:
- `baseline_vs_improved_correctness.png`
- `faithfulness_violations.png`
