# Declaration of AI and External Tools

## 1. Models & APIs Used
- **LLM Inference:** `openai/gpt-oss-120b` accessed via the **Groq API** (temperature=0.0 for deterministic factual question-answering evaluation).
- **Embedding Model:** `all-MiniLM-L6-v2` via **sentence-transformers** for local, dense semantic embeddings normalized for cosine similarity retrieval.
- **Vector Database:** **ChromaDB** (`PersistentClient`) for local vector indexing, persistent storage, and cosine-distance K-nearest-neighbor retrieval.

---

## 2. Third-Party Libraries
- `groq`: Official client library for fast inference through Groq Cloud.
- `sentence-transformers`: Local text embedding generation.
- `chromadb`: Vector indexing and similarity search.
- `python-dotenv`: Environment variable management for `.env` loading.
- `matplotlib`: Bar chart visualization of evaluation and faithfulness metrics.

---

## 3. AI Coding Assistance
- **Claude (Anthropic)** was used throughout development to:
  - Help design the chunking strategy (code-fence preservation, orphaned paragraph merging, and bullet-list decomposition).
  - Debug environment and path-resolution bugs across scripts and directories.
  - Draft candidate evaluation questions from source documentation (which were subsequently manually verified against actual source chunks before freezing `questions.json`).
  - Analyze and grade the baseline and improved evaluation results.

---

## 4. Personal Verification
I personally verified all project deliverables and pipeline stages:
- **Local Execution:** Ran all scripts locally and verified end-to-end execution.
- **Bug Discovery & Resolution:** Identified and resolved real bugs during development, including:
  - Windows/WSL path mismatches and relative path resolution.
  - Incorrect environment variable casing and loading behavior.
  - A dict-vs-list JSON parsing bug when loading evaluation questions.
- **Ground-Truth Cross-Checking:** Manually cross-checked every question's `expected_evidence` `chunk_id` directly against the raw source text chunks prior to freezing `questions.json`.
- **Grading & Failure Classification Review:** Personally reviewed all graded results, abstention behaviors, parametric leakage incidents, and failure classifications for accuracy before drawing final conclusions.
