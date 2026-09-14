# RAG-Guard

**Evaluating and Improving the Reliability of Retrieval-Augmented Generation Systems**

RAG-Guard is a small experimental framework for measuring how reliably a RAG system answers questions from a knowledge corpus — and for testing whether a targeted, evidence-driven improvement can reduce its most important failure mode. The system under test is a RAG pipeline over official Python 3.11 and 3.12 documentation; the actual deliverable is the evaluation methodology and findings, not the chatbot itself.

## Core question

> How reliably can a RAG system answer questions from a knowledge corpus, and can a targeted improvement reduce its most important failure mode?

## Corpus

19 official documentation pages each from Python 3.11 and 3.12 (38 total), covering `asyncio`, `collections`, `json`, `os`, `pathlib`, `re`, exception handling, and the `whatsnew` release notes for both versions. This corpus was chosen specifically because it contains **real, documented version differences** (e.g. `distutils` removed in 3.12, PEP 701 f-string changes) rather than artificially manufactured contradictions.

## Pipeline

```
Raw docs (py311.jsonl / py312.jsonl)
        │
        ▼
  chunker.py            paragraph-aware chunking (~300 words/chunk),
                         preserves code fences and headings,
                         merges orphaned headers, splits oversized bullet lists
        │
        ▼
  embed_and_store.py    embeds chunks with all-MiniLM-L6-v2,
                         stores in a persistent ChromaDB collection
        │
        ▼
  generate.py            for each question: retrieves top-3 chunks,
                          calls openai/gpt-oss-120b (via Groq) twice —
                          once with no context (control) and once RAG-grounded —
                          logs both answers, retrieved chunks, and timing
```

## Evaluation

A frozen set of 24 hand-verified questions (`questions.json`) spanning six categories designed to stress different failure modes:

| Category | Count | Tests |
|---|---|---|
| Direct | 5 | Baseline retrieval/generation competence |
| Paraphrased | 5 | Semantic (not keyword) retrieval |
| Multi-document | 4 | Retrieval across more than one chunk/page |
| Version-sensitive | 4 | Correctly distinguishing 3.11 vs. 3.12 facts |
| Unanswerable | 4 | Correct abstention when evidence is absent |
| Distractor | 2 | Resisting hallucination when related-but-irrelevant context is retrieved |

Every `expected_evidence` chunk_id was manually verified against the real corpus text before the file was frozen — see `06_DECLARATION/AI_EXTERNAL_TOOLS.md` for what was verified and how.

## Key finding: parametric leakage, not retrieval failure

The baseline system scored 88% correctness, but investigation revealed several of those "correct" answers were not actually grounded in the retrieved context — the model was quietly falling back on its own training knowledge when retrieval quality was weak, despite an explicit system-prompt instruction not to. This is more dangerous than an obvious retrieval miss, because it looks reliable in testing while being fundamentally ungrounded.

## The improvement

The generation prompt was changed to force the model to quote exact supporting sentences from the retrieved context **before** producing an answer, and to abstain if no genuine supporting quote could be found — making it structurally harder to silently answer from memory.

## Result: a real trade-off, not a clean win

| Metric | Baseline | Improved |
|---|---|---|
| Overall correctness | 21/24 (88%) | 19/24 (79%) |
| Faithfulness violations (ungrounded/leaked answers) | 4 | **0** |
| Retrieval hit rate | 15/18 (83%, corrected) | unchanged |

The stricter grounding prompt eliminated all detected hallucination-via-leakage, but at the cost of two questions where the system became more conservative and abstained rather than partially answer with incomplete evidence. This is reported as an honest precision/safety-vs-recall trade-off rather than an unqualified improvement — see the Final Report for the full discussion, including one case of observed run-to-run non-determinism from the LLM despite `temperature=0.0`.

## Repository structure

```
├── 01_FINAL_REPORT/       Formal write-up (problem → methodology → evaluation → findings)
├── 02_SOURCE/             chunker.py, embed_and_store.py, generate.py, requirements.txt
├── 03_DATA_EVALUATION/    Raw docs, chunks.jsonl, frozen questions.json
├── 04_EVIDENCE/           Baseline/improved results, graded scorecards, comparison table, plots
├── 05_DEMO/               Reproduction instructions
└── 06_DECLARATION/        AI/tool usage disclosure
```

## Reproducing this project

See `05_DEMO/demo_instructions.md` for exact setup and run steps.

## Limitations

- Small sample size (24 questions, as few as 2-5 per category) — results are diagnostic of failure patterns, not statistically generalizable performance estimates.
- A general-purpose small embedding model (`all-MiniLM-L6-v2`) was used without domain fine-tuning.
- Observed at least one instance of answer variance between runs on an identical question with identical retrieved context, despite `temperature=0.0`.
- Chunk boundaries can fall at different points between the 3.11 and 3.12 versions of the same page even at the same chunk index, since chunking is independent per document.

## Disclosure

AI tool usage during development is fully declared in `06_DECLARATION/AI_EXTERNAL_TOOLS.md`.
