import json
import re
from pathlib import Path

TARGET_WORDS = 300      # target chunk size
MAX_WORDS = 450          # hard cap before forcing a split
MIN_WORDS = 15           # paragraphs smaller than this get merged, never left alone
OVERLAP_PARAS = 1        # number of trailing paragraphs carried into next chunk

CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
BULLET_SPLIT_RE = re.compile(r"\n(?=-\s)")  # split on newline followed by "- "


def split_into_paragraphs(text):
    """Split on blank lines, but keep fenced code blocks intact as single paragraphs."""
    # Temporarily replace code fences with placeholders so blank lines inside
    # them don't cause a split.
    code_blocks = []

    def _stash(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    protected = CODE_FENCE_RE.sub(_stash, text)
    raw_paras = re.split(r"\n\s*\n", protected)

    paras = []
    for p in raw_paras:
        p = p.strip()
        if not p:
            continue
        # restore any code blocks
        for i, block in enumerate(code_blocks):
            p = p.replace(f"__CODE_BLOCK_{i}__", block)
        paras.append(p)

    # Merge any paragraph shorter than MIN_WORDS into the following paragraph
    # (fixes orphaned section headers like a lone "Deprecated" line).
    merged = []
    carry = None
    for p in paras:
        if carry is not None:
            p = carry + "\n\n" + p
            carry = None
        if len(p.split()) < MIN_WORDS:
            carry = p
        else:
            merged.append(p)
    if carry is not None:
        # trailing tiny paragraph with nothing after it: attach to previous
        if merged:
            merged[-1] = merged[-1] + "\n\n" + carry
        else:
            merged.append(carry)

    # Break up any paragraph that is a big bullet-list blob and still exceeds
    # MAX_WORDS into per-bullet sub-paragraphs, so a single chunk doesn't end
    # up covering many unrelated bullet points.
    final = []
    for p in merged:
        if len(p.split()) > MAX_WORDS and BULLET_SPLIT_RE.search(p):
            pieces = BULLET_SPLIT_RE.split(p)
            final.extend([piece.strip() for piece in pieces if piece.strip()])
        else:
            final.append(p)

    return final


def chunk_page(paragraphs, version, source, title):
    chunks = []
    current = []
    current_words = 0

    def flush():
        if current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)

    for para in paragraphs:
        para_words = len(para.split())

        # If a single paragraph alone exceeds MAX_WORDS (e.g. huge code block),
        # keep it as its own chunk rather than breaking it apart.
        if para_words > MAX_WORDS:
            flush()
            chunks.append(para)
            current = []
            current_words = 0
            continue

        if current_words + para_words > TARGET_WORDS and current:
            flush()
            # carry overlap paragraphs into the new chunk
            current = current[-OVERLAP_PARAS:] if OVERLAP_PARAS else []
            current_words = sum(len(p.split()) for p in current)

        current.append(para)
        current_words += para_words

    flush()
    return chunks


def process_file(path, out_records):
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            paragraphs = split_into_paragraphs(doc["text"])
            chunks = chunk_page(paragraphs, doc["version"], doc["source"], doc["title"])
            for i, chunk_text in enumerate(chunks):
                out_records.append({
                    "chunk_id": f"{doc['version']}_{doc['file'].replace('.html','')}_{i:03d}",
                    "version": doc["version"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "chunk_index": i,
                    "n_chunks_in_page": len(chunks),
                    "word_count": len(chunk_text.split()),
                    "text": chunk_text,
                })


if __name__ == "__main__":
    out_records = []
    process_file("../data/cleaned/py311.jsonl", out_records)
    process_file("../data/cleaned/py312.jsonl", out_records)

    out_path = Path("../data/cleaned/chunks.jsonl")
    with open(out_path, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Total chunks: {len(out_records)}")
    from collections import Counter
    by_version = Counter(r["version"] for r in out_records)
    print("By version:", dict(by_version))

    word_counts = [r["word_count"] for r in out_records]
    print(f"Chunk word count: min={min(word_counts)}, max={max(word_counts)}, "
          f"avg={sum(word_counts)/len(word_counts):.0f}")
