"""Stage 1 - Collect: turn raw PDFs / HTML / text in data/raw into clean, overlapping chunks.

Usage:
    python -m src.ingest            # writes data/processed/chunks.jsonl

Each chunk keeps the metadata needed for citations (title, url, activity centre, page).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import config

SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    source_file: str
    url: str = ""
    activity_centre: str = ""
    publisher: str = ""
    page: int | None = None
    chunk_index: int = 0
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        """Flat metadata for ChromaDB (values must be str/int/float/bool)."""
        d = asdict(self)
        d.pop("text")
        d.pop("extra")
        d["page"] = d["page"] if d["page"] is not None else -1
        return d


# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #
def load_sources(path: Path = config.SOURCES_CSV) -> dict[str, dict]:
    """Map local_file name -> row from sources.csv (for citation metadata)."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["local_file"]: row for row in csv.DictReader(f) if row.get("local_file")}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. PyMuPDF first (fast), pdfplumber as fallback."""
    pages: list[tuple[int, str]] = []
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append((i, page.get_text("text")))
    except Exception:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                pages.append((i, page.extract_text() or ""))
    if sum(len(t.strip()) for _, t in pages) < 50:
        print(f"  ! {path.name}: almost no text extracted - probably a scanned PDF "
              "(needs OCR or manual transcription)", file=sys.stderr)
    return pages


def extract_html(path: Path) -> tuple[str, str]:
    """Return (title, main_text) with navigation/boilerplate stripped."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form",
                     "noscript", "svg", "button"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    # keep headings/paragraphs/list items as separate lines
    text = main.get_text("\n", strip=True)
    return title, text


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    text = re.sub(r"-\n(\w)", r"\1", text)                 # de-hyphenate line breaks
    text = re.sub(r"(?im)^\s*page \d+( of \d+)?\s*$", "", text)  # page footers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def chunk_text(text: str, size: int = config.CHUNK_SIZE_WORDS,
               overlap: int = config.CHUNK_OVERLAP_WORDS,
               min_words: int = config.MIN_CHUNK_WORDS) -> list[str]:
    """Sentence-aware sliding window measured in words.

    Sentences are packed until `size` words; the next chunk starts with the last
    ~`overlap` words of the previous one so facts spanning a boundary are not lost.
    Over-long single sentences (e.g. table rows) are hard-split.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    # hard-split over-long "sentences" (tables, lists) so there is always room for overlap
    piece = size - overlap
    sentences: list[list[str]] = []
    for s in split_sentences(text):
        words = s.split()
        for i in range(0, len(words), piece):
            sentences.append(words[i:i + piece])

    chunks: list[str] = []
    current: list[str] = []
    fresh = 0  # words in `current` that are not overlap carried from the last chunk
    for words in sentences:
        if current and len(current) + len(words) > size:
            chunks.append(" ".join(current))
            keep = max(0, min(overlap, size - len(words)))
            current = current[-keep:] if keep else []
            fresh = 0
        current.extend(words)
        fresh += len(words)
    if current and fresh:
        if chunks and fresh < min_words and len(chunks[-1].split()) + fresh <= size:
            # tiny tail: append the new words to the previous chunk instead
            chunks[-1] = chunks[-1] + " " + " ".join(current[-fresh:])
        else:
            chunks.append(" ".join(current))
    return chunks


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _doc_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def process_file(path: Path, meta: dict | None = None) -> list[Chunk]:
    meta = meta or {}
    doc_id = meta.get("doc_id") or _doc_id(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        units = [(p, clean_text(t)) for p, t in extract_pdf(path)]
        default_title = path.stem
    elif suffix in {".html", ".htm"}:
        default_title, text = extract_html(path)
        units = [(None, clean_text(text))]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        first = text.strip().splitlines()[0] if text.strip() else path.stem
        default_title = first.lstrip("# ").strip()[:120]
        units = [(None, clean_text(text))]

    title = meta.get("title") or default_title
    chunks: list[Chunk] = []
    idx = 0
    for page, text in units:
        for piece in chunk_text(text):
            h = hashlib.md5(f"{doc_id}:{idx}:{piece}".encode()).hexdigest()[:10]
            chunks.append(Chunk(
                chunk_id=f"{doc_id}-{idx}-{h}",
                doc_id=doc_id,
                text=piece,
                title=title,
                source_file=path.name,
                url=meta.get("url", ""),
                activity_centre=meta.get("activity_centre", ""),
                publisher=meta.get("publisher", ""),
                page=page,
                chunk_index=idx,
            ))
            idx += 1
    return chunks


def iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p


def ingest(raw_dir: Path = config.RAW_DIR, out_file: Path = config.CHUNKS_FILE,
           sources_csv: Path = config.SOURCES_CSV) -> list[Chunk]:
    sources = load_sources(sources_csv)
    all_chunks: list[Chunk] = []
    files = list(iter_raw_files(raw_dir))
    if not files:
        print(f"No documents found in {raw_dir}. Run scripts/fetch_sources.py or add files.")
    for path in files:
        meta = sources.get(path.name, {})
        if not meta:
            print(f"  ! {path.name} is not listed in sources.csv - citation will lack a URL")
        chunks = process_file(path, meta)
        print(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks from {len(files)} files -> {out_file}")
    return all_chunks


def load_chunks(path: Path = config.CHUNKS_FILE) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    ingest()
