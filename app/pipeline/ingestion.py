"""Document loading.

Supports `.txt`, `.md`, `.pdf`. Directory inputs are walked recursively.
Returns a list of `Document` records carrying source-file metadata that
flows all the way through to the response's `sources` field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pypdf is required for PDF ingestion") from e

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _expand(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            logger.warning("Path not found, skipping: %s", p)
            continue
        if p.is_dir():
            out.extend(
                f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
        else:
            logger.warning("Unsupported file type, skipping: %s", p)
    return out


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    """Load every supported file under `paths` into Document objects."""
    files = _expand(paths)
    docs: list[Document] = []
    for f in files:
        try:
            text = _read_pdf(f) if f.suffix.lower() == ".pdf" else _read_text(f)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Failed to read %s: %s", f, e)
            continue
        text = text.strip()
        if not text:
            logger.warning("Empty document, skipping: %s", f)
            continue
        docs.append(
            Document(
                text=text,
                metadata={"source": str(f.resolve()), "filename": f.name},
            )
        )
    logger.info("Loaded %d document(s) from %d path(s)", len(docs), len(files))
    return docs
