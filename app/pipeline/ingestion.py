"""Document loading.

Supports `.txt`, `.md`, `.pdf`. Directory inputs are walked recursively.
Returns a list of `Document` records carrying source-file metadata that
flows all the way through to the response's `sources` field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_pymupdf(path: Path) -> str:
    import fitz  # PyMuPDF

    pages: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return "\n\n".join(pages)


def _read_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_pdf(path: Path) -> str:
    """PyMuPDF gives better layout-aware extraction; fall back to pypdf.

    Backend is selected by `PDF_BACKEND` (auto|pymupdf|pypdf). On 'auto'
    we try PyMuPDF first because it handles columned/figured PDFs more
    reliably, then fall back to pypdf which is a hard dep.
    """
    backend = settings.pdf_backend
    errors: list[str] = []
    order: list[str]
    if backend == "pymupdf":
        order = ["pymupdf"]
    elif backend == "pypdf":
        order = ["pypdf"]
    else:
        order = ["pymupdf", "pypdf"]

    for choice in order:
        try:
            if choice == "pymupdf":
                return _read_pdf_pymupdf(path)
            return _read_pdf_pypdf(path)
        except ImportError as e:
            errors.append(f"{choice}: not installed ({e})")
            continue
        except Exception as e:
            errors.append(f"{choice}: {e}")
            continue
    raise RuntimeError(f"All PDF backends failed for {path}: {'; '.join(errors)}")


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
