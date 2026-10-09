"""Module corpus — Extraction, nettoyage et découpage du corpus juridique."""

from corpus.extraction import extract_all_documents
from corpus.nettoyage import clean_all_documents
from corpus.decoupage import chunk_all_documents

__all__ = [
    "extract_all_documents",
    "clean_all_documents",
    "chunk_all_documents",
]
