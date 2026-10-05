# path: book/projects/p2-semantic-search/semsearch/adapters/__init__.py
"""Vector store adapters behind the VectorStore protocol."""
from .base import VectorStore
from .numpy_store import NumpyVectorStore

__all__ = ["VectorStore", "NumpyVectorStore"]
