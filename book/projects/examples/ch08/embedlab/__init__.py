# path: book/projects/examples/ch08/embedlab/__init__.py
"""Chapter 8: embeddings. Vector math, versioned spaces, a batching/caching pipeline,
quality evaluation, and non-RAG use cases. Built on aie_core.embeddings."""
from .pipeline import EmbeddingPipeline, NamespacedStore, prepare_text
from .space import EmbeddingSpace, SpaceMismatchError, VectorIndex, plan_reembed

__all__ = ["EmbeddingPipeline", "EmbeddingSpace", "NamespacedStore", "SpaceMismatchError", "VectorIndex", "plan_reembed", "prepare_text"]
