# Chapter 8 examples: embeddings

Vector math, versioned embedding spaces, a batching and caching pipeline, embedding quality
evaluation, and six non-RAG use cases, all on the Northwind shared data. Everything runs
offline with `FakeEmbeddings(vocabulary=...)`, a bag-of-words model whose similarity tracks
shared words. Point the settings at a real provider and the same code runs against it.

```
embedlab/
  vector_math.py        cosine/dot/Euclidean, normalization, truncation, centroid, similarity profile
  corpus.py             load docs and tickets, split sections, build the client from settings
  space.py              EmbeddingSpace fingerprint, VectorIndex (refuses mixed spaces, delete), plan_reembed (cost + window)
  pipeline.py           EmbeddingPipeline: prep, prefixes, token-aware batching with retries, namespaced cache, cost
  quality.py            recall@k, MRR, chunk vs document, Matryoshka truncation report, neighbours
  usecases/
    dedup.py            threshold selection from labeled pairs, duplicate groups
    clustering.py       k-means, silhouette k choice, cluster naming, purity
    classify.py         kNN and centroid classifiers with abstention, leave-one-out
    routing.py          intent router with threshold, margin, fallback, calibration
    anomaly.py          distance to nearest centroid, held-out calibration, flag rate, drift score
    recommend.py        related items with MMR and a relevance floor
data/                   labeled queries, dedup pairs, routes (synthetic, chapter-local)
demo.py                 runs every experiment and prints a report
tests/                  offline tests
```

## Run

```bash
# from the repository root, shared venv
uv pip install --python .venv/bin/python -e book/projects/aie_core   # or: pip install -e ../../aie_core
.venv/bin/python -m pytest book/projects/examples/ch08 -q
cd book/projects/examples/ch08 && ../../../../.venv/bin/python demo.py
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fake` | `fake` builds a bag-of-words model over the corpus; `openai` uses any OpenAI-compatible endpoint |
| `EMBEDDING_MODEL` | `fake-embedding` | model name sent to the provider and recorded in the embedding space |
| `LLM_BASE_URL` | unset | base URL for an OpenAI-compatible server or proxy |
| `OPENAI_API_KEY` | unset | credential for the endpoint |

Integration tests (`-m integration`) run only when `EMBEDDING_PROVIDER` is not `fake`.
