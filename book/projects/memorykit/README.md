# memorykit

Memory stores for AI applications, built in Chapter 21 (Memory Systems). Typed memory
records with provenance, confidence, and expiry; tenant-scoped stores with hard delete,
tombstones, and cascade to derived records; a write policy that blocks memory poisoning,
secrets, and unapproved PII; consolidation of duplicates and conflicts; and four memory
types: conversation, semantic, episodic, and user profile.

## Layout

```
memorykit/
  models.py        MemoryRecord, Owner, MemoryKind, Source, Sensitivity, MemoryStatus, Tombstone
  store.py         MemoryStore protocol, InMemoryStore, SQLiteStore, VersionConflict, keyed fingerprint
  policy.py        WritePolicy, consolidate, write (policy -> prepare -> consolidate)
  conversation.py  ConversationMemory: window + rolling summary + verified exact facts, redact, promote
  semantic.py      SemanticMemory: embed, recall with relevance gate + recency + salience + source weight
  episodic.py      EpisodicStore: task trajectories and outcomes, similar-episode retrieval
  profile.py       UserProfileMemory: propose / confirm / reject / forget
  evaluation.py    evaluate_recall, evaluate_write_policy
tests/             offline tests, each store test runs against both stores
```

## Install and test

```bash
# from the book root, into the shared virtualenv
uv pip install --python .venv/bin/python -e book/projects/aie_core -e book/projects/memorykit
# or: pip install -e ../aie_core && pip install -e .
cd book/projects/memorykit
python -m pytest -q
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | `fake` | summarizer and fact extractor client (via `aie_core.make_llm_client`) |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `fake` | embedding client for semantic and episodic memory |
| `MEMORY_DB_PATH` | `memory.db` | file for `SQLiteStore` in your application |
| `MEMORY_FINGERPRINT_SECRET` | none | HMAC key for tombstone fingerprints; pass as `fingerprint_secret=` |

## Usage

```python
from aie_core import make_embedding_client
from memorykit import Owner, SemanticMemory, Source, SQLiteStore, UserProfileMemory

store = SQLiteStore("memory.db", fingerprint_secret=b"...from your secret store...")
ana = Owner(tenant="retail", user="ana")

profile = UserProfileMemory(store)
profile.propose(ana, "preferred_language", "es", source=Source.USER_STATED, provenance=["turn:s1#2"])

semantic = SemanticMemory(store, make_embedding_client())
semantic.remember(ana, "Ana works the night shift at the Lisbon warehouse", source=Source.USER_STATED)
hits = semantic.recall(ana, "what shift does Ana work")
```
