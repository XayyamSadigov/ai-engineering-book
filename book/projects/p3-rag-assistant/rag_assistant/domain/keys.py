# path: book/projects/p3-rag-assistant/rag_assistant/domain/keys.py
"""Identity functions: document fingerprints, query normalization, index partition keys.

Each function answers "which inputs change the result?". Leaving one input out of a key is
how caches serve stale or leaked answers; putting an irrelevant one in only costs hit rate.
"""
from __future__ import annotations

import re
import unicodedata

from ragkit.documents import Document, short_hash
from ragkit.retrieval import Principal

from .models import SHARED_TENANT

_WS = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s?.!]+$")


def normalize_query(text: str) -> str:
    """Cache-key normalization only (never sent to retrieval): case, width, whitespace, final '?'."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _TRAILING.sub("", _WS.sub(" ", text)).strip()


def doc_fingerprint(doc: Document, *, chunker: str, rules: str, pipeline_version: str, enrichment: bool) -> str:
    """Everything that changes what the indexes hold for this document.

    Content hash alone is not enough: an ACL change leaves the text (and its hash) unchanged
    but must reach every index and cache. Same for a new authority rule or a new chunker.
    """
    meta = doc.metadata
    return short_hash(
        doc.content_hash,
        doc.version,
        doc.tenant,
        ",".join(sorted(doc.acl_groups)),
        meta.get("authority"),
        meta.get("effective_date"),
        ",".join(meta.get("supersedes") or []),
        chunker,
        rules,
        pipeline_version,
        enrichment,
        length=24,
    )


def acl_fingerprint(tenant: str | None, groups: list[str]) -> str:
    return short_hash(tenant, ",".join(sorted(groups)))


def partition_for_doc(tenant: str | None, mode: str) -> str:
    """Which index partition a document lives in: one shared index, or one per tenant."""
    if mode == "shared":
        return "all"
    return tenant or SHARED_TENANT


def partitions_for_principal(principal: Principal, mode: str) -> list[str]:
    """Which partitions a principal's query may touch. Namespace mode never opens another tenant's."""
    if mode == "shared":
        return ["all"]
    return sorted({principal.tenant, SHARED_TENANT})


def generation_scopes(tenant: str | None) -> list[str]:
    """Generation counters bumped by a change to a document of this tenant."""
    return [tenant or SHARED_TENANT]


def principal_scopes(principal: Principal) -> list[str]:
    """Generation counters a principal's cached results depend on."""
    return sorted({principal.tenant, SHARED_TENANT})


__all__ = [
    "acl_fingerprint", "doc_fingerprint", "generation_scopes", "normalize_query", "partition_for_doc",
    "partitions_for_principal", "principal_scopes",
]
