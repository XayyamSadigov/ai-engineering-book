-- path: book/projects/examples/ch28/schema.sql
-- Relational schema for the state, lineage, and audit of an AI application (Chapter 28).
-- PostgreSQL 16 + pgvector. Every tenant-owned row carries tenant_id; every generated
-- artifact points at the versions that produced it.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- versions ----
-- The things that change behavior without a code deploy. Each is a row, never a string
-- buried in a config file, so that a message, a chunk, or an eval run can reference it.

CREATE TABLE prompt_versions (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT NOT NULL,                 -- 'assist.answer'
    version       TEXT NOT NULL,                 -- '3'
    template_hash TEXT NOT NULL,                 -- sha256 of the rendered template
    body          TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE model_versions (
    id            BIGSERIAL PRIMARY KEY,
    provider      TEXT NOT NULL,                 -- 'openai' | 'anthropic' | 'self-hosted'
    model         TEXT NOT NULL,                 -- pinned identifier as the provider reports it
    kind          TEXT NOT NULL CHECK (kind IN ('chat', 'embedding', 'rerank', 'judge')),
    dimensions    INT,                           -- embeddings only
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, model, kind)
);

CREATE TABLE index_versions (
    id                 BIGSERIAL PRIMARY KEY,
    name               TEXT NOT NULL,            -- 'knowledge'
    version            TEXT NOT NULL,            -- 'idx-2026-01'
    embedding_model_id BIGINT NOT NULL REFERENCES model_versions(id),
    chunker_config     JSONB NOT NULL,           -- strategy, size, overlap
    status             TEXT NOT NULL CHECK (status IN ('building', 'active', 'retired')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE tool_schema_versions (
    id          BIGSERIAL PRIMARY KEY,
    tool_name   TEXT NOT NULL,
    version     TEXT NOT NULL,
    schema      JSONB NOT NULL,                  -- JSON Schema of the arguments
    side_effect TEXT NOT NULL CHECK (side_effect IN ('read', 'reversible', 'irreversible', 'external')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tool_name, version)
);

CREATE TABLE policy_versions (
    id         BIGSERIAL PRIMARY KEY,
    version    TEXT NOT NULL UNIQUE,
    rules      JSONB NOT NULL,                   -- allowlists, approval rules, rate limits
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE evaluator_versions (
    id             BIGSERIAL PRIMARY KEY,
    name           TEXT NOT NULL,                -- 'groundedness-judge'
    version        TEXT NOT NULL,
    judge_model_id BIGINT REFERENCES model_versions(id),
    rubric_hash    TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

-- ------------------------------------------------------------ conversations ----

CREATE TABLE conversations (
    id         UUID PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX conversations_tenant_user ON conversations (tenant_id, user_id, updated_at DESC);

CREATE TABLE messages (
    id                 UUID PRIMARY KEY,
    conversation_id    UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tenant_id          TEXT NOT NULL,            -- denormalized so row-level filters never join
    seq                INT  NOT NULL,            -- order inside the conversation
    role               TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content            TEXT NOT NULL,
    content_hash       TEXT NOT NULL,
    -- lineage: NULL for user messages, populated for assistant messages
    request_id         TEXT,
    trace_id           TEXT,
    prompt_version_id  BIGINT REFERENCES prompt_versions(id),
    model_version_id   BIGINT REFERENCES model_versions(id),
    index_version_id   BIGINT REFERENCES index_versions(id),
    policy_version_id  BIGINT REFERENCES policy_versions(id),
    evidence_chunk_ids UUID[] NOT NULL DEFAULT '{}',
    tool_calls         JSONB,                    -- [{name, args_hash, tool_schema_version_id, result_hash}]
    usage              JSONB,                    -- {input_tokens, output_tokens, cached_input_tokens, cost_usd}
    latency_ms         INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);
CREATE INDEX messages_tenant_created ON messages (tenant_id, created_at DESC);
CREATE INDEX messages_prompt_version ON messages (prompt_version_id);  -- "who was affected by prompt v3?"

-- ---------------------------------------------------------------------- jobs ----

CREATE TABLE jobs (
    id              UUID PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    type            TEXT NOT NULL,               -- 'ingest_document' | 'evaluate' | 'agent_run'
    state           TEXT NOT NULL CHECK (state IN ('queued', 'running', 'waiting_approval',
                                                   'succeeded', 'failed', 'cancelled')),
    payload         JSONB NOT NULL,
    result          JSONB,
    error           TEXT,
    attempts        INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 3,
    idempotency_key TEXT,
    callback_url    TEXT,
    leased_by       TEXT,                        -- worker instance id
    lease_until     TIMESTAMPTZ,                 -- expired lease => job is re-queued
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, idempotency_key)
);
CREATE INDEX jobs_pollable ON jobs (state, created_at) WHERE state IN ('queued', 'running');

-- ------------------------------------------------------------------ documents ----
-- documents are the logical unit the user knows; document_versions are what we actually
-- parsed; chunks belong to a (document_version, index_version) pair so that two indexes
-- built with different embedding models can coexist during a migration.

CREATE TABLE documents (
    id          UUID PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    source_uri  TEXT NOT NULL,                   -- where the raw bytes came from
    object_key  TEXT NOT NULL,                   -- raw bytes in object storage
    acl_groups  TEXT[] NOT NULL DEFAULT '{all}',
    metadata    JSONB NOT NULL DEFAULT '{}',
    deleted_at  TIMESTAMPTZ,                     -- soft delete; propagation is a job
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_uri)
);

CREATE TABLE document_versions (
    id           UUID PRIMARY KEY,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,                  -- same hash => skip re-ingestion
    parser       TEXT NOT NULL,                  -- 'pdf-text-layer@2'
    parsed_object_key TEXT NOT NULL,             -- normalized text in object storage
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, content_hash)
);

CREATE TABLE chunks (
    id                  UUID PRIMARY KEY,
    document_version_id UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    index_version_id    BIGINT NOT NULL REFERENCES index_versions(id),
    tenant_id           TEXT NOT NULL,           -- denormalized for the retrieval filter
    acl_groups          TEXT[] NOT NULL,         -- denormalized for the retrieval filter
    ordinal             INT NOT NULL,
    text                TEXT NOT NULL,
    tsv                 TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding           VECTOR(1536),            -- dimension matches index_versions.embedding_model
    metadata            JSONB NOT NULL DEFAULT '{}',
    UNIQUE (document_version_id, index_version_id, ordinal)
);
CREATE INDEX chunks_tenant_index ON chunks (tenant_id, index_version_id);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- ----------------------------------------------------------------- evaluation ----

CREATE TABLE evaluation_runs (
    id                   UUID PRIMARY KEY,
    dataset_name         TEXT NOT NULL,
    dataset_version      TEXT NOT NULL,
    prompt_version_id    BIGINT REFERENCES prompt_versions(id),
    model_version_id     BIGINT REFERENCES model_versions(id),
    index_version_id     BIGINT REFERENCES index_versions(id),
    policy_version_id    BIGINT REFERENCES policy_versions(id),
    evaluator_version_id BIGINT NOT NULL REFERENCES evaluator_versions(id),
    git_sha              TEXT NOT NULL,
    triggered_by         TEXT NOT NULL CHECK (triggered_by IN ('ci', 'schedule', 'manual', 'online_sample')),
    status               TEXT NOT NULL CHECK (status IN ('running', 'passed', 'failed', 'errored')),
    summary              JSONB,                  -- {cases, pass_rate, ci_low, ci_high, slices}
    started_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at          TIMESTAMPTZ
);

CREATE TABLE evaluation_results (
    run_id      UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    case_id     TEXT NOT NULL,
    message_id  UUID REFERENCES messages(id),    -- set when the case is a sampled production answer
    score       DOUBLE PRECISION,
    passed      BOOLEAN,
    details     JSONB,
    PRIMARY KEY (run_id, case_id)
);

-- ---------------------------------------------------------------------- audit ----
-- Append-only. Who did what, with which arguments, under which policy, approved by whom.

CREATE TABLE audit_events (
    id                     BIGSERIAL PRIMARY KEY,
    tenant_id              TEXT NOT NULL,
    actor_type             TEXT NOT NULL CHECK (actor_type IN ('user', 'system', 'agent', 'approver')),
    actor_id               TEXT NOT NULL,
    request_id             TEXT,
    conversation_id        UUID,
    job_id                 UUID,
    event_type             TEXT NOT NULL,        -- 'tool.proposed' | 'tool.approved' | 'tool.executed' |
                                                 -- 'policy.denied' | 'document.deleted' | 'prompt.released'
    tool_name              TEXT,
    tool_schema_version_id BIGINT REFERENCES tool_schema_versions(id),
    policy_version_id      BIGINT REFERENCES policy_versions(id),
    arguments_hash         TEXT,                 -- approval binds to these exact arguments
    payload                JSONB NOT NULL DEFAULT '{}',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX audit_tenant_time ON audit_events (tenant_id, created_at DESC);
CREATE INDEX audit_request ON audit_events (request_id);

-- Hard rule, enforced in application code and by this constraint: no UPDATE/DELETE on audit.
CREATE RULE audit_events_no_update AS ON UPDATE TO audit_events DO INSTEAD NOTHING;
CREATE RULE audit_events_no_delete AS ON DELETE TO audit_events DO INSTEAD NOTHING;
