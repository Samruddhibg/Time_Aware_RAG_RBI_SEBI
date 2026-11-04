CREATE EXTENSION IF NOT EXISTS vector;

-- 1. DOCUMENTS ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    regulator        TEXT NOT NULL CHECK (regulator IN ('RBI','SEBI')),
    circular_no      TEXT,
    title            TEXT,
    doc_type         TEXT CHECK (doc_type IN ('circular','master_direction','master_circular','notification')),
    issue_date       DATE,
    effective_date   DATE,
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active','superseded','withdrawn')),
    superseded_by    TEXT REFERENCES documents(document_id),
    source_url       TEXT,
    content_hash     TEXT,
    parse_ok         BOOLEAN DEFAULT TRUE,
    meta_confidence  REAL,
    meta_verified    BOOLEAN DEFAULT FALSE,
    retrieved_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_date   ON documents (issue_date);

-- 2. SUPERSESSIONS -----------------------------------------------------
CREATE TABLE IF NOT EXISTS supersessions (
    id            BIGSERIAL PRIMARY KEY,
    new_doc_id    TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    old_doc_id    TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    relation      TEXT NOT NULL CHECK (relation IN ('supersedes','withdraws','amends')),
    evidence_text TEXT,
    confidence    REAL,
    verified      BOOLEAN DEFAULT FALSE,
    UNIQUE (new_doc_id, old_doc_id, relation),
    CHECK (new_doc_id <> old_doc_id)
);
CREATE INDEX IF NOT EXISTS idx_supers_old ON supersessions (old_doc_id);

-- 3. CHUNKS ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,
    section      TEXT,
    text         TEXT NOT NULL,
    token_count  INT,
    embed_model  TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv  ON chunks USING gin (tsv);

-- 4. LOGGING -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_log (
    id            BIGSERIAL PRIMARY KEY,
    question      TEXT NOT NULL,
    historical    BOOLEAN NOT NULL DEFAULT FALSE,
    chunk_ids     BIGINT[],
    best_sim      REAL,
    refused       BOOLEAN,
    retrieval_ms  INT,
    ttft_ms       INT,
    total_ms      INT,
    answer        TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    query_id    BIGINT NOT NULL REFERENCES query_log(id) ON DELETE CASCADE,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- 5. STATUS RECOMPUTE --------------------------------------------------
CREATE OR REPLACE FUNCTION recompute_status() RETURNS void AS $$
BEGIN
    UPDATE documents SET status = 'active', superseded_by = NULL;

    UPDATE documents d
    SET status = CASE s.relation WHEN 'withdraws' THEN 'withdrawn' ELSE 'superseded' END,
        superseded_by = s.new_doc_id
    FROM (
        SELECT DISTINCT ON (old_doc_id) old_doc_id, new_doc_id, relation
        FROM supersessions s
        JOIN documents n ON n.document_id = s.new_doc_id
        WHERE s.relation IN ('supersedes','withdraws')
        ORDER BY old_doc_id, n.issue_date DESC
    ) s
    WHERE d.document_id = s.old_doc_id;
END;
$$ LANGUAGE plpgsql;
