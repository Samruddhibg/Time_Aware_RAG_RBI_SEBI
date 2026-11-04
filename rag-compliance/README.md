# Time-Aware RAG — RBI/SEBI Compliance Assistant

A production-grade Retrieval-Augmented Generation system for answering compliance questions about Indian financial regulations. The system is **time-aware**: it tracks document lifecycles (active → superseded → withdrawn), always returns answers from the currently valid circular, and refuses to answer when relevant information isn't in the corpus.

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐    Dense (HNSW)   ┌─────────────────────────────┐
│  FastAPI    │──────────────────►│  PostgreSQL + pgvector      │
│  /api/ask   │    Lexical (FTS)  │  ┌──────────┐ ┌──────────┐ │
│  (SSE)      │──────────────────►│  │documents │ │ chunks   │ │
└─────────────┘                   │  │  status  │ │embedding │ │
      │                           │  └──────────┘ └──────────┘ │
      │  RRF fusion               └─────────────────────────────┘
      │  top-k chunks
      ▼
┌─────────────┐
│  Ollama /   │   Prompt with
│  OpenAI LLM │◄──numbered context
│  (streaming)│   + refusal gate
└─────────────┘
      │
      ▼
 SSE token stream → React UI
```

**Key components:**
- **Backend**: FastAPI + asyncpg + pgvector
- **Embedder**: `BAAI/bge-small-en-v1.5` (fine-tunable via domain data)
- **LLM**: Llama 3.2:3b via Ollama (swappable for any OpenAI-compatible endpoint)
- **Retrieval**: Dense HNSW + full-text search, fused via Reciprocal Rank Fusion
- **Time-awareness**: `status` field on documents (`active`/`superseded`/`withdrawn`), updated automatically on each ingest via supersession chain detection
- **Frontend**: React + Vite, served via nginx in production

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker + Docker Compose | ≥ 24 | PostgreSQL, backend, frontend |
| Python | 3.11+ | Running scripts locally |
| Ollama | latest | Local LLM serving |
| Make | any | Convenience commands |

### 1. Clone and configure

```bash
git clone <repo-url> rag-compliance
cd rag-compliance
make env           # creates .env from .env.example
# Edit .env — at minimum verify DB credentials and LLM settings
```

### 2. Start the database

```bash
make db-start      # starts postgres container, waits for healthy
```

### 3. Install Python dependencies

```bash
make install-dev   # installs requirements.txt + beautifulsoup4 + lxml
```

### 4. Pull the LLM model

```bash
ollama serve &     # if Ollama isn't already running
make ollama-pull   # pulls llama3.2:3b (~2 GB)
```

### 5. Run the full pipeline

```bash
make pipeline DB_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
```

This runs all 9 stages automatically:

```
fetch → ingest → split → synthetic → negatives → finetune → rebuild → annotate → eval
```

Or run individual stages:

```bash
make fetch        # Download PDFs
make ingest       # Parse, chunk, embed, store
make train-all    # split + synthetic + negatives + finetune
make ingest-rebuild  # Re-embed after fine-tuning
make eval-all     # Run all evaluations
```

### 6. Start the API

```bash
make dev-backend  # hot-reload dev server at http://localhost:8000
# or
docker compose up -d db backend frontend
```

Open `http://localhost:8000/docs` for the interactive API docs, or `http://localhost:3000` for the React UI.

---

## Step-by-Step Pipeline Reference

### Stage 1 — Fetch Documents

Downloads the 20 seed regulatory PDFs (10 RBI + 10 SEBI) defined in the manifests.

```bash
make fetch
# or separately:
make fetch-rbi
make fetch-sebi
```

RBI and SEBI serve HTML pages, not direct PDF links. The scraper parses each page to find the embedded PDF. If automatic download fails (some documents require login or have non-standard page layouts), the script prints a **manual download guide** listing the exact URL and destination path for each missing file.

**Manual download**: Visit the URL in your browser → download the PDF → save to `data/raw/rbi/<document_id>.pdf` (or `sebi/`).

After placing all PDFs:
```
data/raw/rbi/
  RBI-2024-25-001.pdf   ← KYC Master Direction
  RBI-2024-25-002.pdf   ← IRAC Norms
  ... (10 total)
data/raw/sebi/
  SEBI-2024-001.pdf     ← MF Expense Ratio
  ... (10 total)
```

---

### Stage 2 — Ingest

Parses PDFs, chunks text, embeds with `BAAI/bge-small-en-v1.5`, and upserts into PostgreSQL.

```bash
make ingest
```

What happens per PDF:
1. **Hash check** — skips if unchanged (SHA-256)
2. **Parse** — pdfplumber → PyMuPDF fallback
3. **Metadata extraction** — regex for circular_no, issue_date, title, supersession references
4. **Chunking** — structure-aware (headings → sections → 250-word windows, 40-word overlap)
5. **Embedding** — batch encode with BGE asymmetric encoding
6. **DB upsert** — documents + chunks + HNSW index update
7. **Supersession linking** — detects "supersedes/withdraws/amends" and updates document statuses

Check what's in the database:
```bash
make db-stats
```

---

### Stage 3 — Train/Test Split

**Must run before generating any synthetic data.** Splits documents into an 80/20 train/test partition at the document level (no chunk from a test document ever appears in training).

```bash
make split DB_URL=<your-db-url>
```

Output: `data/splits/doc_splits.json`

---

### Stage 4 — Synthetic Training Data

Uses the LLM to generate realistic compliance questions for each training chunk. The prompt instructs the model to include at least one time-sensitive question (dates, limits, circular numbers).

```bash
make synthetic DB_URL=<your-db-url>
# or with more data:
make synthetic PAIRS_PER_CHUNK=3 MAX_CHUNKS=1000
```

Output: `data/training/synthetic_pairs.jsonl`  
Format: `{"anchor": "<question>", "positive": "<chunk_text>", "chunk_id": int, "document_id": str}`

Target: **≥ 500 pairs** for meaningful fine-tuning. With 20 docs × ~30 chunks × 2 pairs = ~1,200 pairs.

---

### Stage 5 — Hard Negative Mining

For each (query, positive_chunk) pair, finds a hard negative: a chunk that ranks in positions 10–40 in the similarity ranking but is from a different document. Hard negatives make contrastive training much more effective.

```bash
make negatives DB_URL=<your-db-url>
```

Output: `data/training/pairs_with_negatives.jsonl`  
Format: `{"anchor": ..., "positive": ..., "negative": "<hard_neg_text>", ...}`

---

### Stage 6 — Fine-Tune Embedding Model

Fine-tunes `BAAI/bge-small-en-v1.5` using `MultipleNegativesRankingLoss`. This adapts the model to regulatory language so it retrieves more relevant chunks.

```bash
make finetune
# or with custom settings:
make finetune EPOCHS=5 BATCH_SIZE=32
```

Output: `models/bge-small-ft/`

After fine-tuning, update your `.env`:
```bash
EMBED_MODEL=models/bge-small-ft
```

Then re-embed everything:
```bash
make ingest-rebuild
```

> **Note on GPU**: By default training uses CPU (`fp16=False` on CPU). To enable GPU training, set `fp16=True` in `training/finetune.py` and ensure CUDA is available. Training ~1,000 pairs takes ~10 minutes on CPU, ~2 minutes on GPU.

---

### Stage 7 — Rebuild Index

Re-embeds all corpus chunks using the fine-tuned model. Required whenever `EMBED_MODEL` changes.

```bash
make ingest-rebuild
```

---

### Stage 8 — Annotate Test Queries

Populates `relevant_chunk_ids` in `eval/test_queries.jsonl`. Required for the CI retrieval gate (`test_retrieval_quality.py`) to run.

```bash
# Fast: assume top-1 retrieved chunk is correct
make annotate ANNOTATE_MODE=auto

# Better: LLM judges each retrieved chunk
make annotate ANNOTATE_MODE=llm

# Most accurate: you confirm each chunk manually
make annotate ANNOTATE_MODE=manual
```

---

### Stage 9 — Evaluation

Three evaluation dimensions:

```bash
make eval-all API_URL=http://localhost:8000 DB_URL=<your-db-url>
```

| Eval | Script | What it measures |
|------|--------|-----------------|
| `eval-retrieval` | `eval/retrieval_eval.py` | Recall@k, MRR@k, nDCG@k, stale rate, latency |
| `eval-generation` | `eval/generation_eval.py` | Answer correctness, citation rate, refusal precision/recall |
| `eval-timetraps` | `eval/time_traps_eval.py` | Whether the system returns current vs stale regulatory values |

Results saved to `backend/eval/results/`.

#### Target metrics (post fine-tuning)

| Metric | Target |
|--------|--------|
| Recall@5 | ≥ 0.70 |
| MRR@10 | ≥ 0.60 |
| Stale answer rate | 0.00 |
| Answer correctness | ≥ 0.70 |
| Refusal F1 | ≥ 0.80 |
| Time-trap correct rate | ≥ 0.80 |
| Median latency | ≤ 500ms |

---

## Running Tests

```bash
# Unit + smoke tests (no DB needed)
make test

# Integration tests (requires live DB with data)
make test-integration DB_URL=<your-db-url>
```

The CI gate in `tests/test_retrieval_quality.py` checks Recall@5, MRR@10, and stale-answer rate against configurable thresholds. It automatically skips if `relevant_chunk_ids` aren't annotated yet.

---

## Production Deployment

### Docker Compose (recommended)

```bash
# Build images
make build

# Start everything (DB + backend + nginx frontend)
make up

# Start with Ollama LLM
make up-ollama

# Ingest documents into the running stack
docker compose run --rm --profile ingest ingest
```

The production stack:
- **backend**: 2 uvicorn workers, non-root user, health check on `/api/health`
- **frontend**: nginx serving the React build, proxies `/api/` to backend (avoids CORS)
- **db**: pgvector:pg16 with persistent volume, schema auto-applied on first start
- **ollama**: optional profile, pulls model on startup

### Environment checklist

Before going to production:
- [ ] Change `POSTGRES_PASSWORD` from the default
- [ ] Set a real `LLM_API_KEY` if using OpenAI/cloud LLM
- [ ] Set `CORS_ORIGINS` to your actual frontend domain
- [ ] Set `LOG_LEVEL=warning` to reduce log volume
- [ ] Mount `data/` and `models/` as read-only volumes (already set in compose)
- [ ] Set up a reverse proxy (nginx/traefik) in front of the API for TLS

---

## Project Structure

```
rag-compliance/
├── .env.example              ← All config variables documented
├── .env                      ← Your local config (gitignored)
├── Makefile                  ← All common commands
├── docker-compose.yml        ← Production services
│
├── data/
│   ├── raw/
│   │   ├── rbi/              ← RBI PDFs go here
│   │   ├── sebi/             ← SEBI PDFs go here
│   │   ├── rbi_manifest.json ← 10 RBI document definitions
│   │   └── sebi_manifest.json
│   ├── processed/            ← (unused, reserved for future use)
│   ├── splits/               ← doc_splits.json (train/test)
│   └── training/             ← synthetic_pairs.jsonl, pairs_with_negatives.jsonl
│
├── models/
│   └── bge-small-ft/         ← Fine-tuned embedding model (created by finetune stage)
│
├── frontend/
│   ├── src/                  ← React app source
│   ├── Dockerfile.frontend   ← Multi-stage: Vite build → nginx serve
│   └── package.json
│
└── backend/
    ├── requirements.txt
    ├── Dockerfile
    ├── pyproject.toml        ← pytest config
    │
    ├── app/                  ← FastAPI application
    │   ├── main.py           ← App factory, lifespan, CORS
    │   ├── prompts.py        ← System + user prompt templates
    │   ├── schemas.py        ← Pydantic request/response models
    │   ├── core/
    │   │   ├── config.py     ← Centralised settings (pydantic-settings)
    │   │   └── db.py         ← asyncpg pool + pgvector codec
    │   ├── routers/
    │   │   └── ask.py        ← POST /api/ask (SSE), POST /api/feedback
    │   └── services/
    │       ├── embedder.py   ← Async SentenceTransformer wrapper
    │       ├── retrieval.py  ← Dense + lexical + RRF fusion
    │       ├── llm.py        ← OpenAI-compatible streaming client
    │       └── rag.py        ← Orchestrator: retrieve → prompt → stream → log
    │
    ├── ingest/               ← Data ingestion pipeline
    │   ├── fetch_docs.py     ← Download PDFs (HTML scraper + direct)
    │   ├── parse_pdf.py      ← pdfplumber → PyMuPDF fallback
    │   ├── extract_metadata.py ← Regex metadata + supersession detection
    │   ├── chunk.py          ← Structure-aware sliding-window chunker
    │   ├── build_index.py    ← Orchestrates ingest for a directory of PDFs
    │   └── schema.sql        ← PostgreSQL schema (pgvector, HNSW, FTS)
    │
    ├── training/             ← Embedding model fine-tuning
    │   ├── split_by_doc.py   ← Document-disjoint train/test split
    │   ├── gen_synthetic.py  ← LLM-generated (query, passage) pairs
    │   ├── mine_negatives.py ← Hard negative mining
    │   └── finetune.py       ← SentenceTransformer fine-tuning
    │
    ├── eval/                 ← Evaluation
    │   ├── metrics.py        ← Recall@k, MRR, nDCG, stale rate, refusal P/R
    │   ├── retrieval_eval.py ← Retrieval quality evaluation
    │   ├── generation_eval.py ← Answer quality + refusal evaluation
    │   ├── time_traps_eval.py ← Time-awareness evaluation
    │   ├── test_queries.jsonl ← 20 annotated compliance questions
    │   ├── time_traps.jsonl  ← 15 temporal trap queries
    │   ├── unanswerable.jsonl ← 15 out-of-corpus queries
    │   └── baseline.json     ← Measured baseline metrics (auto-populated by eval)
    │
    ├── scripts/
    │   ├── run_pipeline.py   ← Master end-to-end orchestrator
    │   └── annotate_chunks.py ← Auto-annotate test queries with chunk IDs
    │
    └── tests/
        ├── test_api.py              ← API smoke tests (no DB required)
        └── test_retrieval_quality.py ← CI gate for retrieval regression
```

---

## API Reference

### `POST /api/ask`

Stream an answer to a compliance question.

**Request:**
```json
{
  "question": "What is the KYC cash transaction limit?",
  "historical": false
}
```

**Response** (Server-Sent Events):
```
data: {"type": "sources", "data": [{...chunk...}, ...]}
data: {"type": "token", "data": "The "}
data: {"type": "token", "data": "KYC "}
...
data: {"type": "done", "data": {"query_id": 42, "refused": false, "retrieval_ms": 45, "total_ms": 1230}}
data: [DONE]
```

Set `"historical": true` to include superseded documents (useful for "what was the rule in 2019?" questions).

### `POST /api/feedback`

Submit user rating for a completed query.

```json
{"query_id": 42, "rating": 1, "comment": "Accurate and well-cited"}
```

`rating`: `1` = thumbs up, `-1` = thumbs down.

### `GET /api/health`

Returns `{"status": "ok"}`. Used by Docker health checks.

---

## Troubleshooting

**PDFs not downloading automatically**  
Most RBI and SEBI pages require navigating an HTML page. Run `make fetch` — it will print a manual download guide for any files it couldn't retrieve. Download them via browser and place them in `data/raw/rbi/` or `data/raw/sebi/`.

**"No chunks retrieved" or very low similarity scores**  
The embedding model hasn't been fine-tuned yet. After completing the training pipeline, run `make ingest-rebuild` to re-embed with the domain-adapted model.

**LLM returning `[LLM error 404]`**  
Ollama isn't running or the model isn't pulled. Run `ollama serve` and `ollama pull llama3.2:3b`.

**CI gate (`test_retrieval_quality.py`) always skips**  
`relevant_chunk_ids` haven't been added to `eval/test_queries.jsonl`. Run `make annotate` then copy the output file over the original.

**Hard negative mining produces few triplets**  
With only 20 documents the corpus is small; positions 10–40 may all be from the same documents. Either add more documents or lower `--lo` (e.g., `--lo 5`).

**Fine-tuning OOM on CPU**  
Reduce batch size: `make finetune BATCH_SIZE=16`

**Docker build fails on embedding model download**  
This is non-fatal — the `|| echo "skipped"` in the Dockerfile handles it. The model will be downloaded on first startup instead.
