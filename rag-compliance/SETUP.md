# Developer Setup Guide — RAG-Compliance

> **Time-Aware RAG for RBI/SEBI Circulars** — Step-by-step local dev setup.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker Desktop | 24+ | https://www.docker.com/products/docker-desktop |
| Ollama | latest | https://ollama.com |
| Python | 3.11+ | https://python.org |
| Node.js | 20+ | https://nodejs.org |
| Git | any | https://git-scm.com |

---

## Step 1 — Clone and configure

```powershell
# Windows PowerShell
git clone <repo-url>
cd rag-compliance

# Copy env template
Copy-Item .env.example .env
# Edit .env if needed (defaults work for local dev with Ollama)
```

---

## Step 2 — Start the LLM (Ollama)

```powershell
# Download and run llama3.2:3b (small, fast, CPU-friendly)
ollama pull llama3.2:3b
ollama serve
```

Verify it's running: http://localhost:11434

---

## Step 3 — Start PostgreSQL + Backend + Frontend (Docker)

```powershell
docker compose up --build
```

This starts:
- **PostgreSQL + pgvector** on port 5432
- **FastAPI backend** on port 8000
- **React frontend (Vite)** on port 5173

Wait for `backend` to log `Startup complete` before proceeding.

### Verify services:
```powershell
# Backend health
curl http://localhost:8000/api/health

# API docs
start http://localhost:8000/docs

# Frontend
start http://localhost:5173
```

---

## Step 4 — Ingest regulatory documents

### Option A: Quick start with sample manifests (recommended)

```powershell
# From the rag-compliance/ directory
cd backend

# Ingest RBI circulars from sample manifest
python -m ingest.fetch_docs --regulator RBI --manifest ../data/raw/rbi_manifest.json --out ../data/raw/rbi

# Ingest SEBI circulars from sample manifest
python -m ingest.fetch_docs --regulator SEBI --manifest ../data/raw/sebi_manifest.json --out ../data/raw/sebi

# Build the vector index
python -m ingest.build_index --dir ../data/raw/rbi --regulator RBI
python -m ingest.build_index --dir ../data/raw/sebi --regulator SEBI
```

### Option B: Add your own PDFs

```powershell
# Place PDFs in data/raw/rbi/ or data/raw/sebi/ then:
python -m ingest.build_index --dir ../data/raw/rbi --regulator RBI
python -m ingest.build_index --dir ../data/raw/sebi --regulator SEBI
```

---

## Step 5 — Verify retrieval is working

Open http://localhost:5173 and ask:
- *"What is the KYC cash transaction limit?"*
- *"What are the SEBI LODR disclosure requirements?"*

You should see a streaming answer with numbered citations and source cards.

---

## Step 6 — Run the test suite

```powershell
cd backend
pip install -r requirements.txt

# Smoke tests (no DB required)
pytest tests/test_api.py -v

# Quality gate (requires running DB + ingested docs)
pytest tests/test_retrieval_quality.py -v
```

---

## Training Pipeline (Optional — after ingesting documents)

```powershell
cd backend

# 1. Create document-disjoint train/test split (MUST be first)
python -m training.split_by_doc --db-url postgresql://raguser:ragpass@localhost:5432/ragdb --out ../data/splits

# 2. Generate synthetic query-passage training pairs
python -m training.gen_synthetic \
  --splits ../data/splits/doc_splits.json \
  --db-url postgresql://raguser:ragpass@localhost:5432/ragdb \
  --out ../data/training/synthetic_pairs.jsonl

# 3. Mine hard negatives
python -m training.mine_negatives \
  --pairs ../data/training/synthetic_pairs.jsonl \
  --db-url postgresql://raguser:ragpass@localhost:5432/ragdb \
  --out ../data/training/pairs_with_negatives.jsonl

# 4. Fine-tune the embedder
python -m training.finetune \
  --data ../data/training/pairs_with_negatives.jsonl \
  --base-model BAAI/bge-small-en-v1.5 \
  --output ../models/bge-small-ft \
  --epochs 3 --batch-size 32

# 5. Update .env: EMBED_MODEL=../models/bge-small-ft
# 6. Re-embed all chunks with the fine-tuned model
python -m ingest.build_index --dir ../data/raw --regulator RBI --rebuild
python -m ingest.build_index --dir ../data/raw --regulator SEBI --rebuild
```

---

## Evaluation

```powershell
cd backend

# Retrieval metrics (Recall@k, MRR, nDCG)
python -m eval.retrieval_eval \
  --queries eval/test_queries.jsonl \
  --db-url postgresql://raguser:ragpass@localhost:5432/ragdb \
  --out eval/retrieval_results.json

# Generation metrics (faithfulness, citation rate, refusal)
python -m eval.generation_eval \
  --queries eval/test_queries.jsonl \
  --unanswerable eval/unanswerable.jsonl \
  --api-url http://localhost:8000 \
  --out eval/generation_results.json
```

---

## Load Testing

```powershell
cd backend
# Install locust: pip install locust
locust -f bench/load_test.py --headless -u 10 -r 2 --run-time 60s --host http://localhost:8000
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `asyncpg.exceptions.ConnectionDoesNotExistError` | Ensure Docker is running and DB is healthy |
| `OSError: no file or directory: 'BAAI/bge-small-en-v1.5'` | Run `pip install sentence-transformers` and let it download the model |
| LLM responses empty | Check `ollama serve` is running and model is pulled |
| Frontend shows "Connection Error" | Backend not running; check `docker compose logs backend` |
| `recompute_status()` function not found | Re-run schema.sql: `docker exec -i <db-container> psql -U raguser -d ragdb < backend/ingest/schema.sql` |

---

## Architecture Recap

```
User → React (Vite) → FastAPI → retrieve() → PostgreSQL + pgvector
                              ↓
                         build_context()
                              ↓
                        stream_completion() → Ollama / vLLM
                              ↓
                         SSE stream → React AnswerPanel
```

- **Time-aware filter**: `WHERE d.status = 'active'` — superseded circulars never cited by default
- **Hybrid retrieval**: HNSW dense search ⊕ PostgreSQL FTS, fused via Reciprocal Rank Fusion
- **Refusal path**: answers refused when `best_sim < MIN_SIM_THRESHOLD` (default 0.35)
