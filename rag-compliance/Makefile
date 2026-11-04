# =============================================================================
# Time-Aware RAG — RBI/SEBI Compliance Assistant
# Makefile — common commands for development, training, and production
#
# Usage: make <target>
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Configuration (override via make VAR=value) ───────────────────────────────
DB_URL          ?= postgresql://raguser:ragpass@localhost:5432/ragdb
API_URL         ?= http://localhost:8000
REGULATOR       ?= ALL
ANNOTATE_MODE   ?= auto          # auto | llm | manual
PAIRS_PER_CHUNK ?= 2
MAX_CHUNKS      ?= 500
EPOCHS          ?= 3
BATCH_SIZE      ?= 64

PYTHON          := python
BACKEND         := backend
DATA            := data

# ── Colours ───────────────────────────────────────────────────────────────────
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m

# =============================================================================
# HELP
# =============================================================================
.PHONY: help
help:   ## Show this help message
	@echo ""
	@echo "$(BOLD)Time-Aware RAG — RBI/SEBI Compliance Assistant$(RESET)"
	@echo ""
	@echo "$(CYAN)SETUP$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*SETUP' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*SETUP//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)DATA PIPELINE$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*DATA' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*DATA//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)TRAINING$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*TRAIN' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*TRAIN//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)EVALUATION$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*EVAL' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*EVAL//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)DOCKER$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*DOCKER' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*DOCKER//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)DEVELOPMENT$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*DEV' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*DEV//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)PIPELINE$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*PIPELINE' $(MAKEFILE_LIST) | \
	  sed 's/:.*## .*PIPELINE//' | \
	  awk '{printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "Override defaults: make <target> DB_URL=<url> API_URL=<url>"
	@echo ""

# =============================================================================
# SETUP
# =============================================================================
.PHONY: install
install: ## Install Python dependencies  [SETUP]
	pip install -r $(BACKEND)/requirements.txt
	@echo "$(GREEN)✓ Python dependencies installed$(RESET)"

.PHONY: install-dev
install-dev: install ## Install dev extras (beautifulsoup4, lxml)  [SETUP]
	pip install beautifulsoup4 lxml
	@echo "$(GREEN)✓ Dev extras installed$(RESET)"

.PHONY: env
env: ## Copy .env.example → .env (skips if .env exists)  [SETUP]
	@if [ -f .env ]; then \
	  echo "$(YELLOW).env already exists — skipping. Edit it manually.$(RESET)"; \
	else \
	  cp .env.example .env; \
	  echo "$(GREEN)✓ Created .env from .env.example — edit it before running further steps.$(RESET)"; \
	fi

.PHONY: db-start
db-start: ## Start only the PostgreSQL container  [SETUP]
	docker compose up -d db
	@echo "$(YELLOW)Waiting for DB to be healthy...$(RESET)"
	@until docker compose exec db pg_isready -U raguser -d ragdb; do sleep 2; done
	@echo "$(GREEN)✓ Database ready$(RESET)"

.PHONY: db-schema
db-schema: ## Apply schema.sql to the running database  [SETUP]
	docker compose exec -T db psql -U raguser -d ragdb \
	  < $(BACKEND)/ingest/schema.sql
	@echo "$(GREEN)✓ Schema applied$(RESET)"

.PHONY: ollama-pull
ollama-pull: ## Pull the default LLM model in Ollama  [SETUP]
	ollama pull llama3.2:3b
	@echo "$(GREEN)✓ LLM model ready$(RESET)"

# =============================================================================
# DATA PIPELINE
# =============================================================================
.PHONY: fetch
fetch: ## Download RBI + SEBI PDFs from manifests  [DATA]
	cd $(BACKEND) && $(PYTHON) -m ingest.fetch_docs \
	  --regulator $(REGULATOR) \
	  --out ../$(DATA)/raw

.PHONY: fetch-rbi
fetch-rbi: ## Download only RBI PDFs  [DATA]
	cd $(BACKEND) && $(PYTHON) -m ingest.fetch_docs \
	  --regulator RBI \
	  --manifest ../$(DATA)/raw/rbi_manifest.json \
	  --out ../$(DATA)/raw/rbi

.PHONY: fetch-sebi
fetch-sebi: ## Download only SEBI PDFs  [DATA]
	cd $(BACKEND) && $(PYTHON) -m ingest.fetch_docs \
	  --regulator SEBI \
	  --manifest ../$(DATA)/raw/sebi_manifest.json \
	  --out ../$(DATA)/raw/sebi

.PHONY: ingest
ingest: ## Parse PDFs → chunk → embed → upsert into PostgreSQL  [DATA]
	cd $(BACKEND) && $(PYTHON) -m ingest.build_index \
	  --dir ../$(DATA)/raw/rbi --regulator RBI
	cd $(BACKEND) && $(PYTHON) -m ingest.build_index \
	  --dir ../$(DATA)/raw/sebi --regulator SEBI

.PHONY: ingest-rebuild
ingest-rebuild: ## Force re-ingest all PDFs (re-embed everything)  [DATA]
	cd $(BACKEND) && $(PYTHON) -m ingest.build_index \
	  --dir ../$(DATA)/raw/rbi --regulator RBI --rebuild
	cd $(BACKEND) && $(PYTHON) -m ingest.build_index \
	  --dir ../$(DATA)/raw/sebi --regulator SEBI --rebuild

.PHONY: db-stats
db-stats: ## Show document and chunk counts in the database  [DATA]
	docker compose exec db psql -U raguser -d ragdb -c \
	  "SELECT regulator, status, COUNT(*) AS docs FROM documents GROUP BY regulator, status ORDER BY regulator, status;"
	docker compose exec db psql -U raguser -d ragdb -c \
	  "SELECT COUNT(*) AS total_chunks, MIN(token_count) AS min_tokens, \
	   ROUND(AVG(token_count)::numeric,1) AS avg_tokens, MAX(token_count) AS max_tokens \
	   FROM chunks;"

# =============================================================================
# TRAINING
# =============================================================================
.PHONY: split
split: ## Create document-disjoint train/test split  [TRAIN]
	cd $(BACKEND) && $(PYTHON) -m training.split_by_doc \
	  --db-url $(DB_URL) \
	  --out ../$(DATA)/splits

.PHONY: synthetic
synthetic: ## Generate synthetic (query, passage) training pairs via LLM  [TRAIN]
	cd $(BACKEND) && $(PYTHON) -m training.gen_synthetic \
	  --splits ../$(DATA)/splits/doc_splits.json \
	  --db-url $(DB_URL) \
	  --out ../$(DATA)/training/synthetic_pairs.jsonl \
	  --pairs-per-chunk $(PAIRS_PER_CHUNK) \
	  --max-chunks $(MAX_CHUNKS)

.PHONY: negatives
negatives: ## Mine hard negatives for each training pair  [TRAIN]
	cd $(BACKEND) && $(PYTHON) -m training.mine_negatives \
	  --pairs ../$(DATA)/training/synthetic_pairs.jsonl \
	  --db-url $(DB_URL) \
	  --out ../$(DATA)/training/pairs_with_negatives.jsonl

.PHONY: finetune
finetune: ## Fine-tune the embedding model on domain data  [TRAIN]
	cd $(BACKEND) && $(PYTHON) -m training.finetune \
	  --data ../$(DATA)/training/pairs_with_negatives.jsonl \
	  --base-model BAAI/bge-small-en-v1.5 \
	  --output ../models/bge-small-ft \
	  --epochs $(EPOCHS) \
	  --batch-size $(BATCH_SIZE)
	@echo ""
	@echo "$(YELLOW)Fine-tuning complete!$(RESET)"
	@echo "Next steps:"
	@echo "  1. Update EMBED_MODEL=models/bge-small-ft in your .env"
	@echo "  2. Run: make ingest-rebuild  (re-embed all chunks)"

.PHONY: train-all
train-all: split synthetic negatives finetune ## Run full training pipeline  [TRAIN]
	@echo "$(GREEN)✓ Full training pipeline complete$(RESET)"

# =============================================================================
# EVALUATION
# =============================================================================
.PHONY: annotate
annotate: ## Auto-annotate test queries with relevant chunk IDs  [EVAL]
	cd $(BACKEND) && $(PYTHON) -m scripts.annotate_chunks \
	  --queries eval/test_queries.jsonl \
	  --db-url $(DB_URL) \
	  --mode $(ANNOTATE_MODE) \
	  --out eval/test_queries_annotated.jsonl
	@echo "$(GREEN)✓ Annotated queries saved. Copy to test_queries.jsonl to enable CI gate.$(RESET)"

.PHONY: eval-retrieval
eval-retrieval: ## Run retrieval evaluation (Recall, MRR, nDCG, latency)  [EVAL]
	@mkdir -p $(BACKEND)/eval/results
	cd $(BACKEND) && $(PYTHON) -m eval.retrieval_eval \
	  --queries eval/test_queries.jsonl \
	  --db-url $(DB_URL) \
	  --out eval/results/retrieval_results.json
	@cat $(BACKEND)/eval/results/retrieval_results.json

.PHONY: eval-generation
eval-generation: ## Run generation evaluation (correctness, citations, refusal)  [EVAL]
	@mkdir -p $(BACKEND)/eval/results
	cd $(BACKEND) && $(PYTHON) -m eval.generation_eval \
	  --queries eval/test_queries.jsonl \
	  --unanswerable eval/unanswerable.jsonl \
	  --api-url $(API_URL) \
	  --out eval/results/generation_results.json

.PHONY: eval-timetraps
eval-timetraps: ## Run time-trap evaluation (stale answer detection)  [EVAL]
	@mkdir -p $(BACKEND)/eval/results
	cd $(BACKEND) && $(PYTHON) -m eval.time_traps_eval \
	  --traps eval/time_traps.jsonl \
	  --api-url $(API_URL) \
	  --out eval/results/time_traps_results.json

.PHONY: eval-all
eval-all: eval-retrieval eval-generation eval-timetraps ## Run all evaluations  [EVAL]
	@echo "$(GREEN)✓ All evaluations complete. Results in backend/eval/results/$(RESET)"

.PHONY: test
test: ## Run unit and smoke tests (no DB required)  [EVAL]
	cd $(BACKEND) && pytest tests/test_api.py -v

.PHONY: test-integration
test-integration: ## Run integration tests (requires live DB)  [EVAL]
	cd $(BACKEND) && pytest tests/ -v -m "not integration" 
	cd $(BACKEND) && pytest tests/test_retrieval_quality.py -v

# =============================================================================
# DOCKER
# =============================================================================
.PHONY: up
up: ## Start all services (DB + backend + frontend)  [DOCKER]
	docker compose up -d db backend frontend
	@echo "$(GREEN)✓ Services started$(RESET)"
	@echo "  API    : http://localhost:8000/docs"
	@echo "  UI     : http://localhost:3000"

.PHONY: up-ollama
up-ollama: ## Start all services including Ollama LLM  [DOCKER]
	docker compose --profile ollama up -d
	@echo "$(YELLOW)Waiting for Ollama to download model (may take a few minutes)...$(RESET)"

.PHONY: up-ingest
up-ingest: ## Run the ingest worker container then stop it  [DOCKER]
	docker compose --profile ingest run --rm ingest

.PHONY: down
down: ## Stop all running containers  [DOCKER]
	docker compose down

.PHONY: down-volumes
down-volumes: ## Stop containers AND delete all data volumes  [DOCKER]
	@echo "$(YELLOW)WARNING: This will delete the database. Press Ctrl-C to cancel.$(RESET)"
	@sleep 5
	docker compose down -v

.PHONY: build
build: ## Build all Docker images  [DOCKER]
	docker compose build

.PHONY: logs
logs: ## Tail logs from all containers  [DOCKER]
	docker compose logs -f

.PHONY: logs-backend
logs-backend: ## Tail backend logs only  [DOCKER]
	docker compose logs -f backend

# =============================================================================
# DEVELOPMENT
# =============================================================================
.PHONY: dev-backend
dev-backend: ## Start backend API with hot-reload (not Docker)  [DEV]
	cd $(BACKEND) && uvicorn app.main:app \
	  --host 0.0.0.0 --port 8000 --reload --log-level debug

.PHONY: lint
lint: ## Run ruff linter on backend code  [DEV]
	cd $(BACKEND) && ruff check . || true

.PHONY: format
format: ## Format backend code with ruff  [DEV]
	cd $(BACKEND) && ruff format . || true

.PHONY: check-db
check-db: ## Verify database connection  [DEV]
	cd $(BACKEND) && $(PYTHON) -c "\
import asyncio, asyncpg; \
async def check(): \
  pool = await asyncpg.create_pool('$(DB_URL)'); \
  v = await pool.fetchval('SELECT version()'); \
  print('DB OK:', v[:60]); \
  await pool.close(); \
asyncio.run(check())"

.PHONY: check-llm
check-llm: ## Verify LLM endpoint is reachable  [DEV]
	curl -sf http://localhost:11434/api/tags | python -m json.tool | head -20 || \
	  echo "Ollama not running. Start it with: ollama serve"

# =============================================================================
# FULL PIPELINE
# =============================================================================
.PHONY: pipeline
pipeline: ## Run the complete end-to-end pipeline  [PIPELINE]
	cd $(BACKEND) && $(PYTHON) -m scripts.run_pipeline \
	  --db-url $(DB_URL) \
	  --api-url $(API_URL)

.PHONY: pipeline-from-ingest
pipeline-from-ingest: ## Run pipeline starting from ingest (skip fetch)  [PIPELINE]
	cd $(BACKEND) && $(PYTHON) -m scripts.run_pipeline \
	  --db-url $(DB_URL) \
	  --api-url $(API_URL) \
	  --start-from ingest

.PHONY: pipeline-eval-only
pipeline-eval-only: ## Run only the evaluation stages  [PIPELINE]
	cd $(BACKEND) && $(PYTHON) -m scripts.run_pipeline \
	  --db-url $(DB_URL) \
	  --api-url $(API_URL) \
	  --only eval

.PHONY: pipeline-dry-run
pipeline-dry-run: ## Show what the pipeline would do without running it  [PIPELINE]
	cd $(BACKEND) && $(PYTHON) -m scripts.run_pipeline \
	  --db-url $(DB_URL) \
	  --dry-run
