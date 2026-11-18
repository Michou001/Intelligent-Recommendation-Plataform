# Intelligent Recommendation Platform
## Software Engineering II — Milestone 3: Testing, Validation, and Deployment of AI Systems

> Milestone 3 additions are documented in [`docs/MILESTONE3_IMPLEMENTATION.md`](docs/MILESTONE3_IMPLEMENTATION.md).
>
> **Entrega final** (aplicación web en español, cuentas, arranque en frío, tendencias en vivo, métricas y demo en varias máquinas): [`docs/ENTREGA_FINAL.md`](docs/ENTREGA_FINAL.md).

**Universidad Autónoma del Estado de México (UAEMex)**  
*Faculty of Engineering — Degree in Artificial Intelligence Engineering*

### Team Members
- Betzy Perla Ramirez Solis
- Fabiola Laureano Martinez
- Michel Gonzalez Becerril
- Angel Gabriel Serrano Vazquez

**Responsible for the Subject:** M. Jessica Emma Cortez Garcia

---

## 📌 Architecture Overview

This platform implements a **Layered Architecture** following SOLID principles and GoF / ML design patterns, ensuring that the AI inference engine evolves independently from the business logic and persistence layers:

1. **Presentation Layer (`src/presentation/`)**: Pydantic v2 schemas validating request/response contracts and enforcing *Fail-Fast* input validation (HTTP 400), plus a dependency-free web interface (`static/index.html`) that consumes the public endpoints and holds no business or AI logic.
2. **Application Layer (`src/application/`)**: Orchestration via `RecommendationController`, real-time event ingestion via `InteractionIngestionService`, and REST API endpoints built with FastAPI.
3. **AI & Recommendation Engine Layer (`src/ai/`)**:
   - **Strategy Pattern (`src/ai/strategies/`)**: Interchangeable recommendation models implementing `IRecommendationStrategy.predict(user_features)`:
     - `CollaborativeFilteringStrategy` (implicit ALS)
     - `ContentBasedStrategy` (Sentence Transformers + Scikit-Learn cosine similarity)
     - `DeepLearningStrategy` (Neural Collaborative Filtering in PyTorch)
     - `PopularityStrategy` (Cold-start fallback)
   - **Factory Pattern (`src/ai/model_factory.py`)**: Registry-based model instantiation (`@register_strategy`).
   - **Decorator Pattern (`src/ai/decorators/`)**: Dynamic operational enhancements (`CachingDecorator`, `TimingDecorator`, `LoggingDecorator`).
4. **Data Access & Feature Store Layer (`src/data/`)**:
   - `FeatureStoreManager` serving precomputed user and item vectors in Apache Parquet.
   - Offline 4-stage data pipeline: `ingest` → `clean` → `transform` → `materialize`.
5. **Persistence & Storage Layer (`src/persistence/`)**: Segregated repository interfaces (`IUserProfileRepository`, `ICatalogRepository`, `IInteractionHistoryRepository`, `IEmbeddingRepository`).
6. **Messaging & Asynchronous Logging (`src/messaging/`)**: `InteractionEventQueue` and `FeedbackHistoryService` background worker consuming interaction events outside the critical request path.

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev,als,deep,content,cache,broker,nosql]"
```

### 2. Running Automated Tests

Run the complete test suite (300 unit, integration, failure-injection and system tests):

```bash
python -m pytest
```

Static analysis and strict architecture import verification:

```bash
python -m ruff check .
python -m mypy
```

### 3. Measuring Latency Benchmark (NFR-01 Target: p95 < 200 ms)

```bash
# In-process controller benchmark with caching:
python benchmarks/latency_p95.py

# Worst-case benchmark (no cache, forced inference on every request):
python benchmarks/latency_p95.py --no-cache
```

### 4. Running the REST API

```bash
# Set PYTHONPATH to src if running without editable pip install:
$env:PYTHONPATH="src"
python -m uvicorn application.api:app --reload --port 8000
```

Once running, the following are available:
- Web interface: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — query recommendations for any user, inspect the served model and latency, and register interactions
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Milestone 3 — validation, profiling and deployment

```bash
python benchmarks/evaluate_models.py          # temporal holdout + Precision/Recall/F1/NDCG per strategy
python benchmarks/memory_profile.py           # tracemalloc + RSS per component
python benchmarks/load_test.py --workers 4    # throughput and p95 under concurrent load
docker compose up --build                     # API container at http://127.0.0.1:8000/ui/ (API_PORT=8080 if 8000 is taken)
docker compose --profile full up --build      # + Redis, RabbitMQ, MongoDB and the FR-06 worker (API on port 8001)
kubectl apply -k k8s/                         # Kubernetes manifests
```

Reports are written to `reports/`. Errors returned to clients are in Spanish; handled validation and
domain errors, and every degradation, are recorded in the error journal `artifacts/logs/errors.jsonl` (see
[`docs/ERROR_HANDLING_ALGORITHM.md`](docs/ERROR_HANDLING_ALGORITHM.md)). CI runs on every push to `main`,
on pull requests and on demand: `.github/workflows/ci.yml`.

---

## 📂 Repository Structure

```text
recommendation-platform/
├── src/
│   ├── presentation/       # Layer 1: Request/response schemas (Pydantic v2)
│   │   └── static/         #          Web interface served at /ui
│   ├── application/        # Layer 2: Controller, Ingestion service, FastAPI routes
│   ├── ai/                 # Layer 3: Interfaces, ModelFactory, Strategies & Decorators
│   ├── data/               # Layer 4: Feature Store Manager & 4-stage offline pipeline
│   ├── persistence/        # Layer 5: Segregated repository interfaces and adapters
│   ├── messaging/          # Asynchronous event queue & FeedbackHistoryService worker
│   └── config/             # Typed settings configuration
├── tests/                  # Mirrors src/ package by package + failure injection tests
├── benchmarks/             # Latency p95, model evaluation, memory profile, load test
├── reports/                # Milestone 3 measurement reports (JSON / Markdown)
├── docs/                   # Guides, error handling, and user/developer manuals
│   ├── MILESTONE3_IMPLEMENTATION.md
│   ├── ERROR_HANDLING_ALGORITHM.md
│   ├── User_Manual_v1.pdf
│   └── Developer_Manual_v2.pdf
├── k8s/                    # Kubernetes manifests (kubectl apply -k k8s/)
├── Dockerfile              # API image; docker-compose.yml for local deployment
├── datasets/               # MovieLens ml-latest-small source data
├── artifacts/              # Parquet feature store, SQLite DB, model weights & reports
├── IMPLEMENTATION_NOTES.md # Component/file/status table, measured p95, deviations
└── pyproject.toml          # Tooling configuration (ruff, mypy, pytest)
```
