# Intelligent Recommendation Platform
## Software Engineering II — Milestone 2: Implementation of AI Software Components

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

1. **Presentation Layer (`src/presentation/`)**: Pydantic v2 schemas validating request/response contracts and enforcing *Fail-Fast* input validation (HTTP 400).
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

Run the complete test suite (258 unit and integration tests):

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

Interactive API documentation will be available at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📂 Repository Structure

```text
recommendation-platform/
├── src/
│   ├── presentation/       # Layer 1: Request/response schemas (Pydantic v2)
│   ├── application/        # Layer 2: Controller, Ingestion service, FastAPI routes
│   ├── ai/                 # Layer 3: Interfaces, ModelFactory, Strategies & Decorators
│   ├── data/               # Layer 4: Feature Store Manager & 4-stage offline pipeline
│   ├── persistence/        # Layer 5: Segregated repository interfaces and adapters
│   ├── messaging/          # Asynchronous event queue & FeedbackHistoryService worker
│   └── config/             # Typed settings configuration
├── tests/                  # Mirrors src/ package by package + failure injection tests
├── benchmarks/             # Latency p95 benchmark replay script
├── datasets/               # MovieLens ml-latest-small source data
├── artifacts/              # Parquet feature store, SQLite DB, model weights & reports
└── pyproject.toml          # Tooling configuration (ruff, mypy, pytest)
```
