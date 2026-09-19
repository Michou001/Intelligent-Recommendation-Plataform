# Implementation Notes — Milestone 2

Companion to *Intelligent Recommendation Platform — Software Architecture Proposal*.
Everything below is reported against the document: Section 9 is the normative
specification, and this file records what was built, what was measured, where
the implementation departed from the text, and what was left out.

- **Language / runtime:** Python 3.11 target (`requires-python = ">=3.11"`), developed and verified on CPython 3.13.14.
- **Source:** 39 modules, 5 584 lines. **Tests:** 31 modules, 3 286 lines, **258 tests, all passing**.
- **Static analysis:** `ruff check .` clean, `black --check .` clean, `mypy` clean (38 source files).
- **Dataset:** MovieLens `ml-latest-small` — 100 836 ratings ingested, 100 789 after cleaning, 610 users, 9 690 items; interaction matrix 609 × 7 343 (61 687 non-zeros, 1.38 % density).

---

## (a) Component ↔ file ↔ status

### Table 11 of the document (code-to-architecture traceability)

| Architecture component | Layer | Code module | Requirement | Status |
|---|---|---|---|---|
| `RecommendationController` | Application | `src/application/recommendation_controller.py` | FR-01, FR-04 | **Implemented** |
| `InteractionIngestionService` | Application | `src/application/interaction_ingestion_service.py` | FR-02 | **Implemented** |
| `IRecommendationStrategy` | AI engine | `src/ai/interfaces.py` | FR-04 | **Implemented** |
| `ModelFactory` (with registry) | AI engine | `src/ai/model_factory.py` | FR-04 | **Implemented** |
| `CollaborativeFilteringStrategy` | AI engine | `src/ai/strategies/collaborative_filtering_strategy.py` | FR-04 | **Implemented** (ALS via `implicit`) |
| `ContentBasedStrategy` | AI engine | `src/ai/strategies/content_based_strategy.py` | FR-04 | **Partial** — cosine similarity complete; embeddings default to TF-IDF+SVD, not sentence-transformers (deviation 8) |
| `DeepLearningStrategy` | AI engine | `src/ai/strategies/deep_learning_strategy.py` | FR-04 | **Implemented** (PyTorch NCF, trained: loss 0.4129 → 0.3486 over 3 epochs) |
| `PopularityStrategy` | AI engine | `src/ai/strategies/popularity_strategy.py` | FR-05 | **Implemented** |
| `CachingDecorator` | AI engine | `src/ai/decorators/caching_decorator.py` | NFR-01, NFR-04 | **Implemented** (Redis backend written, in-process LRU used) |
| `TimingDecorator` | AI engine | `src/ai/decorators/timing_decorator.py` | NFR-01, NFR-04 | **Implemented** |
| `LoggingDecorator` | AI engine | `src/ai/decorators/logging_decorator.py` | NFR-01, NFR-04 | **Implemented** (structlog JSON) |
| `FeatureStoreManager` | Data access | `src/data/feature_store_manager.py` | FR-03 | **Implemented** (read-only, Parquet) |
| Offline feature pipeline | Data access | `src/data/pipeline/` | FR-03 | **Implemented** (4 stages, CLI per stage) |
| `InteractionEventQueue` | Messaging | `src/messaging/interaction_event_queue.py` | FR-02, FR-06 | **Implemented** (RabbitMQ backend written, in-process used) |
| `FeedbackHistoryService` | Messaging | `src/messaging/feedback_history_service.py` | FR-06 | **Implemented** (idempotent, retries) |
| `IUserProfileRepository`, `ICatalogRepository`, `IInteractionHistoryRepository`, `IEmbeddingRepository` | Persistence | `src/persistence/interfaces.py` + adapters | FR-01, FR-06, NFR-03 | **Implemented** |

### Concrete adapters named in Section 3.2

| Adapter | Code module | Backend used | Status |
|---|---|---|---|
| `SqlUserProfileRepository` | `src/persistence/sql_user_profile_repository.py` | SQLite (PostgreSQL by config) | **Implemented** |
| `SqlCatalogRepository` | `src/persistence/sql_catalog_repository.py` | SQLite (PostgreSQL by config) | **Implemented** |
| `NoSqlInteractionHistoryRepository` | `src/persistence/nosql_interaction_history_repository.py` | local JSONL (MongoDB by config) | **Implemented** |
| `VectorEmbeddingRepository` | `src/persistence/vector_embedding_repository.py` | compressed NumPy archive | **Implemented** |

### Table 8 — pipeline stages

| Stage | Module | Status | Result on `ml-latest-small` |
|---|---|---|---|
| 1. Ingestion | `src/data/pipeline/ingest.py` | **Implemented** | 100 836 rows in 3 chunks, 0 rejected; schema checked per chunk |
| 2. Cleaning | `src/data/pipeline/clean.py` | **Implemented** | −47 orphan ratings, −34 genre-less items, −18 unrated items |
| 3. Transformation | `src/data/pipeline/transform.py` | **Implemented** | ALS 64 factors, 64-dim content embeddings, popularity + 19 category rankings |
| 4. Materialization | `src/data/pipeline/materialize.py` | **Implemented** | `artifacts/feature_store/v1/`, 609 user / 7 343 item vectors, version tag |

### Table 10 — error handling

| Exception | Code | Response | Behaviour | Status |
|---|---|---|---|---|
| `ValidationError` (Pydantic) | `validation_error` | **400** | Rejected before the AI layer | **Implemented** (handler overrides FastAPI's default 422) |
| `UserNotFoundError` | `user_not_found` | **404** | No inference attempted | **Implemented** |
| `ColdStartError` | `cold_start` | **200** | `PopularityStrategy` fallback | **Implemented** |
| `FeatureStoreUnavailableError` | `feature_store_unavailable` | **200 degraded** | Popularity fallback, logged at error level | **Implemented** |
| `ModelUnavailableError` | `model_unavailable` | **200 degraded** | Popularity fallback | **Implemented** |
| `EventPublishError` | `event_publish_failed` | no effect on response | Written to local retry buffer | **Implemented** |

### Additional modules (not named in the Section 9.2 tree)

| File | Why it exists | Status |
|---|---|---|
| `src/errors.py` | Table 10's exceptions have no assigned module; they are raised in four layers (deviation 5) | **Implemented** |
| `src/persistence/__main__.py` | Seeds profiles/catalog/embeddings so `UserNotFoundError` is answerable (deviation 10) | **Implemented** |
| `src/data/pipeline/__main__.py` | Runs the four stages in order | **Implemented** |
| `benchmarks/latency_p95.py` | Required by Section 9.7 | **Implemented** |
| `tests/test_architecture_imports.py` | The import rule "verified by a test" of Section 9.2 | **Implemented** |
| `tests/test_failure_injection.py` | The failure-injection cases of Section 9.9 | **Implemented** |

---

## (b) Measured p95 latency

Target: **NFR-01 — 200 ms at the 95th percentile.**
Measured with `benchmarks/latency_p95.py`, 1 000 timed requests after 25 warm-up
requests, users drawn at random from the materialised store, `k = 10`, dataset
`ml-latest-small`, on the development machine (Windows 11, CPython 3.13.14, CPU only).

| Mode | Caching | p50 | **p95** | p99 | max | Verdict |
|---|---|---|---|---|---|---|
| In-process controller | on | 0.911 ms | **1.710 ms** | 2.682 ms | 3.622 ms | **PASS** |
| In-process controller | off (every request infers) | 1.056 ms | **1.579 ms** | 2.057 ms | 2.966 ms | **PASS** |
| **HTTP end-to-end (uvicorn)** | on | 2.578 ms | **3.388 ms** | 3.736 ms | 5.383 ms | **PASS** |

Per strategy, cache disabled, 500 requests each:

| Strategy | p50 | **p95** | p99 | max |
|---|---|---|---|---|
| `popularity` | 0.790 ms | **1.302 ms** | 1.596 ms | 3.341 ms |
| `als` | 1.067 ms | **1.612 ms** | 2.033 ms | 3.023 ms |
| `content` | 1.081 ms | **1.689 ms** | 1.934 ms | 3.040 ms |
| `ncf` (PyTorch) | 4.118 ms | **4.935 ms** | 5.484 ms | 9.065 ms |

**Reading of the result.** The headline figure for NFR-01 is the client-observed
**3.388 ms at p95**, roughly 59× below the 200 ms budget. Three caveats keep
that number honest:

1. It is single-node, single-client, with no concurrency. It measures the
   inference cycle the architecture is responsible for, not a loaded system.
2. `ml-latest-small` is 7 343 items. Scoring is `O(items)` per request, so the
   figure grows roughly linearly with the catalog: `ml-25m` (~59 000 items)
   would be about 8× the scoring cost and still inside the budget, but that
   extrapolation has not been measured.
3. The margin is large enough that caching is not what buys compliance — the
   no-cache p95 is also ~1.6 ms. This is what justifies Section 9.7's decision
   to leave an approximate-nearest-neighbour index out of scope: the vectorised
   `argpartition` selection already meets the target with no added
   infrastructure.

Reproduce with:

```bash
python benchmarks/latency_p95.py --requests 1000 --json
```

---

## (c) Deviations from the document, and why

1. **Runtime is CPython 3.13, not 3.11.** Section 9 specifies Python 3.11;
   `pyproject.toml` declares `requires-python = ">=3.11"`, but the only
   interpreter available was 3.13.14, so that is what the suite and the
   benchmark ran on. Related: `[tool.mypy]` carries no `python_version` pin,
   because the installed numpy ships type stubs using 3.12+ syntax, and pinning
   the check to 3.11 aborts before it reaches this project's code.

2. **`predict(user_features)` became `predict(user_features, k=10)`.** Table 4
   writes the contract without `k`, but FR-04 ("the K items with the highest
   relevance score") and the endpoint of Section 9.6 (`?k=10`) make the ranking
   size a per-request value. It is a defaulted argument, so the call site still
   reads as the document writes it.

3. **`IRecommendationStrategy` gained a `build(settings)` classmethod.** This is
   the construction hook `ModelFactory` calls after resolving a class through
   the registry. Without it the factory would need to know how each strategy
   loads its artifacts — which is exactly the per-strategy branch Section 9.5
   forbids. The default implementation needs no artifacts.

4. **`IDataRepository` is not declared.** Section 3.2 describes it as the base
   contract that was *split*, and Table 6 records that split as the Milestone 1
   correction. Re-adding a live `save`/`find_by_id` ancestor would restore the
   universal repository the correction removed and force every client to
   inherit operations it does not use. The four segregated interfaces are
   independent ABCs, and a test asserts no common ancestor was reintroduced.

5. **Table 10's exceptions live in `src/errors.py`, at the source root.** The
   tree of Section 9.2 assigns them no module. They are raised in four
   different layers (persistence, data, AI, messaging) and translated by one
   handler in the application layer, so putting them inside any single layer
   would force the others to import from it and break the dependency direction.

6. **`FeatureStoreManager` and `InteractionEventQueue` are abstract base
   classes, with their concrete backends in the same module.** The document
   names them as components and names only `ai/interfaces.py` and
   `persistence/interfaces.py` as interface modules. Declaring them as ABCs
   lets the application layer depend on the contract — as the dependency rule
   requires — without inventing modules outside the Section 9.2 tree.

7. **Infrastructure backends fall back to in-process equivalents.** Table 7
   specifies Redis, RabbitMQ, PostgreSQL and MongoDB; none was available. Each
   is implemented behind the same interface, alongside a fallback selected by
   configuration when no endpoint is set: in-process LRU cache, in-process
   queue, SQLite, and a local JSONL document store. Pointing `RECO_REDIS_URL`,
   `RECO_RABBITMQ_URL`, `RECO_SQL_URL` or `RECO_MONGO_URL` at a real server
   switches backend with no code change.

8. **Content embeddings default to TF-IDF + truncated SVD, not
   sentence-transformers.** Table 7 specifies sentence-transformers; the package
   was not installed and its models require a download at first use, which would
   make the pipeline non-reproducible offline. The default backend is
   deterministic and needs no network. The documented backend is implemented and
   selected with `RECO_CONTENT_EMBEDDING_BACKEND=sentence-transformers`.

9. **Only the neural model has a dedicated training script.** Section 9.5 says
   "the training of each model is an offline script", but the Section 9.2 tree
   declares no training module. The ALS factorisation, the content embeddings
   and the popularity ranking are produced by `data/pipeline/transform.py` and
   published by `materialize.py` — they *are* the feature tables Table 8 stage 4
   writes. Only `DeepLearningStrategy` needs separate weights, and it carries
   its own entry point: `python -m ai.strategies.deep_learning_strategy --train`.
   In every case the inference path never trains, which is the property the
   section is protecting.

10. **`src/persistence/__main__.py` was added.** Table 10 requires the profile
    repository to answer `UserNotFoundError`, and the controller enriches a
    ranking with catalog titles; both need the stores populated. This CLI loads
    them from the artifacts the offline pipeline already produced, so the
    databases are never written from the request path.

11. **The import-direction rule has a declared composition-root exemption.**
    `application/api.py` (whole module) and `messaging/feedback_history_service.py::main`
    name concrete classes, because wiring implementations together is what a
    composition root is for. `tests/test_architecture_imports.py` declares that
    exempt set, asserts it is exactly those two, and additionally verifies the
    detector catches a real violation — so a green result is evidence, not an
    artefact of a rule that checks nothing.

12. **A null-object strategy was added to the composition root.** If neither the
    active strategy nor the fallback can load — a clean machine with no
    materialised artifacts — the service previously failed at startup. A service
    that cannot start is not the controlled degradation NFR-05 asks for, so it
    now starts, `/health` reports `degraded`, and a recommendation request
    returns an explicit translated error instead of a crash.

13. **Two endpoints beyond the specified one.** Section 9.6 specifies
    `GET /recommendations/{user_id}?k=10`. `POST /interactions` was added because
    FR-02 needs an entry point for real-time interaction capture, and
    `GET /health` because the failure-injection tests and the operator need to
    see which backends and strategy are live.

14. **`DeepLearningStrategy` receives no demographic attributes.** Table 9 lists
    "embedding identifiers, demographic attributes, aggregated statistics" as its
    inputs. MovieLens ships no demographic data in this variant, so the model is
    fed the embedding identifiers and the three aggregated statistics only.

15. **The response carries `degraded` and `degradation_reason`.** Not specified
    in the document. Table 10 distinguishes a normal 200 from a "200 degraded"
    but gives the client no way to tell them apart; these two fields make the
    distinction observable, and the failure-injection tests assert on them.

16. **A shared `StrategyDecorator` base and a per-request operational context
    were added** in `ai/decorators/__init__.py`. Section 9.9 requires one
    operational record carrying which strategy ran, whether the cache was hit,
    the latency and the model version — facts produced by three different
    decorators. A `ContextVar` lets each contribute its fields so
    `LoggingDecorator` emits one record, without the decorators knowing about
    each other and without breaking concurrent requests.

---

## (d) Not implemented

**Specified in the document but absent:**

- **Authentication and authorization.** Section 5.2 assigns them to the
  Application/Controller layer. No auth exists: every request is anonymous.
- **Presentation layer beyond schemas.** Section 5.2 describes a UI/Web/Mobile
  layer. The Section 9.2 tree declares only `presentation/schemas.py`, and that
  is all that was built — there is no user interface.
- **Horizontal scalability (NFR-02).** No container image, no orchestration
  manifest, no load-balancing configuration. The in-process cache and queue are
  per-instance by construction, so running several replicas requires switching
  to the Redis and RabbitMQ backends first.
- **The pre-commit hook.** Section 9.2 states that black, ruff and mypy run in a
  pre-commit hook. All three are configured in `pyproject.toml` and pass, but no
  `.pre-commit-config.yaml` was added; they are run manually.

**Implemented but never exercised against real infrastructure:**

- The Redis, RabbitMQ, PostgreSQL and MongoDB backends. The code paths exist and
  are typed and linted, but no server was available, so they have never handled
  a real connection. Only the fallbacks are covered by tests.
- The sentence-transformers embedding backend, for the same reason.
- The `ml-25m` variant. The dataset path and variant are configuration
  (`RECO_DATASET_VARIANT`) and ingestion reads in chunks, and a test asserts the
  variant switch needs no code change — but only `ml-latest-small` was actually
  processed end to end.

**Mentioned as future extension, deliberately out of scope:**

- `RateLimitingDecorator` and A/B testing (Sections 3.3, 3.5, 7) — cited in the
  document as examples of what the design *allows*, not as Milestone 2 work. The
  decorator registry makes either a new file plus one registration.
- Approximate-nearest-neighbour index (Section 9.7) — the document itself leaves
  it out, and the measured p95 above confirms the reasoning.

---

## Verification commands

```bash
python -m pytest                    # 258 tests
python -m ruff check .              # clean
python -m black --check .           # clean
python -m mypy                      # clean, 38 source files
python benchmarks/latency_p95.py    # p95 vs the 200 ms target, exit 1 on miss
```

Full rebuild from the raw dataset:

```bash
python -m data.pipeline --download                          # 4 stages, writes the feature store
python -m ai.strategies.deep_learning_strategy --train      # NCF weights
python -m persistence --seed                                # profiles, catalog, embeddings
python -m uvicorn application.api:app --port 8000           # API
python -m messaging.feedback_history_service                # FR-06 worker (separate process)
```
