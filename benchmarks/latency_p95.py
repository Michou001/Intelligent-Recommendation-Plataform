"""Replay a sample of requests and report the p95 latency (NFR-01, Section 9.7).

    python benchmarks/latency_p95.py                  # in-process controller
    python benchmarks/latency_p95.py --no-cache       # worst case, every request infers
    python benchmarks/latency_p95.py --mode http --base-url http://127.0.0.1:8000

The point of this script is that compliance with the 200 ms target at the
95th percentile is reported as a *measured* value and not as an assumption.
It therefore prints the sample size, the distribution, and an explicit verdict
against the target, and it exits non-zero when the target is missed, so it can
be wired into CI.

Two modes are offered because they answer different questions. ``controller``
measures the inference cycle the architecture is responsible for - validation,
feature retrieval, decorated execution, ranking, enrichment - without HTTP
framing. ``http`` measures what a client observes, including serialisation and
the ASGI stack.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai.decorators import build_chain  # noqa: E402
from ai.model_factory import ModelFactory  # noqa: E402
from application.recommendation_controller import RecommendationController  # noqa: E402
from config.settings import load_settings  # noqa: E402
from data.feature_store_manager import ParquetFeatureStoreManager  # noqa: E402
from persistence.sql_catalog_repository import SqlCatalogRepository  # noqa: E402
from persistence.sql_user_profile_repository import SqlUserProfileRepository  # noqa: E402

#: The target of NFR-01, in milliseconds.
TARGET_P95_MS = 200.0


@dataclass
class BenchmarkResult:
    """Distribution of the replayed latencies, in milliseconds."""

    samples: list[float] = field(default_factory=list)
    errors: int = 0
    degraded: int = 0

    def percentile(self, percentile: float) -> float:
        return float(np.percentile(np.asarray(self.samples, dtype=np.float64), percentile))

    def summary(self, target_ms: float = TARGET_P95_MS) -> dict[str, object]:
        if not self.samples:
            return {"requests": 0, "verdict": "NO DATA"}
        p95 = self.percentile(95)
        return {
            "requests": len(self.samples),
            "errors": self.errors,
            "degraded_responses": self.degraded,
            "min_ms": round(min(self.samples), 3),
            "mean_ms": round(statistics.fmean(self.samples), 3),
            "p50_ms": round(self.percentile(50), 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(self.percentile(99), 3),
            "max_ms": round(max(self.samples), 3),
            "target_p95_ms": target_ms,
            "headroom_ms": round(target_ms - p95, 3),
            "verdict": "PASS" if p95 <= target_ms else "FAIL",
        }


def _sample_user_ids(count: int, seed: int) -> list[int]:
    """Draw the user identifiers to replay, from the materialised store."""
    import pandas as pd

    settings = load_settings()
    frame = pd.read_parquet(
        settings.feature_store.version_dir / "user_features.parquet", columns=["user_id"]
    )
    available = frame["user_id"].to_numpy(dtype=np.int64)
    if available.size == 0:
        raise SystemExit("the feature store holds no user; run the offline pipeline first")
    rng = np.random.default_rng(seed)
    return [int(u) for u in rng.choice(available, size=count, replace=True)]


def build_controller(*, use_cache: bool, strategy_id: str | None) -> RecommendationController:
    """Compose the controller the same way the service does at startup."""
    settings = load_settings()
    chain = list(settings.model.decorator_chain)
    if not use_cache:
        chain = [name for name in chain if name != "caching"]

    fallback = ModelFactory.create(settings.model.fallback_strategy, settings)
    base = ModelFactory.create(strategy_id or settings.model.active_strategy, settings)
    return RecommendationController(
        strategy=build_chain(base, chain, settings=settings),
        feature_store=ParquetFeatureStoreManager.from_settings(settings),
        catalog_repository=SqlCatalogRepository.from_settings(settings),
        profile_repository=SqlUserProfileRepository.from_settings(settings),
        fallback_strategy=fallback,
        settings=settings,
    )


def replay(
    call: Callable[[int], bool],
    user_ids: Sequence[int],
    *,
    warmup: int,
) -> BenchmarkResult:
    """Time each replayed request, after discarding the warm-up ones."""
    result = BenchmarkResult()
    for index, user_id in enumerate(user_ids):
        started = time.perf_counter()
        try:
            degraded = call(user_id)
        except Exception:  # noqa: BLE001 - an error is a data point, not a crash
            if index >= warmup:
                result.errors += 1
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if index < warmup:
            continue
        result.samples.append(elapsed_ms)
        if degraded:
            result.degraded += 1
    return result


def run_controller_mode(
    *, requests: int, k: int, warmup: int, seed: int, use_cache: bool, strategy_id: str | None
) -> BenchmarkResult:
    controller = build_controller(use_cache=use_cache, strategy_id=strategy_id)
    user_ids = _sample_user_ids(requests + warmup, seed)

    def call(user_id: int) -> bool:
        return controller.recommend(user_id, k).degraded

    return replay(call, user_ids, warmup=warmup)


def run_http_mode(
    *, requests: int, k: int, warmup: int, seed: int, base_url: str
) -> BenchmarkResult:
    import httpx

    user_ids = _sample_user_ids(requests + warmup, seed)
    with httpx.Client(base_url=base_url, timeout=10.0) as client:

        def call(user_id: int) -> bool:
            response = client.get(f"/recommendations/{user_id}", params={"k": k})
            response.raise_for_status()
            return bool(response.json().get("degraded", False))

        return replay(call, user_ids, warmup=warmup)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python benchmarks/latency_p95.py",
        description="Measure the p95 latency of the recommendation endpoint against NFR-01.",
    )
    parser.add_argument("--requests", type=int, default=500, help="Timed requests to replay")
    parser.add_argument("--warmup", type=int, default=25, help="Untimed requests before measuring")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategy", default=None, help="Override the active strategy")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Drop CachingDecorator, so every request runs the real inference",
    )
    parser.add_argument("--mode", choices=("controller", "http"), default="controller")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target-ms", type=float, default=TARGET_P95_MS)
    parser.add_argument("--json", action="store_true", help="Print only the JSON summary")
    args = parser.parse_args(argv)

    if args.mode == "controller":
        result = run_controller_mode(
            requests=args.requests,
            k=args.k,
            warmup=args.warmup,
            seed=args.seed,
            use_cache=not args.no_cache,
            strategy_id=args.strategy,
        )
    else:
        result = run_http_mode(
            requests=args.requests,
            k=args.k,
            warmup=args.warmup,
            seed=args.seed,
            base_url=args.base_url,
        )

    settings = load_settings()
    summary = {
        "mode": args.mode,
        "strategy": args.strategy or settings.model.active_strategy,
        "caching": not args.no_cache,
        "k": args.k,
        "dataset_variant": settings.dataset.variant,
        "feature_store_version": settings.feature_store.version,
        **result.summary(args.target_ms),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\nLatency benchmark - NFR-01 target: {args.target_ms:.0f} ms at p95")
        print("-" * 58)
        for key, value in summary.items():
            print(f"  {key:24} {value}")
        print("-" * 58)
        print(f"  VERDICT: {summary.get('verdict')}\n")

    return 0 if summary.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
