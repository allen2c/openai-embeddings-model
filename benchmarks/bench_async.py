"""Benchmark: AsyncOpenAIEmbeddingsModel — event loop health and thread-pool limits.

Answers one question: is this safe to put inside a FastAPI service, and where
does it stop scaling? Everything here targets `AsyncOpenAIEmbeddingsModel`
specifically — its single shared `ThreadPoolExecutor` (created once in
`_new_executor`, sized by `executor_max_workers`, default `None` ->
`min(32, cpu_count + 4)` = 14 on the reference machine) and how much of
`get_embeddings` actually runs ON the event loop rather than being offloaded
to that pool.

Reading `openai_embeddings_model/__init__.py` end to end shows that inside
`AsyncOpenAIEmbeddingsModel.get_embeddings`, the following run directly on the
event loop, never touching the executor:

  - `validate_input`                        — type/emptiness checks, O(n)
  - `deduplicate_texts`                      — O(n) dict lookups
  - the per-text key generation   — one SHA-256 hash per unique
                                                 text (`generate_cache_key`)
  - the `validate_cached_embedding` list
    comprehension inside `_cache_get_many`    — `_cache_get_many` only
                                                 offloads the raw `cache.get`
                                                 calls to the executor; once
                                                 the raw values are back, this
                                                 comprehension (base64 decode +
                                                 `np.isfinite` + a length
                                                 check per hit) runs back on
                                                 the loop
  - `ModelResponse.model_validate` at the end — pydantic construction

Everything else — `_prepare_batches` (tokenization), the actual `cache.get` /
`cache.set` sqlite calls, and (implicitly, via the SDK's own httpx client)
the network request — is offloaded or naturally async. Experiment 1 measures
what that residual on-loop work actually costs in wall-clock stall time.

Run: `python tmp/bench/bench_async.py --quick` from the repo root.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import pathlib
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as c

MODEL = c.MODEL
DIM = c.DIM
DEFAULT_WORKERS = min(32, (os.cpu_count() or 1) + 4)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def ms(seconds: float) -> float:
    return seconds * 1000.0


def percentiles(samples: list[float], ps: tuple[int, ...] = (50, 95, 99)) -> dict[str, float]:
    if not samples:
        return {f"p{p}": 0.0 for p in ps}
    ordered = sorted(samples)
    n = len(ordered)
    return {f"p{p}": ordered[min(n - 1, int(n * p / 100))] for p in ps}


class TrackedExecutor:
    """Wraps a ThreadPoolExecutor to record the peak number of jobs in flight.

    `run_in_executor` calls `submit` under the hood, so substituting this in
    for `model._executor` exposes whether the pool itself is the bottleneck
    without touching any library code.
    """

    def __init__(self, real: ThreadPoolExecutor) -> None:
        self._real = real
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        fut = self._real.submit(fn, *args, **kwargs)
        fut.add_done_callback(self._on_done)
        return fut

    def _on_done(self, _fut) -> None:
        with self._lock:
            self._active -= 1

    def shutdown(self, *args, **kwargs):
        return self._real.shutdown(*args, **kwargs)


def track(model) -> TrackedExecutor:
    tracked = TrackedExecutor(model._executor)
    model._executor = tracked
    return tracked


async def warm_hit_cache(model, texts: list[str], settings) -> list[str]:
    """Pre-populate `model`'s cache so a later call to the same texts is an all-hit."""
    keys = [c.legacy_cache_key(model, t, settings) for t in texts]
    c.warm_cache(model._cache, keys, c.b64_vector(DIM))
    return keys


def max_overlap(intervals: list[tuple[float, float]]) -> int:
    """Peak number of simultaneously-open (start, end) intervals — effective parallelism."""
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


# ---------------------------------------------------------------------------
# 1. event loop blocking during a single get_embeddings call
# ---------------------------------------------------------------------------


async def exp1_loop_blocking(quick: bool, results: dict) -> None:
    c.banner("1. Event loop blocking during a single get_embeddings call")
    sizes = [64, 512] if quick else [64, 512, 2048, 4096]
    reps = 1 if quick else 3
    settings = c.settings()

    rows = []
    raw = {}
    for n in sizes:
        texts = c.make_texts(n, seed=n)
        for state in ("all_hit", "all_miss_0ms", "no_cache"):
            summaries = []
            for rep in range(reps):
                with tempfile.TemporaryDirectory() as tmpdir:
                    cache = None if state == "no_cache" else c.fresh_cache(tmpdir)
                    model = c.async_model(c.async_create(latency=0.0), cache=cache)
                    try:
                        if state == "all_hit":
                            await warm_hit_cache(model, texts, settings)
                        # Warm-up call on different texts: pays for tokenizer / JIT
                        # / thread-spawn overhead without touching the keys used
                        # for the actual measurement below.
                        warm_texts = c.make_texts(8, seed=n * 1000 + rep)
                        await model.get_embeddings(warm_texts, settings)

                        probe = c.LoopLagProbe()
                        async with probe:
                            await model.get_embeddings(texts, settings)
                        summaries.append(probe.summary())
                    finally:
                        await model.aclose()
                        if cache is not None:
                            cache.close()

            agg = {
                "median_ms": ms(statistics.median(s["median"] for s in summaries)),
                "p95_ms": ms(max(s["p95"] for s in summaries)),
                "max_ms": ms(max(s["max"] for s in summaries)),
            }
            rows.append({"n": n, "state": state, **agg})
            raw[f"{n}_{state}"] = summaries

    print(c.table(rows, ["n", "state", "median_ms", "p95_ms", "max_ms"]))
    results["loop_blocking"] = {
        "note": (
            "On-loop work per call: validate_input, deduplicate_texts, the "
            "per-text SHA-256 hashes, the validate_cached_embedding "
            "decode+isfinite pass over cache hits, and ModelResponse.model_validate. "
            "all_hit pays the full validate_cached_embedding cost; all_miss_0ms and "
            "no_cache skip most of it (nothing to decode)."
        ),
        "rows": rows,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# 2. a blocked loop hurts other traffic
# ---------------------------------------------------------------------------


async def exp2_deadline_impact(quick: bool, results: dict) -> None:
    c.banner("2. Simulated 5ms-cadence requests during one large get_embeddings call")
    n = 1024 if quick else 4096
    settings = c.settings()

    rows = []
    raw = {}
    for state in ("all_hit", "all_miss_0ms"):
        texts = c.make_texts(n, seed=999)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = c.fresh_cache(tmpdir)
            model = c.async_model(c.async_create(latency=0.0), cache=cache)
            try:
                if state == "all_hit":
                    await warm_hit_cache(model, texts, settings)
                await model.get_embeddings(c.make_texts(8, seed=1), settings)  # warm-up

                probe = c.LoopLagProbe()
                t0 = time.perf_counter()
                async with probe:
                    await model.get_embeddings(texts, settings)
                wall = time.perf_counter() - t0

                # A "request" that arrives on the 5ms cadence but is serviced more
                # than one interval late (>10ms total) has missed its deadline.
                lags = probe.lags
                missed = [lag for lag in lags if lag > probe.interval]
                rows.append(
                    {
                        "state": state,
                        "n_ticks": len(lags),
                        "missed": len(missed),
                        "miss_pct": 100.0 * len(missed) / len(lags) if lags else 0.0,
                        "worst_delay_ms": ms(max(lags)) if lags else 0.0,
                        "call_wall_ms": ms(wall),
                    }
                )
                raw[state] = {"lags_ms": [ms(x) for x in lags], "wall_ms": ms(wall)}
            finally:
                await model.aclose()
                cache.close()

    print(c.table(rows, ["state", "n_ticks", "missed", "miss_pct", "worst_delay_ms", "call_wall_ms"]))
    results["deadline_impact"] = {"rows": rows, "raw": raw, "n_texts": n}


# ---------------------------------------------------------------------------
# 3. executor_max_workers sweep
# ---------------------------------------------------------------------------


async def exp3_executor_workers_sweep(quick: bool, results: dict) -> None:
    c.banner("3. executor_max_workers sweep, N concurrent get_embeddings, all-hit")
    worker_counts = [1, 4, 14] if quick else [1, 2, 4, 8, 14, 32, 64]
    concurrency = 8 if quick else 32
    n_texts = 64 if quick else 512
    settings = c.settings()
    texts = c.make_texts(n_texts, seed=42)

    rows = []
    raw = {}
    for workers in worker_counts:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = c.fresh_cache(tmpdir)
            model = c.async_model(c.async_create(latency=0.0), cache=cache, executor_max_workers=workers)
            tracked = track(model)
            try:
                await warm_hit_cache(model, texts, settings)
                await asyncio.gather(*[model.get_embeddings(texts, settings) for _ in range(min(4, concurrency))])
                tracked.max_active = 0  # reset after warm-up

                latencies: list[float] = []

                async def one_call(model=model, latencies=latencies) -> None:
                    t0 = time.perf_counter()
                    await model.get_embeddings(texts, settings)
                    latencies.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                await asyncio.gather(*[one_call() for _ in range(concurrency)])
                wall = time.perf_counter() - t0

                pct = percentiles(latencies)
                rows.append(
                    {
                        "workers": workers,
                        "wall_s": wall,
                        "p50_ms": ms(pct["p50"]),
                        "p95_ms": ms(pct["p95"]),
                        "peak_active_jobs": tracked.max_active,
                    }
                )
                raw[str(workers)] = {"latencies_ms": [ms(x) for x in latencies], "wall_s": wall}
            finally:
                await model.aclose()
                cache.close()

    print(c.table(rows, ["workers", "wall_s", "p50_ms", "p95_ms", "peak_active_jobs"]))

    if DEFAULT_WORKERS in worker_counts:
        default_row = next(r for r in rows if r["workers"] == DEFAULT_WORKERS)
        fastest = min(rows, key=lambda r: r["wall_s"])
        if fastest["workers"] == DEFAULT_WORKERS:
            verdict = "default is fastest"
        else:
            delta = (1 - fastest["wall_s"] / default_row["wall_s"]) * 100
            verdict = f"workers={fastest['workers']} beats default by {delta:.1f}%"
        print(f"-> default executor_max_workers={DEFAULT_WORKERS}: wall={default_row['wall_s']:.3f}s. {verdict}.")

    results["executor_workers_sweep"] = {
        "rows": rows,
        "raw": raw,
        "concurrency": concurrency,
        "n_texts": n_texts,
        "default_workers": DEFAULT_WORKERS,
    }


# ---------------------------------------------------------------------------
# 4. max_concurrent_batches sweep
# ---------------------------------------------------------------------------


async def exp4_max_concurrent_batches_sweep(quick: bool, results: dict) -> None:
    c.banner("4. max_concurrent_batches sweep, all-miss + fake provider latency")
    values = [1, 5, 20] if quick else [1, 2, 5, 10, 20, 50]
    n_texts = 1024 if quick else 8192
    batch_size = 50 if quick else 200
    latency = 0.01 if quick else 0.05
    settings = c.settings()
    texts = c.make_texts(n_texts, seed=7)

    rows = []
    raw = {}
    for mcb in values:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = c.fresh_cache(tmpdir)
            intervals: list[tuple[float, float]] = []

            async def create(input, _intervals=intervals, **kwargs):
                items = list(input)
                t0 = time.perf_counter()
                await asyncio.sleep(latency)
                _intervals.append((t0, time.perf_counter()))
                return c.build_response(len(items))

            model = c.async_model(create, cache=cache, max_concurrent_batches=mcb, max_batch_size=batch_size)
            try:
                t0 = time.perf_counter()
                await model.get_embeddings(texts, settings)
                wall = time.perf_counter() - t0

                n_batches = len(intervals)
                parallelism = max_overlap(intervals)
                rows.append(
                    {
                        "max_concurrent_batches": mcb,
                        "n_batches": n_batches,
                        "wall_s": wall,
                        "effective_parallelism": parallelism,
                    }
                )
                raw[str(mcb)] = {"n_batches": n_batches, "wall_s": wall, "effective_parallelism": parallelism}
            finally:
                await model.aclose()
                cache.close()

    print(c.table(rows, ["max_concurrent_batches", "n_batches", "wall_s", "effective_parallelism"]))

    if len(rows) >= 2:
        best = min(r["wall_s"] for r in rows)
        for r in rows:
            if r["wall_s"] <= best * 1.05:
                print(
                    f"-> diminishing returns at max_concurrent_batches={r['max_concurrent_batches']} "
                    "(wall within 5% of best)."
                )
                break

    results["max_concurrent_batches_sweep"] = {
        "rows": rows,
        "raw": raw,
        "n_texts": n_texts,
        "batch_size": batch_size,
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# 5. concurrency sweep on one shared model
# ---------------------------------------------------------------------------


async def exp5_concurrency_sweep(quick: bool, results: dict) -> None:
    c.banner("5. Concurrency sweep, one shared model, all-hit")
    levels = [1, 8, 32] if quick else [1, 4, 16, 64, 256]
    n_texts = 32 if quick else 128
    settings = c.settings()
    texts = c.make_texts(n_texts, seed=123)

    rows = []
    raw = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = c.fresh_cache(tmpdir)
        model = c.async_model(c.async_create(latency=0.0), cache=cache)
        tracked = track(model)
        try:
            await warm_hit_cache(model, texts, settings)
            await model.get_embeddings(texts, settings)  # warm-up

            for level in levels:
                tracked.max_active = 0
                latencies: list[float] = []

                async def one_call(latencies=latencies) -> None:
                    t0 = time.perf_counter()
                    await model.get_embeddings(texts, settings)
                    latencies.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                await asyncio.gather(*[one_call() for _ in range(level)])
                wall = time.perf_counter() - t0

                pct = percentiles(latencies)
                texts_per_sec = (level * n_texts) / wall if wall > 0 else float("inf")
                rows.append(
                    {
                        "concurrency": level,
                        "texts_per_sec": texts_per_sec,
                        "p50_ms": ms(pct["p50"]),
                        "p95_ms": ms(pct["p95"]),
                        "p99_ms": ms(pct["p99"]),
                        "peak_active_jobs": tracked.max_active,
                    }
                )
                raw[str(level)] = {"latencies_ms": [ms(x) for x in latencies], "wall_s": wall}
        finally:
            await model.aclose()
            cache.close()

    print(c.table(rows, ["concurrency", "texts_per_sec", "p50_ms", "p95_ms", "p99_ms", "peak_active_jobs"]))
    print(
        f"-> peak_active_jobs saturating near executor_max_workers ({DEFAULT_WORKERS}) points at the shared "
        "thread pool as the first bottleneck, not the GIL or sqlite directly."
    )
    results["concurrency_sweep"] = {"rows": rows, "raw": raw, "n_texts": n_texts, "default_workers": DEFAULT_WORKERS}


# ---------------------------------------------------------------------------
# 6. one model vs many models
# ---------------------------------------------------------------------------


async def exp6_one_vs_many_models(quick: bool, results: dict) -> None:
    c.banner("6. One shared model vs 4 model instances (shared cache directory)")
    total_calls = 16 if quick else 64
    n_texts = 64 if quick else 256
    n_models = 4
    settings = c.settings()
    texts = c.make_texts(n_texts, seed=55)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = c.fresh_cache(tmpdir)

        primer = c.async_model(c.async_create(), cache=cache)
        await warm_hit_cache(primer, texts, settings)
        await primer.aclose()

        # Scenario A: one shared model instance handles every call.
        model = c.async_model(c.async_create(latency=0.0), cache=cache)
        try:
            await model.get_embeddings(texts, settings)  # warm-up
            latencies_one: list[float] = []

            async def call_one() -> None:
                t0 = time.perf_counter()
                await model.get_embeddings(texts, settings)
                latencies_one.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            await asyncio.gather(*[call_one() for _ in range(total_calls)])
            wall_one = time.perf_counter() - t0
        finally:
            await model.aclose()

        # Scenario B: 4 instances, each with its own executor, sharing the cache.
        models = [c.async_model(c.async_create(latency=0.0), cache=cache) for _ in range(n_models)]
        try:
            await asyncio.gather(*[m.get_embeddings(texts, settings) for m in models])  # warm-up
            latencies_many: list[float] = []

            async def call_many(m) -> None:
                t0 = time.perf_counter()
                await m.get_embeddings(texts, settings)
                latencies_many.append(time.perf_counter() - t0)

            assignments = [models[i % n_models] for i in range(total_calls)]
            t0 = time.perf_counter()
            await asyncio.gather(*[call_many(m) for m in assignments])
            wall_many = time.perf_counter() - t0
        finally:
            for m in models:
                await m.aclose()

        cache.close()

    pct_one = percentiles(latencies_one)
    pct_many = percentiles(latencies_many)
    rows = [
        {"scenario": "1 shared model", "instances": 1, "wall_s": wall_one, "p95_ms": ms(pct_one["p95"])},
        {
            "scenario": f"{n_models} models, shared cache dir",
            "instances": n_models,
            "wall_s": wall_many,
            "p95_ms": ms(pct_many["p95"]),
        },
    ]
    print(c.table(rows, ["scenario", "instances", "wall_s", "p95_ms"]))
    results["one_vs_many_models"] = {
        "rows": rows,
        "raw": {
            "one_model": {"latencies_ms": [ms(x) for x in latencies_one], "wall_s": wall_one},
            "many_models": {"latencies_ms": [ms(x) for x in latencies_many], "wall_s": wall_many},
        },
        "total_calls": total_calls,
        "n_texts": n_texts,
        "n_models": n_models,
    }


# ---------------------------------------------------------------------------
# 7. sync vs async head-to-head
# ---------------------------------------------------------------------------


async def exp7_sync_vs_async(quick: bool, results: dict) -> None:
    c.banner("7. Sync vs async head-to-head at equal concurrency")
    concurrency = 8 if quick else 32
    n_texts = 64 if quick else 512
    latency = 0.01 if quick else 0.05
    settings = c.settings()
    texts = c.make_texts(n_texts, seed=321)

    rows = []
    raw = {}
    for case, lat in (("all_hit", 0.0), ("all_miss_latency", latency)):
        with tempfile.TemporaryDirectory() as tmpdir_sync, tempfile.TemporaryDirectory() as tmpdir_async:
            cache_sync = c.fresh_cache(tmpdir_sync)
            cache_async = c.fresh_cache(tmpdir_async)
            sync_mod = c.sync_model(c.sync_create(latency=lat), cache=cache_sync)
            async_mod = c.async_model(c.async_create(latency=lat), cache=cache_async)
            try:
                if case == "all_hit":
                    skeys = [c.legacy_cache_key(sync_mod, t, settings) for t in texts]
                    c.warm_cache(cache_sync, skeys, c.b64_vector(DIM))
                    akeys = [c.legacy_cache_key(async_mod, t, settings) for t in texts]
                    c.warm_cache(cache_async, akeys, c.b64_vector(DIM))

                pool = ThreadPoolExecutor(max_workers=concurrency)
                loop = asyncio.get_running_loop()
                try:
                    t0 = time.perf_counter()
                    await asyncio.gather(
                        *[
                            loop.run_in_executor(pool, sync_mod.get_embeddings, texts, settings)
                            for _ in range(concurrency)
                        ]
                    )
                    wall_sync = time.perf_counter() - t0
                finally:
                    pool.shutdown(wait=True)

                t0 = time.perf_counter()
                await asyncio.gather(*[async_mod.get_embeddings(texts, settings) for _ in range(concurrency)])
                wall_async = time.perf_counter() - t0

                rows.append(
                    {
                        "case": case,
                        "sync_wall_s": wall_sync,
                        "async_wall_s": wall_async,
                        "winner": "sync" if wall_sync < wall_async else "async",
                    }
                )
                raw[case] = {"sync_wall_s": wall_sync, "async_wall_s": wall_async}
            finally:
                await async_mod.aclose()
                cache_sync.close()
                cache_async.close()

    print(c.table(rows, ["case", "sync_wall_s", "async_wall_s", "winner"]))
    results["sync_vs_async"] = {"rows": rows, "raw": raw, "concurrency": concurrency, "n_texts": n_texts}


# ---------------------------------------------------------------------------
# 8. executor lifetime
# ---------------------------------------------------------------------------


async def exp8_executor_lifetime(quick: bool, results: dict) -> None:
    c.banner("8. Executor lifetime: create/close cost and thread-leak footprint")
    reps = 5 if quick else 20
    cycles = 20 if quick else 200
    settings = c.settings()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = c.fresh_cache(tmpdir)

        create_close_s: list[float] = []
        for i in range(reps):
            t0 = time.perf_counter()
            m = c.async_model(c.async_create(), cache=cache, executor_max_workers=2)
            await m.get_embeddings([f"lifetime-{i}"], settings)
            await m.aclose()
            create_close_s.append(time.perf_counter() - t0)
        lifetime_stats = c.stats(create_close_s)

        baseline = threading.active_count()

        # Deliberately NOT calling aclose() below: this is the exact misuse
        # (a model created per request and dropped) the experiment measures.
        # The interpreter's __del__ safety net (see AsyncOpenAIEmbeddingsModel
        # .__del__) is the thing under test — does plain refcounting reclaim
        # the executor's thread, or is gc.collect() required?
        counts_no_gc: list[int] = []
        m = None
        for i in range(cycles):
            m = c.async_model(c.async_create(), cache=cache, executor_max_workers=1)
            await m.get_embeddings([f"leak-nogc-{i}"], settings)
            counts_no_gc.append(threading.active_count())
        del m
        await asyncio.sleep(0.1)  # let any reclaimed executors finish tearing down
        after_no_gc = threading.active_count()

        counts_with_gc: list[int] = []
        m = None
        for i in range(cycles):
            m = c.async_model(c.async_create(), cache=cache, executor_max_workers=1)
            await m.get_embeddings([f"leak-gc-{i}"], settings)
            gc.collect()
            counts_with_gc.append(threading.active_count())
        del m
        await asyncio.sleep(0.1)
        after_with_gc = threading.active_count()

        cache.close()

    rows = [
        {"metric": "create+call+aclose median (ms)", "value": ms(lifetime_stats["median"])},
        {"metric": "create+call+aclose p95 (ms)", "value": ms(lifetime_stats["p95"])},
        {"metric": "baseline active threads", "value": baseline},
        {"metric": f"after {cycles} cycles, no aclose, no gc.collect()", "value": after_no_gc},
        {"metric": f"after {cycles} cycles, no aclose, gc.collect() every cycle", "value": after_with_gc},
    ]
    print(c.table(rows, ["metric", "value"]))

    results["executor_lifetime"] = {
        "create_close_ms_stats": {k: (ms(v) if k != "n" else v) for k, v in lifetime_stats.items()},
        "baseline_threads": baseline,
        "cycles": cycles,
        "thread_count_series_no_gc": counts_no_gc,
        "thread_count_series_with_gc": counts_with_gc,
        "after_no_gc": after_no_gc,
        "after_with_gc": after_with_gc,
        "max_threads_no_gc": max(counts_no_gc) if counts_no_gc else baseline,
        "max_threads_with_gc": max(counts_with_gc) if counts_with_gc else baseline,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AsyncOpenAIEmbeddingsModel event-loop & scaling benchmark")
    parser.add_argument("--quick", action="store_true", help="tiny params, smoke test (<45s)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    quick = args.quick
    t_start = time.perf_counter()

    results: dict = {
        "meta": {
            "quick": quick,
            "model": MODEL,
            "dim": DIM,
            "machine": "Apple M4 Pro, 10 cores, 32 GB, Python 3.12",
            "default_executor_max_workers": DEFAULT_WORKERS,
        }
    }

    await exp1_loop_blocking(quick, results)
    await exp2_deadline_impact(quick, results)
    await exp3_executor_workers_sweep(quick, results)
    await exp4_max_concurrent_batches_sweep(quick, results)
    await exp5_concurrency_sweep(quick, results)
    await exp6_one_vs_many_models(quick, results)
    await exp7_sync_vs_async(quick, results)
    await exp8_executor_lifetime(quick, results)

    total_s = time.perf_counter() - t_start
    results["meta"]["total_wall_s"] = total_s
    path = c.save("async", results)
    c.banner(f"Done in {total_s:.1f}s. Results saved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
