"""Multi-threaded benchmark for `OpenAIEmbeddingsModel` (sync client).

Scope: one model instance (and usually one cache) shared across a thread
pool -- the classic Flask/Django/gunicorn-worker-threads deployment. See
`tmp/bench/common.py` for the shared harness and its ground rules (no
network, fake provider, `results/<name>.json` output).

Run:
    python tmp/bench/bench_sync.py --quick   # <45s smoke test
    python tmp/bench/bench_sync.py           # full run, ~8 minutes
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import itertools
import json
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import typing
import zlib

import numpy as np
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage as OpenAIUsage

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
# `common.py` itself imports `openai_embeddings_model`, which is not
# pip-installed in this environment (no console entry, no editable install)
# -- it only resolves when the repo root is on sys.path, which a plain
# `python tmp/bench/bench_sync.py` does not add on its own. Adding both
# directories here keeps `python tmp/bench/bench_sync.py` working from the
# repo root without requiring an install step.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
import common as c  # noqa: E402

from openai_embeddings_model import (  # noqa: E402
    ModelResponse,
    Usage,
    deduplicate_texts,
    validate_cached_embedding,
    validate_input,
)

THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
CPU_CORES = 10  # Apple M4 Pro, per the benchmark brief


# --------------------------------------------------------------------------
# concurrency harness
# --------------------------------------------------------------------------


def run_concurrent(
    n_threads: int,
    calls_per_thread: int,
    call_fn: typing.Callable[[int, int], int],
    *,
    warmup_calls: int = 2,
) -> tuple[float, list[float], int]:
    """Run `call_fn(thread_id, call_idx)` on `n_threads` threads, synchronized start.

    `call_fn` returns the number of texts it processed and is called
    `warmup_calls` extra times per thread (idx < 0) before the barrier releases,
    so each thread's first, cold call -- new sqlite connection, first executor
    hand-off, etc. -- does not pollute the measured window.

    Returns (aggregate_wall_seconds, pooled_latencies_seconds, total_texts).
    The aggregate window runs from the barrier release (all threads started
    together) to the last thread finishing -- not the sum of per-thread time.
    """
    state: dict[str, float] = {}

    def on_release() -> None:
        state["start"] = time.perf_counter()

    barrier = threading.Barrier(n_threads, action=on_release)
    lat_by_thread: list[list[float]] = [[] for _ in range(n_threads)]
    texts_by_thread = [0] * n_threads
    end_by_thread = [0.0] * n_threads

    def worker(tid: int) -> None:
        for w in range(1, warmup_calls + 1):
            call_fn(tid, -w)
        barrier.wait()
        texts = 0
        lat = lat_by_thread[tid]
        for i in range(calls_per_thread):
            t0 = time.perf_counter()
            texts += call_fn(tid, i)
            lat.append(time.perf_counter() - t0)
        end_by_thread[tid] = time.perf_counter()
        texts_by_thread[tid] = texts

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(worker, tid) for tid in range(n_threads)]
        for f in futures:
            f.result()

    wall = max(end_by_thread) - state["start"]
    all_lat = [x for thread_lat in lat_by_thread for x in thread_lat]
    return wall, all_lat, sum(texts_by_thread)


def scaling_note(rows: list[dict], *, throughput_key: str = "texts_per_sec") -> dict:
    """Summarize where throughput stops improving with more threads."""
    baseline = rows[0][throughput_key]
    best = max(rows, key=lambda r: r[throughput_key])
    plateau_at = rows[-1]["threads"]
    for prev, cur in itertools.pairwise(rows):
        if cur[throughput_key] < prev[throughput_key] * 1.15:
            plateau_at = prev["threads"]
            break
    return {
        "baseline_threads": rows[0]["threads"],
        "best_threads": best["threads"],
        "best_scaling_factor": round(best[throughput_key] / baseline, 3),
        "plateau_or_peak_at_threads": plateau_at,
        "cpu_cores": CPU_CORES,
    }


def unique_text_assignments(
    n_threads: int, calls_per_thread: int, texts_per_call: int, warmup_calls: int, seed_base: int
) -> dict[tuple[int, int], list[str]]:
    """Globally-distinct texts for every (thread, call) slot, including warmup.

    `make_texts` prefixes every text with `doc-{seed}-{i}`, so a distinct seed
    per slot guarantees distinct texts even when two threads race for the
    cache at the same instant -- required for a genuine all-miss workload,
    where a real hit (from a sibling thread's write) would understate the
    provider-call cost being measured.
    """
    assignments: dict[tuple[int, int], list[str]] = {}
    counter = 0
    for tid in range(n_threads):
        for i in range(-warmup_calls, calls_per_thread):
            assignments[(tid, i)] = c.make_texts(texts_per_call, seed=seed_base + counter)
            counter += 1
    return assignments


def latency_stats_ms(latencies: list[float]) -> dict[str, float]:
    s = c.stats(latencies)
    return {"median_ms": s["median"] * 1000, "p95_ms": s["p95"] * 1000}


# --------------------------------------------------------------------------
# 1. all-hit path
# --------------------------------------------------------------------------


def bench_all_hit(quick: bool) -> dict:
    c.banner("1. All-hit path (cache warm, provider never called)")

    pool_size = 500 if quick else 4000
    texts_per_call = 8 if quick else 16
    calls_per_thread = 20 if quick else 200

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        calls: list[int] = []
        model = c.sync_model(c.sync_create(calls=calls), cache=cache)
        settings = c.settings()
        pool = c.make_texts(pool_size, seed=1)
        keys = [c.legacy_cache_key(model, t, settings) for t in pool]
        c.warm_cache(cache, keys, c.b64_vector())

        def call_fn(tid: int, i: int) -> int:
            start = (tid * 997 + i * 131) % (pool_size - texts_per_call)
            texts = pool[start : start + texts_per_call]
            model.get_embeddings(texts, settings)
            return len(texts)

        rows = []
        for n in THREAD_COUNTS:
            wall, lat, total_texts = run_concurrent(n, calls_per_thread, call_fn, warmup_calls=3)
            row = {
                "threads": n,
                "wall_s": wall,
                "texts_per_sec": total_texts / wall,
                "calls_per_sec": (n * calls_per_thread) / wall,
                **latency_stats_ms(lat),
            }
            rows.append(row)

        provider_calls = len(calls)

    for r in rows:
        r["scaling_vs_1thread"] = r["texts_per_sec"] / rows[0]["texts_per_sec"]

    print(c.table(rows, ["threads", "texts_per_sec", "calls_per_sec", "median_ms", "p95_ms", "scaling_vs_1thread"]))
    note = scaling_note(rows)
    print(f"-> {note}")
    if provider_calls:
        print(f"WARNING: {provider_calls} provider calls happened on the all-hit path (expected 0)")

    return {
        "params": {"pool_size": pool_size, "texts_per_call": texts_per_call, "calls_per_thread": calls_per_thread},
        "provider_calls": provider_calls,
        "rows": rows,
        "note": note,
    }


# --------------------------------------------------------------------------
# 2 & 3. all-miss path (with cache) and no-cache baseline
# --------------------------------------------------------------------------


def _bench_all_miss(quick: bool, *, use_cache: bool, title: str) -> dict:
    c.banner(title)

    latency = 0.05
    texts_per_call = 4 if quick else 8
    calls_per_thread = 3 if quick else 20
    warmup_calls = 1

    rows = []
    total_provider_calls = 0
    for n in THREAD_COUNTS:
        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d) if use_cache else None
            calls: list[int] = []
            model = c.sync_model(c.sync_create(latency=latency, calls=calls), cache=cache)
            settings = c.settings()

            assignments = unique_text_assignments(n, calls_per_thread, texts_per_call, warmup_calls, seed_base=17)

            def call_fn(tid: int, i: int, _assignments=assignments, _model=model, _settings=settings) -> int:
                texts = _assignments[(tid, i)]
                _model.get_embeddings(texts, _settings)
                return len(texts)

            wall, lat, total_texts = run_concurrent(n, calls_per_thread, call_fn, warmup_calls=warmup_calls)
            total_provider_calls += len(calls)

        rows.append(
            {
                "threads": n,
                "wall_s": wall,
                "texts_per_sec": total_texts / wall,
                "calls_per_sec": (n * calls_per_thread) / wall,
                **latency_stats_ms(lat),
            }
        )

    for r in rows:
        r["scaling_vs_1thread"] = r["texts_per_sec"] / rows[0]["texts_per_sec"]

    print(c.table(rows, ["threads", "texts_per_sec", "calls_per_sec", "median_ms", "p95_ms", "scaling_vs_1thread"]))
    note = scaling_note(rows)
    print(f"-> {note}")

    return {
        "params": {
            "latency_s": latency,
            "texts_per_call": texts_per_call,
            "calls_per_thread": calls_per_thread,
            "cache": use_cache,
        },
        "provider_calls_total": total_provider_calls,
        "rows": rows,
        "note": note,
    }


def bench_all_miss(quick: bool) -> dict:
    return _bench_all_miss(quick, use_cache=True, title="2. All-miss path, 50ms/batch provider latency, shared cache")


def bench_no_cache(quick: bool) -> dict:
    return _bench_all_miss(quick, use_cache=False, title="3. No-cache baseline, 50ms/batch provider latency")


# --------------------------------------------------------------------------
# 4. where the wall-clock goes (all-hit, single call, component breakdown)
# --------------------------------------------------------------------------


def bench_breakdown(quick: bool) -> dict:
    c.banner("4. Wall-clock breakdown, single-threaded, all-hit")

    n = 64 if quick else 512
    repeats = 10 if quick else 50

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        model = c.sync_model(c.sync_create(), cache=cache)
        settings = c.settings()
        texts = c.make_texts(n, seed=999)
        keys = [c.legacy_cache_key(model, t, settings) for t in texts]
        c.warm_cache(cache, keys, c.b64_vector())
        raw_values = [cache.get(k) for k in keys]
        decoded = [validate_cached_embedding(k, v) for k, v in zip(keys, raw_values, strict=True)]
        payload = {
            "output": decoded,
            "usage": Usage(input_tokens=0, total_tokens=0, cache_hits=n, truncated_texts=0),
        }

        components = {
            "validate_input": lambda: validate_input(texts),
            "deduplicate_texts": lambda: deduplicate_texts(texts),
            f"cache_key_for (x{n})": lambda: [c.legacy_cache_key(model, t, settings) for t in texts],
            f"cache_reads (raw, x{n})": lambda: [cache.get(k) for k in keys],
            f"validate_cached_embedding (x{n})": (
                lambda: [validate_cached_embedding(k, v) for k, v in zip(keys, raw_values, strict=True)]
            ),
            "ModelResponse.model_validate": lambda: ModelResponse.model_validate(payload),
        }

        component_stats = {}
        for name, fn in components.items():
            component_stats[name] = c.timeit(fn, repeats=repeats, warmup=5)

        full_stats = c.timeit(lambda: model.get_embeddings(texts, settings), repeats=repeats, warmup=5)

    total_us = full_stats["median"] * 1e6
    rows = []
    summed_us = 0.0
    for name, s in component_stats.items():
        us = s["median"] * 1e6
        summed_us += us
        rows.append({"component": name, "us": us, "pct_of_total": 100 * us / total_us})
    rows.append({"component": "measured full get_embeddings() call", "us": total_us, "pct_of_total": 100.0})

    print(c.table(rows, ["component", "us", "pct_of_total"]))
    unaccounted_us = total_us - summed_us
    print(
        f"-> components sum to {summed_us:.1f}us ({100 * summed_us / total_us:.1f}% of the {total_us:.1f}us "
        f"measured call); {unaccounted_us:.1f}us ({100 * unaccounted_us / total_us:.1f}%) is unaccounted for -- "
        "model_settings.validate_for_model, the missing/cache_hits list comprehensions, logger.debug() f-string "
        "formatting (evaluated even though the default level suppresses the record), Usage construction, and "
        "general Python call/loop overhead not isolated above."
    )

    return {
        "params": {"n_texts": n, "repeats": repeats},
        "components_us": {name: s["median"] * 1e6 for name, s in component_stats.items()},
        "full_call_us": total_us,
        "summed_components_us": summed_us,
        "unaccounted_us": unaccounted_us,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# 5. batching behaviour (_prepare_batches: correctness, CPU cost, GIL)
# --------------------------------------------------------------------------


def bench_batching(quick: bool) -> dict:
    c.banner("5. Batching behaviour (_prepare_batches)")

    model = c.sync_model(c.sync_create())

    # Part A: batch count for a large all-miss request.
    n_large = 512 if quick else 4096
    large_texts = c.make_texts(n_large, seed=2)
    model._prepare_batches(large_texts)  # warm
    _, batches, truncated = model._prepare_batches(large_texts)
    print(f"-> {n_large} texts -> {len(batches)} provider batches (truncated={truncated})")

    # Part B: pure CPU cost (tiktoken encode_batch) at several sizes.
    sizes = [32, 64, 128, 256] if quick else [128, 512, 2048, 4096]
    repeats = 5 if quick else 10
    size_rows = []
    for size in sizes:
        texts = c.make_texts(size, seed=3)
        model._prepare_batches(texts)  # warm
        s = c.timeit(lambda t=texts: model._prepare_batches(t), repeats=repeats, warmup=2)
        size_rows.append(
            {
                "size": size,
                "median_ms": s["median"] * 1000,
                "us_per_text": s["median"] * 1e6 / size,
            }
        )
    print(c.table(size_rows, ["size", "median_ms", "us_per_text"]))

    # Part C: does _prepare_batches scale across threads? (tiktoken releases
    # the GIL during encode_batch in principle -- measure whether it actually
    # does on this build/machine.)
    thread_texts_per_call = 128 if quick else 512
    calls_per_thread = 5 if quick else 15
    threads_to_test = [1, 4] if quick else [1, 4, 8]

    pool_size = thread_texts_per_call * 4
    pool = c.make_texts(pool_size, seed=4)

    def call_fn(tid: int, i: int) -> int:
        start = (tid * 811 + i * 97) % (pool_size - thread_texts_per_call)
        texts = pool[start : start + thread_texts_per_call]
        model._prepare_batches(texts)
        return len(texts)

    thread_rows = []
    for n in threads_to_test:
        wall, _lat, total_texts = run_concurrent(n, calls_per_thread, call_fn, warmup_calls=2)
        thread_rows.append(
            {
                "threads": n,
                "wall_s": wall,
                "calls_per_sec": (n * calls_per_thread) / wall,
                "texts_per_sec": total_texts / wall,
            }
        )
    for r in thread_rows:
        r["scaling_vs_1thread"] = r["calls_per_sec"] / thread_rows[0]["calls_per_sec"]

    print(c.table(thread_rows, ["threads", "calls_per_sec", "texts_per_sec", "scaling_vs_1thread"]))
    gil_note = (
        "_prepare_batches (tiktoken encode_batch) does NOT scale across threads on this "
        "machine/tiktoken build -- throughput stays roughly flat or degrades as thread count "
        "rises, rather than increasing toward the core count. Either encode_batch is not "
        "releasing the GIL for useful stretches here, or per-call FFI/allocation overhead "
        "dominates once threads contend."
        if thread_rows[-1]["scaling_vs_1thread"] < 1.3
        else "_prepare_batches scales with threads here, consistent with tiktoken releasing the GIL."
    )
    print(f"-> {gil_note}")

    return {
        "large_request": {"n_texts": n_large, "batches": len(batches), "truncated": truncated},
        "cpu_cost_by_size": size_rows,
        "thread_scaling": {
            "params": {"texts_per_call": thread_texts_per_call, "calls_per_thread": calls_per_thread},
            "rows": thread_rows,
            "note": gil_note,
        },
    }


# --------------------------------------------------------------------------
# 6. correctness under concurrency
# --------------------------------------------------------------------------


def _deterministic_vector(text: str, dim: int = c.DIM) -> np.ndarray:
    seed = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _deterministic_b64(text: str, dim: int = c.DIM) -> str:
    return base64.b64encode(_deterministic_vector(text, dim).tobytes()).decode()


def _keyed_sync_create(dim: int = c.DIM) -> typing.Callable:
    """Fake provider whose output is a pure function of the input text.

    Unlike `common.sync_create` (one fixed vector for every call), this lets
    a correctness check confirm each returned embedding actually belongs to
    its text -- catching cross-text contamination from a cache-key collision
    or a race, not just "some vector came back".
    """

    def create(input, **kwargs):
        items = list(input)
        data = [
            Embedding.model_construct(embedding=_deterministic_b64(t, dim), index=i, object="embedding")
            for i, t in enumerate(items)
        ]
        return CreateEmbeddingResponse.model_construct(
            data=data,
            model=c.MODEL,
            object="list",
            usage=OpenAIUsage(prompt_tokens=len(items), total_tokens=len(items)),
        )

    return create


def bench_correctness(quick: bool) -> dict:
    c.banner("6. Correctness under concurrency")

    n_threads = 16
    pool_size = 60 if quick else 300
    calls_per_thread = 15 if quick else 150
    max_texts_per_call = 12

    failures: list[str] = []
    exceptions: list[str] = []
    lock = threading.Lock()
    totals = {"calls": 0, "texts_checked": 0}

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        model = c.sync_model(_keyed_sync_create(), cache=cache)
        settings = c.settings()
        pool = c.make_texts(pool_size, seed=42)

        def worker(tid: int) -> None:
            rng = random.Random(1000 + tid)
            local_calls = 0
            local_checked = 0
            for i in range(calls_per_thread):
                k = rng.randint(1, max_texts_per_call)
                texts = rng.sample(pool, k)
                if rng.random() < 0.2:  # occasionally add a text unique to this thread/call
                    texts.append(f"unique-{tid}-{i}-{rng.random()}")
                try:
                    resp = model.get_embeddings(texts, settings)
                except Exception as e:
                    with lock:
                        exceptions.append(f"tid={tid} i={i} error={e!r}")
                    continue
                local_calls += 1
                for text, out_b64 in zip(texts, resp.output, strict=True):
                    local_checked += 1
                    if out_b64 != _deterministic_b64(text):
                        with lock:
                            failures.append(f"tid={tid} i={i} text={text[:40]!r}")
            with lock:
                totals["calls"] += local_calls
                totals["texts_checked"] += local_checked

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker, tid) for tid in range(n_threads)]
            for f in futures:
                f.result()

    passed = not failures and not exceptions
    result = {
        "params": {
            "threads": n_threads,
            "pool_size": pool_size,
            "calls_per_thread": calls_per_thread,
            "max_texts_per_call": max_texts_per_call,
        },
        "total_calls": totals["calls"],
        "total_texts_checked": totals["texts_checked"],
        "mismatches": len(failures),
        "exceptions": len(exceptions),
        "passed": passed,
        "failure_samples": failures[:5],
        "exception_samples": exceptions[:5],
    }
    status = "PASS" if passed else "FAIL"
    print(
        f"-> {status}: {totals['calls']} calls, {totals['texts_checked']} vectors checked, "
        f"{len(failures)} mismatches, {len(exceptions)} exceptions"
    )
    return result


# --------------------------------------------------------------------------
# 7. get_embeddings_generator vs one big get_embeddings (wall + peak RSS)
# --------------------------------------------------------------------------


def _rss_worker_oneshot(n: int, latency: float) -> None:
    import resource

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        model = c.sync_model(c.sync_create(latency=latency), cache=cache)
        settings = c.settings()
        texts = c.make_texts(n, seed=7)
        t0 = time.perf_counter()
        resp = model.get_embeddings(texts, settings)
        wall = time.perf_counter() - t0
        assert len(resp.output) == n

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"wall_s": wall, "max_rss_bytes": peak}))


def _rss_worker_generator(n: int, latency: float, chunk_size: int) -> None:
    import resource

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        model = c.sync_model(c.sync_create(latency=latency), cache=cache)
        settings = c.settings()
        texts = c.make_texts(n, seed=7)
        t0 = time.perf_counter()
        total = 0
        for resp in model.get_embeddings_generator(texts, settings, chunk_size=chunk_size):
            total += len(resp.output)  # each chunk's response goes out of scope right after
        wall = time.perf_counter() - t0
        assert total == n

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"wall_s": wall, "max_rss_bytes": peak}))


def bench_generator_vs_oneshot(quick: bool) -> dict:
    c.banner("7. get_embeddings_generator vs one big get_embeddings (wall + peak RSS)")

    n = 512 if quick else 4096
    latency = 0.01
    chunk_size = 64 if quick else 256
    repeats = 1 if quick else 3

    def spawn(mode: str) -> list[dict]:
        runs = []
        for _ in range(repeats):
            cmd = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--rss-worker",
                mode,
                "--rss-n",
                str(n),
                "--rss-latency",
                str(latency),
                "--rss-chunk-size",
                str(chunk_size),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
            runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
        return runs

    oneshot_runs = spawn("oneshot")
    generator_runs = spawn("generator")

    oneshot_wall = statistics.median(r["wall_s"] for r in oneshot_runs)
    oneshot_rss = statistics.median(r["max_rss_bytes"] for r in oneshot_runs)
    generator_wall = statistics.median(r["wall_s"] for r in generator_runs)
    generator_rss = statistics.median(r["max_rss_bytes"] for r in generator_runs)

    rows = [
        {"mode": "one big get_embeddings()", "wall_s": oneshot_wall, "peak_rss_mb": oneshot_rss / 1e6},
        {
            "mode": f"get_embeddings_generator(chunk_size={chunk_size})",
            "wall_s": generator_wall,
            "peak_rss_mb": generator_rss / 1e6,
        },
    ]
    print(c.table(rows, ["mode", "wall_s", "peak_rss_mb"]))

    rss_delta_pct = 100 * (generator_rss - oneshot_rss) / oneshot_rss
    wall_delta_pct = 100 * (generator_wall - oneshot_wall) / oneshot_wall
    verdict = (
        f"generator uses {rss_delta_pct:+.1f}% peak RSS and {wall_delta_pct:+.1f}% wall-clock vs one big call. "
        + (
            "More, smaller provider round-trips (one per chunk instead of one per max_batch_size/"
            "max_tokens_a_request-worth of texts) costs wall-clock; whether it's worth it depends on "
            "whether the caller actually keeps memory bounded by not retaining each chunk's response, "
            "which this measurement does (each ModelResponse is dropped once its length is read)."
        )
    )
    print(f"-> {verdict}")

    return {
        "params": {"n_texts": n, "latency_s": latency, "chunk_size": chunk_size, "repeats": repeats},
        "oneshot": {"wall_s": oneshot_wall, "peak_rss_bytes": oneshot_rss, "runs": oneshot_runs},
        "generator": {"wall_s": generator_wall, "peak_rss_bytes": generator_rss, "runs": generator_runs},
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny params, <45s, smoke test")
    parser.add_argument("--rss-worker", choices=["oneshot", "generator"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rss-n", type=int, default=4096, help=argparse.SUPPRESS)
    parser.add_argument("--rss-latency", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--rss-chunk-size", type=int, default=256, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rss_worker == "oneshot":
        _rss_worker_oneshot(args.rss_n, args.rss_latency)
        return
    if args.rss_worker == "generator":
        _rss_worker_generator(args.rss_n, args.rss_latency, args.rss_chunk_size)
        return

    t_start = time.perf_counter()
    quick = args.quick

    payload: dict[str, typing.Any] = {
        "name": "sync",
        "quick": quick,
        "machine": {
            "python": sys.version,
            "platform": sys.platform,
            "cpu_cores": CPU_CORES,
        },
        "thread_counts": THREAD_COUNTS,
        "experiments": {},
    }

    payload["experiments"]["all_hit"] = bench_all_hit(quick)
    payload["experiments"]["all_miss"] = bench_all_miss(quick)
    payload["experiments"]["no_cache"] = bench_no_cache(quick)
    payload["experiments"]["breakdown"] = bench_breakdown(quick)
    payload["experiments"]["batching"] = bench_batching(quick)
    payload["experiments"]["correctness"] = bench_correctness(quick)
    payload["experiments"]["generator_vs_oneshot"] = bench_generator_vs_oneshot(quick)

    total_wall = time.perf_counter() - t_start
    payload["total_wall_s"] = total_wall

    out_path = c.save("sync", payload)
    c.banner(f"Done in {total_wall:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
