"""Benchmark: the diskcache layer in isolation.

Scope is deliberately narrow — this never touches OpenAIEmbeddingsModel or
AsyncOpenAIEmbeddingsModel. It measures the primitives those classes build
on: a raw `diskcache.Cache`, `validate_cached_embedding`, `generate_cache_key`
and `cache_scope_digest`. The question is whether the cache layer is a
bottleneck, and if so, where.

Run from the repo root:
    python tmp/bench/bench_cache.py --quick   # smoke test, <30s
    python tmp/bench/bench_cache.py           # full run, ~<6 min
"""

from __future__ import annotations

import base64
import itertools
import pathlib
import random
import statistics
import sys
import tempfile
import threading
import time

import diskcache
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# Running `python tmp/bench/bench_cache.py` puts only the script's own
# directory on sys.path (not the cwd), so common.py's `import
# openai_embeddings_model` would fail unless the repo root is added too.
sys.path.insert(0, str(_HERE.parent.parent))
import common as c  # noqa: E402

from openai_embeddings_model import cache_scope_digest, generate_cache_key, validate_cached_embedding  # noqa: E402

QUICK = "--quick" in sys.argv
DIM = c.DIM  # 1536, matches text-embedding-3-small

if QUICK:
    READ_SIZES = [200, 1_000, 3_000]
    READ_OPS = 300
    WRITE_N = 300
    VALIDATE_BATCH = 500
    KEYGEN_BATCH = 500
    THREAD_COUNTS = [1, 2, 4]
    OPS_PER_THREAD = 150
    MIXED_THREADS = 4
    MIXED_OPS_PER_THREAD = 150
    VALUE_SIZE_N = 300
    VALUE_SIZE_WRITE_N = 150
    REPEATS = 3
    CONCURRENCY_TRIALS = 2
else:
    READ_SIZES = [1_000, 10_000, 100_000]
    READ_OPS = 2_000
    WRITE_N = 5_000
    VALIDATE_BATCH = 5_000
    KEYGEN_BATCH = 5_000
    THREAD_COUNTS = [1, 2, 4, 8, 16]
    OPS_PER_THREAD = 2_000
    MIXED_THREADS = 8
    MIXED_OPS_PER_THREAD = 1_500
    VALUE_SIZE_N = 2_000
    VALUE_SIZE_WRITE_N = 1_000
    REPEATS = 5
    CONCURRENCY_TRIALS = 3

REQUEST_TEXT_COUNT = 2048  # a full MAX_BATCH_SIZE request, for the key-gen waste calc


# --------------------------------------------------------------------------
# 1. read throughput, single thread, vs cache size
# --------------------------------------------------------------------------


def exp1_read_throughput() -> dict:
    c.banner("1. Read throughput (single thread) vs cache size")
    rows = []
    for size in READ_SIZES:
        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d)
            value = c.b64_vector(DIM)
            keys = [f"k:{i}" for i in range(size)]
            c.warm_cache(cache, keys, value)

            rng = random.Random(42)
            n_ops = min(READ_OPS, size)
            hit_sample = [rng.choice(keys) for _ in range(n_ops)]
            miss_sample = [f"miss:{i}" for i in range(n_ops)]

            def run_hits(hit_sample=hit_sample, cache=cache):
                for k in hit_sample:
                    cache.get(k)

            def run_misses(miss_sample=miss_sample, cache=cache):
                for k in miss_sample:
                    cache.get(k)

            hit_stats = c.timeit(run_hits, repeats=REPEATS, warmup=1)
            miss_stats = c.timeit(run_misses, repeats=REPEATS, warmup=1)
            hit_ops = n_ops / hit_stats["median"]
            miss_ops = n_ops / miss_stats["median"]

            rows.append(
                {
                    "cache_size": size,
                    "hit_ops_per_sec": hit_ops,
                    "miss_ops_per_sec": miss_ops,
                    "hit_us": hit_stats["median"] / n_ops * 1e6,
                    "miss_us": miss_stats["median"] / n_ops * 1e6,
                }
            )
            cache.close()

    print(c.table(rows, ["cache_size", "hit_ops_per_sec", "miss_ops_per_sec", "hit_us", "miss_us"]))
    return {"params": {"read_sizes": READ_SIZES, "ops_per_trial": READ_OPS, "repeats": REPEATS}, "rows": rows}


# --------------------------------------------------------------------------
# 2. write throughput: naive loop vs transact()
# --------------------------------------------------------------------------


def exp2_write_throughput() -> dict:
    c.banner("2. Write throughput: naive set() loop vs transact()")
    rows = []
    value = c.b64_vector(DIM)

    for label, use_transact in (("naive", False), ("transact", True)):
        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d)
            counter = itertools.count()

            def run(cache=cache, counter=counter, use_transact=use_transact):
                base = next(counter) * WRITE_N
                keys = [f"w:{base + i}" for i in range(WRITE_N)]
                if use_transact:
                    with cache.transact():
                        for k in keys:
                            cache.set(k, value)
                else:
                    for k in keys:
                        cache.set(k, value)

            stats = c.timeit(run, repeats=REPEATS, warmup=1)
            ops = WRITE_N / stats["median"]
            rows.append({"mode": label, "n": WRITE_N, "median_s": stats["median"], "ops_per_sec": ops})
            cache.close()

    speedup = rows[1]["ops_per_sec"] / rows[0]["ops_per_sec"]
    print(c.table(rows, ["mode", "n", "median_s", "ops_per_sec"]))
    print(f"\ntransact() speedup over naive loop: {speedup:.2f}x")
    return {
        "params": {"n": WRITE_N, "repeats": REPEATS},
        "rows": rows,
        "transact_speedup_over_naive": speedup,
    }


# --------------------------------------------------------------------------
# 3. validate_cached_embedding cost breakdown
# --------------------------------------------------------------------------


def exp3_validation_cost() -> dict:
    c.banner("3. validate_cached_embedding cost breakdown")
    value = c.b64_vector(DIM)
    batch = VALIDATE_BATCH

    def run_full():
        for _ in range(batch):
            validate_cached_embedding("k", value, DIM)

    def run_decode():
        for _ in range(batch):
            base64.b64decode(value, validate=True)

    raw = base64.b64decode(value, validate=True)

    def run_isfinite():
        for _ in range(batch):
            np.isfinite(np.frombuffer(raw, dtype=np.float32)).all()

    full_stats = c.timeit(run_full, repeats=REPEATS, warmup=1)
    decode_stats = c.timeit(run_decode, repeats=REPEATS, warmup=1)
    isfinite_stats = c.timeit(run_isfinite, repeats=REPEATS, warmup=1)

    full_us = full_stats["median"] / batch * 1e6
    decode_us = decode_stats["median"] / batch * 1e6
    isfinite_us = isfinite_stats["median"] / batch * 1e6
    rest_us = max(0.0, full_us - decode_us - isfinite_us)

    # Baseline: a raw cache.get() with no validation, single entry so the
    # read cost is not itself a function of cache size (see experiment 1).
    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        cache.set("only", value)

        def run_get():
            for _ in range(batch):
                cache.get("only")

        get_stats = c.timeit(run_get, repeats=REPEATS, warmup=1)
        get_us = get_stats["median"] / batch * 1e6
        cache.close()

    rows = [
        {"component": "base64 decode", "us_per_entry": decode_us, "pct_of_validate": decode_us / full_us * 100},
        {
            "component": "np.isfinite(...).all()",
            "us_per_entry": isfinite_us,
            "pct_of_validate": isfinite_us / full_us * 100,
        },
        {
            "component": "rest (type/len/dim checks)",
            "us_per_entry": rest_us,
            "pct_of_validate": rest_us / full_us * 100,
        },
        {"component": "TOTAL validate_cached_embedding", "us_per_entry": full_us, "pct_of_validate": 100.0},
    ]
    print(c.table(rows, ["component", "us_per_entry", "pct_of_validate"]))
    print(f"\ncache.get() alone (1-entry cache): {get_us:.2f} us/entry")
    print(f"validate as % of a bare get():           {full_us / get_us * 100:.1f}%")
    print(f"validate as % of (get + validate) total: {full_us / (full_us + get_us) * 100:.1f}%")

    return {
        "params": {"batch": batch, "dim": DIM, "repeats": REPEATS},
        "rows": rows,
        "cache_get_us": get_us,
        "validate_pct_of_get": full_us / get_us * 100,
        "validate_pct_of_get_plus_validate": full_us / (full_us + get_us) * 100,
    }


# --------------------------------------------------------------------------
# 4. generate_cache_key / cache_scope_digest cost
# --------------------------------------------------------------------------


def exp4_keygen_cost() -> dict:
    c.banner("4. generate_cache_key / cache_scope_digest cost")
    text = c.make_texts(1)[0]
    provider = "http://localhost:1"  # matches str(client.base_url) in the library: truthy, so
    # cache_scope_digest always takes the hashing path, never the "default" fast path.
    batch = KEYGEN_BATCH

    def run_keygen():
        for _ in range(batch):
            generate_cache_key(model=c.MODEL, dimensions=DIM, text=text, provider=provider, extra_body=None)

    def run_scope():
        for _ in range(batch):
            cache_scope_digest(provider, None)

    keygen_stats = c.timeit(run_keygen, repeats=REPEATS, warmup=1)
    scope_stats = c.timeit(run_scope, repeats=REPEATS, warmup=1)
    keygen_us = keygen_stats["median"] / batch * 1e6
    scope_us = scope_stats["median"] / batch * 1e6

    n = REQUEST_TEXT_COUNT
    total_keygen_ms = n * keygen_us / 1000
    # pre-0.6.0 key generation called generate_cache_key once per text; cache_scope_digest is
    # recomputed n times even though provider/extra_body are constant for the whole
    # request. Waste = the (n - 1) redundant scope-digest computations.
    waste_ms = (n - 1) * scope_us / 1000
    waste_pct_of_keygen = waste_ms / total_keygen_ms * 100

    rows = [
        {"function": "generate_cache_key", "us_per_call": keygen_us},
        {"function": "cache_scope_digest", "us_per_call": scope_us},
        {"function": "scope_digest as % of one generate_cache_key call", "us_per_call": scope_us / keygen_us * 100},
    ]
    print(c.table(rows, ["function", "us_per_call"]))
    print(f"\nFor a {n}-text request (a full MAX_BATCH_SIZE):")
    print(f"  total generate_cache_key time:        {total_keygen_ms:.2f} ms")
    print(f"  redundant scope-digest recomputation: {waste_ms:.2f} ms ({waste_pct_of_keygen:.1f}% of the above)")

    return {
        "params": {"batch": batch, "provider": provider, "request_text_count": n, "repeats": REPEATS},
        "generate_cache_key_us": keygen_us,
        "cache_scope_digest_us": scope_us,
        "scope_digest_pct_of_keygen": scope_us / keygen_us * 100,
        "request_total_keygen_ms": total_keygen_ms,
        "request_redundant_scope_digest_ms": waste_ms,
        "request_redundant_scope_digest_pct": waste_pct_of_keygen,
    }


# --------------------------------------------------------------------------
# 5 & 6 & 7: concurrency helpers
# --------------------------------------------------------------------------


def _run_concurrent_reads(cache: diskcache.Cache, keys: list[str], n_threads: int, ops_per_thread: int) -> float:
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        rng = random.Random(threading.get_ident())
        barrier.wait()
        for _ in range(ops_per_thread):
            cache.get(keys[rng.randrange(len(keys))])

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return n_threads * ops_per_thread / elapsed


def exp5_concurrent_readers() -> dict:
    c.banner("5. Concurrent readers (hits only) on one shared Cache")
    rows = []
    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        value = c.b64_vector(DIM)
        keys = [f"k:{i}" for i in range(5_000)]
        c.warm_cache(cache, keys, value)

        baseline = None
        for n_threads in THREAD_COUNTS:
            trials = [_run_concurrent_reads(cache, keys, n_threads, OPS_PER_THREAD) for _ in range(CONCURRENCY_TRIALS)]
            ops = statistics.median(trials)
            if baseline is None:
                baseline = ops
            rows.append({"threads": n_threads, "ops_per_sec": ops, "scaling_vs_1_thread": ops / baseline})
        cache.close()

    print(c.table(rows, ["threads", "ops_per_sec", "scaling_vs_1_thread"]))
    ideal = rows[-1]["threads"]
    actual = rows[-1]["scaling_vs_1_thread"]
    verdict = "reads scale roughly linearly" if actual > ideal * 0.6 else "reads do NOT scale — contention dominates"
    print(f"\nAt {rows[-1]['threads']} threads: {actual:.2f}x vs 1 thread (ideal would be {ideal}x). {verdict}")

    return {
        "params": {"thread_counts": THREAD_COUNTS, "ops_per_thread": OPS_PER_THREAD, "trials": CONCURRENCY_TRIALS},
        "rows": rows,
        "verdict": verdict,
    }


def _run_concurrent_writes(
    cache: diskcache.Cache, n_threads: int, ops_per_thread: int, run_id: int
) -> tuple[int, int, float]:
    value = c.b64_vector(DIM)
    barrier = threading.Barrier(n_threads)
    results: list[tuple[int, int]] = [(0, 0)] * n_threads

    def worker(idx: int) -> None:
        success = 0
        timeouts = 0
        barrier.wait()
        for i in range(ops_per_thread):
            key = f"cw:{run_id}:{idx}:{i}"
            try:
                cache.set(key, value)
                success += 1
            except diskcache.Timeout:
                timeouts += 1
        results[idx] = (success, timeouts)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    total_success = sum(r[0] for r in results)
    total_timeouts = sum(r[1] for r in results)
    return total_success, total_timeouts, elapsed


def exp6_concurrent_writers() -> dict:
    c.banner("6. Concurrent writers (distinct keys) on one shared Cache")
    rows = []
    run_id = itertools.count()
    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)

        baseline = None
        for n_threads in THREAD_COUNTS:
            trial_ops = []
            total_timeouts = 0
            for _ in range(CONCURRENCY_TRIALS):
                success, timeouts, elapsed = _run_concurrent_writes(cache, n_threads, OPS_PER_THREAD, next(run_id))
                trial_ops.append(success / elapsed)
                total_timeouts += timeouts
            ops = statistics.median(trial_ops)
            if baseline is None:
                baseline = ops
            rows.append(
                {
                    "threads": n_threads,
                    "ops_per_sec": ops,
                    "scaling_vs_1_thread": ops / baseline,
                    "timeouts_seen": total_timeouts,
                }
            )
        cache.close()

    print(c.table(rows, ["threads", "ops_per_sec", "scaling_vs_1_thread", "timeouts_seen"]))
    ideal = rows[-1]["threads"]
    actual = rows[-1]["scaling_vs_1_thread"]
    verdict = "writes scale" if actual > ideal * 0.6 else "writes serialise on the sqlite write lock — do NOT scale"
    any_timeouts = any(r["timeouts_seen"] for r in rows)
    print(f"\nAt {rows[-1]['threads']} threads: {actual:.2f}x vs 1 thread (ideal would be {ideal}x). {verdict}")
    print(f"diskcache.Timeout raised anywhere in the run: {any_timeouts}")

    return {
        "params": {"thread_counts": THREAD_COUNTS, "ops_per_thread": OPS_PER_THREAD, "trials": CONCURRENCY_TRIALS},
        "rows": rows,
        "verdict": verdict,
        "any_timeouts": any_timeouts,
    }


# --------------------------------------------------------------------------
# 7. mixed workload
# --------------------------------------------------------------------------


def _run_mixed(
    cache: diskcache.Cache, keys: list[str], n_threads: int, ops_per_thread: int, write_frac: float, run_id: int
) -> float:
    value = c.b64_vector(DIM)
    barrier = threading.Barrier(n_threads)

    def worker(idx: int) -> None:
        rng = random.Random(run_id * 1_000 + idx)
        barrier.wait()
        for i in range(ops_per_thread):
            if rng.random() < write_frac:
                cache.set(f"mx:{run_id}:{idx}:{i}", value)
            else:
                cache.get(keys[rng.randrange(len(keys))])

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return n_threads * ops_per_thread / elapsed


def exp7_mixed_workload() -> dict:
    c.banner("7. Mixed workload: 90% read / 10% write vs 100% read")
    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        value = c.b64_vector(DIM)
        keys = [f"k:{i}" for i in range(5_000)]
        c.warm_cache(cache, keys, value)

        run_id = itertools.count()
        mixed_trials = [
            _run_mixed(cache, keys, MIXED_THREADS, MIXED_OPS_PER_THREAD, 0.10, next(run_id))
            for _ in range(CONCURRENCY_TRIALS)
        ]
        pure_trials = [
            _run_mixed(cache, keys, MIXED_THREADS, MIXED_OPS_PER_THREAD, 0.0, next(run_id))
            for _ in range(CONCURRENCY_TRIALS)
        ]
        cache.close()

    mixed_ops = statistics.median(mixed_trials)
    pure_ops = statistics.median(pure_trials)
    rows = [
        {"workload": "90% read / 10% write", "threads": MIXED_THREADS, "ops_per_sec": mixed_ops},
        {"workload": "100% read", "threads": MIXED_THREADS, "ops_per_sec": pure_ops},
    ]
    print(c.table(rows, ["workload", "threads", "ops_per_sec"]))
    ratio = mixed_ops / pure_ops
    print(f"\nmixed / pure-read ratio: {ratio:.2f}")

    return {
        "params": {"threads": MIXED_THREADS, "ops_per_thread": MIXED_OPS_PER_THREAD, "trials": CONCURRENCY_TRIALS},
        "rows": rows,
        "mixed_over_pure_read_ratio": ratio,
    }


# --------------------------------------------------------------------------
# 8. value size sensitivity
# --------------------------------------------------------------------------


def exp8_value_size_sensitivity() -> dict:
    c.banner("8. Value size sensitivity (dims 256/768/1536/3072)")
    rows = []
    for dim in (256, 768, 1536, 3072):
        value = c.b64_vector(dim)
        b64_bytes = len(value.encode())

        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d)
            threshold = cache.disk_min_file_size
            inline = b64_bytes < threshold

            keys = [f"k:{i}" for i in range(VALUE_SIZE_N)]
            c.warm_cache(cache, keys, value)

            rng = random.Random(7)
            hit_sample = [rng.choice(keys) for _ in range(VALUE_SIZE_N)]

            def run_read(hit_sample=hit_sample, cache=cache):
                for k in hit_sample:
                    cache.get(k)

            read_stats = c.timeit(run_read, repeats=REPEATS, warmup=1)
            read_ops = VALUE_SIZE_N / read_stats["median"]

            counter = itertools.count()

            def run_write(cache=cache, counter=counter, value=value):
                base = next(counter) * VALUE_SIZE_WRITE_N
                wkeys = [f"w:{base + i}" for i in range(VALUE_SIZE_WRITE_N)]
                for k in wkeys:
                    cache.set(k, value)

            write_stats = c.timeit(run_write, repeats=REPEATS, warmup=1)
            write_ops = VALUE_SIZE_WRITE_N / write_stats["median"]

            rows.append(
                {
                    "dim": dim,
                    "b64_bytes": b64_bytes,
                    "disk_min_file_size": threshold,
                    "storage": "inline (sqlite)" if inline else "spilled (file)",
                    "read_ops_per_sec": read_ops,
                    "write_ops_per_sec": write_ops,
                }
            )
            cache.close()

    print(c.table(rows, ["dim", "b64_bytes", "storage", "read_ops_per_sec", "write_ops_per_sec"]))
    all_inline = all(r["storage"] == "inline (sqlite)" for r in rows)
    print(
        f"\nAll four dims fit under disk_min_file_size ({rows[0]['disk_min_file_size']} bytes): {all_inline}. "
        "Every realistic embedding value (even 3072-dim text-embedding-3-large) is stored inline in "
        "sqlite by default — the file-spill path never triggers for this library's payloads."
        if all_inline
        else "Some sizes spilled to a separate file — see per-row storage column."
    )

    return {
        "params": {"n": VALUE_SIZE_N, "write_n": VALUE_SIZE_WRITE_N, "repeats": REPEATS},
        "rows": rows,
        "all_inline": all_inline,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    mode = "quick" if QUICK else "full"
    c.banner(f"bench_cache.py — diskcache layer isolation ({mode} mode)")

    payload = {
        "meta": {
            "mode": mode,
            "dim": DIM,
            "model": c.MODEL,
            "diskcache_version": diskcache.__version__,
            "read_sizes": READ_SIZES,
            "thread_counts": THREAD_COUNTS,
        },
        "read_throughput": exp1_read_throughput(),
        "write_throughput": exp2_write_throughput(),
        "validation_cost": exp3_validation_cost(),
        "keygen_cost": exp4_keygen_cost(),
        "concurrent_readers": exp5_concurrent_readers(),
        "concurrent_writers": exp6_concurrent_writers(),
        "mixed_workload": exp7_mixed_workload(),
        "value_size_sensitivity": exp8_value_size_sensitivity(),
    }

    elapsed = time.perf_counter() - t0
    payload["meta"]["elapsed_s"] = elapsed
    path = c.save("cache", payload)

    c.banner(f"Done in {elapsed:.1f}s — results saved to {path}")


if __name__ == "__main__":
    main()
