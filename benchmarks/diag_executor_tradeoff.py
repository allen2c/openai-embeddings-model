"""Is `executor_max_workers=1` a safe recommendation, or only half a story?

bench_async.py found that one executor thread beats the default 14 by 77% on
concurrent all-hit calls. But the executor does two different jobs: batched
cache I/O (short, sqlite, collapses under threads) and `_prepare_batches`
(tiktoken, ~25 us/text, CPU-bound, ~115 ms for a 4096-text call). With a
single worker, concurrent callers queue behind each other's tokenisation.

Sweep both workloads over the same worker counts so the recommendation can be
stated with its cost attached.

`blocking_cache_case` then asks the question that decides whether
`executor_max_workers` should remain a knob at all. Every argument for one
worker rests on the cache being local and fast, so that `cache.get` is
GIL-bound rather than I/O-bound. Nothing stops a caller passing a cache-like
object backed by a network filesystem, Redis, or S3 — and there, threads wait
on real I/O and genuinely overlap. If a blocking backend flips the answer, the
parameter has a real use and must stay.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

WORKERS = [1, 2, 4, 8, 14, 32]
CONCURRENCY = 16
N_TEXTS = 512
TRIALS = 3


async def run_case(label: str, *, warm: bool, latency: float, n_texts: int = N_TEXTS) -> list[dict]:
    rows = []
    for workers in WORKERS:
        trials = []
        for _ in range(TRIALS):
            with tempfile.TemporaryDirectory() as d:
                cache = c.fresh_cache(d)
                model = c.async_model(c.async_create(latency=latency), cache=cache, executor_max_workers=workers)
                settings = c.settings(dimensions=c.DIM)
                # Distinct texts per concurrent caller, stable across trials so
                # the warm and cold variants see the same work.
                batches = [c.make_texts(n_texts, seed=1000 + i) for i in range(CONCURRENCY)]
                if warm:
                    for texts in batches:
                        await model.get_embeddings(texts, settings)

                t0 = time.perf_counter()
                await asyncio.gather(*(model.get_embeddings(texts, settings) for texts in batches))
                trials.append(time.perf_counter() - t0)

                await model.aclose()
                cache.close()
        rows.append({"workers": workers, "wall_s": statistics.median(trials)})

    best = min(rows, key=lambda r: r["wall_s"])
    for row in rows:
        row["vs_best"] = row["wall_s"] / best["wall_s"]
    print(f"\n--- {label} ({CONCURRENCY} concurrent x {n_texts} texts) ---")
    print(c.table(rows, ["workers", "wall_s", "vs_best"]))
    print(f"best: workers={best['workers']}")
    return rows


class BlockingCache:
    """A cache whose reads and writes wait on I/O, the way a remote one would.

    `time.sleep` is the honest stand-in: like a socket read it releases the
    GIL, so several of these really can be in flight at once. Everything else
    delegates to a real `diskcache.Cache`, so results stay correct.
    """

    def __init__(self, inner, latency: float) -> None:
        self._inner = inner
        self._latency = latency

    def get(self, key, *args, **kwargs):
        time.sleep(self._latency)
        return self._inner.get(key, *args, **kwargs)

    def set(self, key, value, *args, **kwargs):
        time.sleep(self._latency)
        return self._inner.set(key, value, *args, **kwargs)

    def close(self):
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def blocking_cache_case(latencies: list[float], workers: list[int], concurrency: int, n_texts: int) -> list[dict]:
    """Does a cache that blocks on I/O make extra workers worth having?"""
    rows = []
    batches = [c.make_texts(n_texts, seed=6000 + i) for i in range(concurrency)]
    settings = c.settings(dimensions=c.DIM)

    for latency in latencies:
        best_wall = None
        for w in workers:
            walls = []
            for _ in range(2):
                with tempfile.TemporaryDirectory() as d:
                    inner = c.fresh_cache(d)
                    # Warm through the real cache, so the timed run is all hits
                    # and the only cost measured is the simulated I/O.
                    warm_model = c.async_model(c.async_create(), cache=inner, executor_max_workers=1)
                    for texts in batches:
                        await warm_model.get_embeddings(texts, settings)
                    await warm_model.aclose()

                    slow = BlockingCache(inner, latency)
                    model = c.async_model(c.async_create(), cache=slow, executor_max_workers=w)
                    t0 = time.perf_counter()
                    await asyncio.gather(*(model.get_embeddings(t, settings) for t in batches))
                    walls.append(time.perf_counter() - t0)
                    await model.aclose()
                    inner.close()
            wall = statistics.median(walls)
            best_wall = wall if best_wall is None else min(best_wall, wall)
            rows.append({"cache_latency_ms": latency * 1000, "workers": w, "wall_s": wall})

    for row in rows:
        peers = [r["wall_s"] for r in rows if r["cache_latency_ms"] == row["cache_latency_ms"]]
        row["vs_best"] = row["wall_s"] / min(peers)

    print(f"\n--- blocking cache backend ({concurrency} concurrent x {n_texts} texts, all hits) ---")
    print(c.table(rows, ["cache_latency_ms", "workers", "wall_s", "vs_best"]))
    for latency in latencies:
        group = [r for r in rows if r["cache_latency_ms"] == latency * 1000]
        winner = min(group, key=lambda r: r["wall_s"])
        one = next(r for r in group if r["workers"] == 1)
        print(
            f"  cache latency {latency * 1000:>5.1f} ms -> best workers={winner['workers']}"
            f" ({one['wall_s'] / winner['wall_s']:.1f}x faster than one worker)"
        )
    return rows


async def main() -> None:
    payload: dict = {"workers": WORKERS, "concurrency": CONCURRENCY, "n_texts": N_TEXTS, "trials": TRIALS}

    payload["all_hit"] = await run_case("all-hit (executor does cache reads only)", warm=True, latency=0.0)
    payload["all_miss_no_latency"] = await run_case(
        "all-miss, 0 ms provider (executor does tiktoken + cache writes)", warm=False, latency=0.0
    )
    payload["all_miss_20ms"] = await run_case(
        "all-miss, 20 ms provider (realistic: network dominates)", warm=False, latency=0.02
    )

    payload["blocking_cache"] = await blocking_cache_case(
        latencies=[0.0, 0.0005, 0.002, 0.010],
        workers=[1, 2, 4, 8, 14],
        concurrency=16,
        n_texts=64,
    )

    out = c.RESULTS / "diag_executor_tradeoff.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
