"""The workload the other benchmarks skipped: a partly-warm cache.

Everything else here measures 0% or 100% hit rate. Production sits between:
a corpus grows by a few documents, a query overlaps yesterday's, a re-index
touches a subset. So sweep the hit rate and answer three things.

  1. What does the cost curve actually look like? Per-item work splits in two:
     key generation, `cache.get` and `validate_cached_embedding` run over
     *every* text, while tiktoken, the provider call and the cache write run
     only over the missing ones. So the floor is set by the total and the
     slope by the misses.

  2. Do the 0.6.0 fixes still pay off in the middle? The scope-digest hoist
     scales with the total; the write transaction scales with the misses.
     Neither was measured against a partly-warm cache.

  3. Does anything go non-linear — a hit rate where a batch boundary, the
     dedup pass or the executor behaves worse than at either extreme?
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import random
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

DIM = 1536
RATES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]


def _warm_subset(model, texts: list[str], rate: float, settings) -> int:
    """Pre-populate `rate` of `texts`, chosen at random rather than as a prefix.

    A prefix would land every miss in the same trailing batch; scattering them
    is what a real incremental corpus looks like.
    """
    if rate <= 0:
        return 0
    rng = random.Random(1234)
    subset = rng.sample(texts, int(len(texts) * rate))
    if subset:
        model.get_embeddings(subset, settings)
    return len(subset)


def sync_sweep(n: int, latency: float, repeats: int) -> list[dict]:
    settings = c.settings(dimensions=DIM)
    rows = []
    for rate in RATES:
        walls, hits, misses = [], [], []
        for run in range(repeats):
            texts = c.make_texts(n, seed=7000 + run)
            with tempfile.TemporaryDirectory() as d:
                cache = c.fresh_cache(d)
                model = c.sync_model(c.sync_create(latency=latency), cache=cache)
                warmed = _warm_subset(model, texts, rate, settings)

                t0 = time.perf_counter()
                response = model.get_embeddings(texts, settings)
                walls.append(time.perf_counter() - t0)
                hits.append(response.usage.cache_hits)
                misses.append(n - response.usage.cache_hits)
                assert response.usage.cache_hits == warmed, "warm-up did not produce the intended hit rate"
                cache.close()
        wall = statistics.median(walls)
        miss = statistics.median(misses)
        rows.append(
            {
                "hit_rate": f"{rate:.0%}",
                "hits": statistics.median(hits),
                "misses": miss,
                "wall_ms": wall * 1000,
                "us_per_text": wall * 1e6 / n,
                "us_per_miss": (wall * 1e6 / miss) if miss else float("nan"),
            }
        )
    return rows


async def async_sweep(n: int, latency: float, repeats: int) -> list[dict]:
    settings = c.settings(dimensions=DIM)
    rows = []
    for rate in RATES:
        walls, lags = [], []
        for run in range(repeats):
            texts = c.make_texts(n, seed=7000 + run)
            with tempfile.TemporaryDirectory() as d:
                cache = c.fresh_cache(d)
                model = c.async_model(c.async_create(latency=latency), cache=cache)
                if rate > 0:
                    rng = random.Random(1234)
                    subset = rng.sample(texts, int(n * rate))
                    if subset:
                        await model.get_embeddings(subset, settings)

                async with c.LoopLagProbe() as probe:
                    t0 = time.perf_counter()
                    await model.get_embeddings(texts, settings)
                    walls.append(time.perf_counter() - t0)
                lags.append(probe.summary()["max"] * 1000)

                await model.aclose()
                cache.close()
        rows.append(
            {
                "hit_rate": f"{rate:.0%}",
                "wall_ms": statistics.median(walls) * 1000,
                "max_loop_lag_ms": statistics.median(lags),
            }
        )
    return rows


def component_model(n: int, repeats: int) -> list[dict]:
    """Time the two halves of the per-item cost directly.

    `whole_input` runs over every text however warm the cache is;
    `misses_only` runs over what is left. Together they predict the curve.
    """
    settings = c.settings(dimensions=DIM)
    texts = c.make_texts(n, seed=4242)
    rows = []

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        model = c.sync_model(c.sync_create(), cache=cache)
        model.get_embeddings(texts, settings)  # fully warm
        keys = model._cache_keys_for(texts, settings)

        keygen = c.timeit(lambda: model._cache_keys_for(texts, settings), repeats=repeats)
        reads = c.timeit(lambda: [model._cache_get(k, DIM) for k in keys], repeats=repeats)
        tokenise = c.timeit(lambda: model._prepare_batches(texts), repeats=repeats)
        cache.close()

    rows.append(
        {
            "component": "key generation + cache read + validate",
            "runs_over": "every text",
            "us_per_text": (keygen["median"] + reads["median"]) * 1e6 / n,
        }
    )
    rows.append(
        {
            "component": "tiktoken (_prepare_batches)",
            "runs_over": "misses only",
            "us_per_text": tokenise["median"] * 1e6 / n,
        }
    )
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick

    n = 256 if q else 4096
    repeats = 1 if q else 3
    payload: dict = {"n_texts": n, "repeats": repeats, "rates": RATES, "quick": q}

    c.banner(f"Per-item cost model ({n} texts)")
    payload["components"] = component_model(n, repeats=2 if q else 5)
    print(c.table(payload["components"], ["component", "runs_over", "us_per_text"]))

    c.banner(f"Sync, hit-rate sweep, {n} texts, no provider latency")
    payload["sync_no_latency"] = sync_sweep(n, latency=0.0, repeats=repeats)
    print(c.table(payload["sync_no_latency"], ["hit_rate", "hits", "misses", "wall_ms", "us_per_text", "us_per_miss"]))

    c.banner(f"Sync, hit-rate sweep, {n} texts, 20 ms per provider batch")
    payload["sync_latency"] = sync_sweep(n, latency=0.02, repeats=repeats)
    print(c.table(payload["sync_latency"], ["hit_rate", "hits", "misses", "wall_ms", "us_per_text"]))

    c.banner(f"Async, hit-rate sweep, {n} texts, 20 ms per provider batch")
    payload["async_latency"] = await async_sweep(n, latency=0.02, repeats=repeats)
    print(c.table(payload["async_latency"], ["hit_rate", "wall_ms", "max_loop_lag_ms"]))

    # Where is the cheapest place to be? Report cost per *useful* unit of work.
    baseline = next(r for r in payload["sync_no_latency"] if r["hit_rate"] == "0%")
    full = next(r for r in payload["sync_no_latency"] if r["hit_rate"] == "100%")
    print(
        f"\nFloor (100% hits): {full['wall_ms']:.1f} ms — the price of asking, paid whatever the hit rate."
        f"\nCeiling (0% hits): {baseline['wall_ms']:.1f} ms."
        f"\nA 90% hit rate costs "
        f"{next(r for r in payload['sync_no_latency'] if r['hit_rate'] == '90%')['wall_ms']:.1f} ms."
    )

    out = c.save("diag_partial_hit", payload)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
