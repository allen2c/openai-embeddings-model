"""Mechanism checks for the two fixes: `executor_max_workers` and `transact()`.

Both fixes were found by measurement before they were understood. Research
then produced four falsifiable claims about *why* they work. Each is tested
here, because a fix whose mechanism is wrong is a fix that will surprise us
later.

  A. Offloading cache reads to a thread is not pointless. Releasing the GIL
     inside `sqlite3_step` lets other OS threads run, but a coroutine only
     yields at an `await`, so doing the reads inline on the loop would
     serialise every concurrent caller. Predicts: inline is WORSE than one
     executor worker.

  B. One worker means one FIFO queue shared by cache I/O and tiktoken. A
     115 ms `_prepare_batches` job should block a cache read queued behind it.
     Predicts: mixing a large miss-path call into hit-path traffic inflates
     hit-path p99, and splitting the two job kinds across two 1-worker
     executors removes it.

  C. The `transact()` speedup shrinks from 2.07x at 512 writes to 1.23x at
     4096 because 4096 x 8 KB is roughly sqlite's default 32 MB page cache
     (8192 pages x 4 KB), so a big transaction starts spilling dirty pages
     mid-flight. Predicts: raising `sqlite_cache_size` restores the speedup
     at 4096, and lowering it destroys the speedup at 512.

  D. A batch wrapped in `transact()` rolls back entirely if one write fails,
     where unwrapped writes leave the prefix committed. Predicts: an injected
     mid-batch failure leaves N entries without the wrapper and 0 with it.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import statistics
import sys
import tempfile
import time

import diskcache

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

from openai_embeddings_model import validate_cached_embedding

DIM = 1536


# --------------------------------------------------------------------------
# A. inline on the loop vs one executor worker
# --------------------------------------------------------------------------


class InlineModel(c.AsyncOpenAIEmbeddingsModel):
    """Reads the cache directly on the event loop, no executor hop."""

    async def _cache_get_many(self, keys, expected_dimensions=None):
        if self._cache is None or not keys:
            return [None] * len(keys)
        cache = self._cache
        return [validate_cached_embedding(key, cache.get(key), expected_dimensions) for key in keys]

    async def _cache_set_many(self, items):
        if self._cache is None or not items:
            return
        for key, value in items:
            self._cache.set(key, value)


def inline_model(create, **kwargs) -> InlineModel:
    import openai

    client = openai.AsyncOpenAI(api_key="bench", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]
    return InlineModel(model=c.MODEL, openai_client=client, **kwargs)


async def claim_a(concurrency: int, n_texts: int, trials: int) -> dict:
    c.banner("A. inline-on-loop vs executor: is offloading pointless?")
    rows = []
    batches = [c.make_texts(n_texts, seed=2000 + i) for i in range(concurrency)]
    settings = c.settings(dimensions=DIM)

    for label, factory in (
        ("executor, workers=1", lambda cache: c.async_model(c.async_create(), cache=cache, executor_max_workers=1)),
        ("executor, workers=14 (default)", lambda cache: c.async_model(c.async_create(), cache=cache)),
        ("inline on the event loop", lambda cache: inline_model(c.async_create(), cache=cache)),
    ):
        walls, lags = [], []
        for _ in range(trials):
            with tempfile.TemporaryDirectory() as d:
                cache = c.fresh_cache(d)
                model = factory(cache)
                for texts in batches:  # warm
                    await model.get_embeddings(texts, settings)

                async with c.LoopLagProbe() as probe:
                    t0 = time.perf_counter()
                    await asyncio.gather(*(model.get_embeddings(t, settings) for t in batches))
                    walls.append(time.perf_counter() - t0)
                lags.append(probe.summary()["max"] * 1000)

                await model.aclose()
                cache.close()
        rows.append({"variant": label, "wall_s": statistics.median(walls), "max_loop_lag_ms": statistics.median(lags)})

    best = min(rows, key=lambda r: r["wall_s"])
    for row in rows:
        row["vs_best"] = row["wall_s"] / best["wall_s"]
    print(c.table(rows, ["variant", "wall_s", "vs_best", "max_loop_lag_ms"]))

    inline = next(r for r in rows if "inline" in r["variant"])
    w1 = next(r for r in rows if "workers=1" in r["variant"])
    # Throughput and responsiveness are different questions, and the answer
    # differs between them — report both rather than collapsing to one verdict.
    wall_ratio = inline["wall_s"] / w1["wall_s"]
    lag_ratio = inline["max_loop_lag_ms"] / w1["max_loop_lag_ms"]
    verdict = (
        f"throughput: inline is {wall_ratio:.2f}x the executor's wall-clock; "
        f"responsiveness: inline blocks the loop {lag_ratio:.1f}x longer"
    )
    print(f"\n-> Claim A {verdict}")
    return {
        "rows": rows,
        "verdict": verdict,
        "wall_ratio_inline_over_workers1": wall_ratio,
        "lag_ratio_inline_over_workers1": lag_ratio,
        "params": {"concurrency": concurrency, "n_texts": n_texts},
    }


# --------------------------------------------------------------------------
# B. head-of-line blocking behind tiktoken
# --------------------------------------------------------------------------


class SplitExecutorModel(c.AsyncOpenAIEmbeddingsModel):
    """Cache I/O and tiktoken get one dedicated worker each.

    The research recommendation: a long `_prepare_batches` job should not sit
    in front of another caller's cache read.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import concurrent.futures

        self._cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"openai-emb-cpu-{id(self)}"
        )

    async def _embed_missing(self, texts, keys, model_settings):
        loop = asyncio.get_running_loop()
        original = self._executor
        try:
            # `_embed_missing` submits `_prepare_batches` to `self._executor`;
            # point that at the CPU pool for the duration.
            self._executor = self._cpu_executor
            prepared = await loop.run_in_executor(self._cpu_executor, self._prepare_batches, texts)
        finally:
            self._executor = original
        self._prepared = prepared
        return await super()._embed_missing(texts, keys, model_settings)

    async def aclose(self) -> None:
        self._cpu_executor.shutdown(wait=False, cancel_futures=True)
        await super().aclose()


async def claim_b(n_small: int, small_texts: int, big_texts: int, trials: int) -> dict:
    c.banner("B. head-of-line blocking: does tiktoken stall cache reads?")
    rows = []
    small = [c.make_texts(small_texts, seed=3000 + i) for i in range(n_small)]
    big = c.make_texts(big_texts, seed=9999)
    settings = c.settings(dimensions=DIM)

    async def measure(model, with_big: bool) -> list[float]:
        for texts in small:  # warm the hit path
            await model.get_embeddings(texts, settings)

        latencies: list[float] = []

        async def hit_call(texts):
            t0 = time.perf_counter()
            await model.get_embeddings(texts, settings)
            latencies.append((time.perf_counter() - t0) * 1000)

        tasks = [asyncio.create_task(hit_call(t)) for t in small]
        if with_big:
            # An uncached call: its `_prepare_batches` is the long CPU job.
            tasks.append(asyncio.create_task(model.get_embeddings(big, settings)))
        await asyncio.gather(*tasks)
        return latencies

    for label, factory, with_big in (
        (
            "workers=1, hits only",
            lambda cache: c.async_model(c.async_create(), cache=cache, executor_max_workers=1),
            False,
        ),
        (
            "workers=1, + one big miss",
            lambda cache: c.async_model(c.async_create(), cache=cache, executor_max_workers=1),
            True,
        ),
        (
            "split pools, + one big miss",
            lambda cache: SplitExecutorModel(
                model=c.MODEL, openai_client=_async_client(c.async_create()), cache=cache, executor_max_workers=1
            ),
            True,
        ),
    ):
        samples: list[float] = []
        for _ in range(trials):
            with tempfile.TemporaryDirectory() as d:
                cache = c.fresh_cache(d)
                model = factory(cache)
                samples.extend(await measure(model, with_big))
                await model.aclose()
                cache.close()
        summary = c.stats(samples)
        rows.append(
            {
                "variant": label,
                "hit_p50_ms": summary["median"],
                "hit_p95_ms": summary["p95"],
                "hit_max_ms": summary["max"],
            }
        )

    print(c.table(rows, ["variant", "hit_p50_ms", "hit_p95_ms", "hit_max_ms"]))
    baseline, blocked, split = rows
    inflation = blocked["hit_max_ms"] / baseline["hit_max_ms"]
    fixed = split["hit_max_ms"] < blocked["hit_max_ms"]
    verdict = (
        f"CONFIRMED: a big miss inflates hit-path max by {inflation:.1f}x"
        if inflation > 1.5
        else f"NOT REPRODUCED: inflation only {inflation:.1f}x"
    )
    print(f"\n-> Claim B {verdict}; split pools help: {fixed}")
    return {"rows": rows, "verdict": verdict, "split_pools_help": fixed, "inflation": inflation}


def _async_client(create):
    import openai

    client = openai.AsyncOpenAI(api_key="bench", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]
    return client


# --------------------------------------------------------------------------
# C. does sqlite_cache_size explain the shrinking transact() speedup?
# --------------------------------------------------------------------------


def _write_batch(cache: diskcache.Cache, keys: list[str], value: str, *, transacted: bool) -> float:
    t0 = time.perf_counter()
    if transacted:
        with cache.transact():
            for key in keys:
                cache.set(key, value)
    else:
        for key in keys:
            cache.set(key, value)
    return time.perf_counter() - t0


def claim_c(repeats: int, sizes: list[int]) -> dict:
    c.banner("C. is the shrinking transact() speedup sqlite's page cache spilling?")
    value = c.b64_vector(DIM)
    value_kb = len(value.encode()) / 1024
    rows = []

    # default 8192 pages x 4 KB = 32 MB; 1024 pages = 4 MB; 131072 pages = 512 MB
    for cache_pages, label in ((1_024, "4 MB (starved)"), (8_192, "32 MB (default)"), (131_072, "512 MB (ample)")):
        for n in sizes:
            payload_mb = n * value_kb / 1024
            plain, txn = [], []
            for run in range(repeats):
                for transacted, sink in ((False, plain), (True, txn)):
                    with tempfile.TemporaryDirectory() as d:
                        cache = c.fresh_cache(d, sqlite_cache_size=cache_pages)
                        keys = [f"w:{run}:{i}" for i in range(n)]
                        sink.append(_write_batch(cache, keys, value, transacted=transacted))
                        cache.close()
            rows.append(
                {
                    "page_cache": label,
                    "n": n,
                    "payload_mb": payload_mb,
                    "plain_ms": statistics.median(plain) * 1000,
                    "transact_ms": statistics.median(txn) * 1000,
                    "speedup": statistics.median(plain) / statistics.median(txn),
                }
            )

    print(c.table(rows, ["page_cache", "n", "payload_mb", "plain_ms", "transact_ms", "speedup"]))

    big = max(sizes)
    starved = next(r for r in rows if r["n"] == big and r["page_cache"].startswith("4 MB"))
    default = next(r for r in rows if r["n"] == big and r["page_cache"].startswith("32 MB"))
    ample = next(r for r in rows if r["n"] == big and r["page_cache"].startswith("512 MB"))
    recovered = ample["speedup"] > default["speedup"] * 1.15
    verdict = (
        "CONFIRMED: a larger page cache restores the speedup at the big batch"
        if recovered
        else "REFUTED: page cache size does not explain it"
    )
    print(
        f"\n-> Claim C {verdict}\n"
        f"   at n={big}: 4 MB {starved['speedup']:.2f}x | 32 MB {default['speedup']:.2f}x "
        f"| 512 MB {ample['speedup']:.2f}x"
    )
    return {"rows": rows, "verdict": verdict, "recovered": recovered}


# --------------------------------------------------------------------------
# E. does the transact() win survive a big, already-populated cache?
# --------------------------------------------------------------------------


def claim_e(repeats: int, sizes: list[int], populations: list[int]) -> dict:
    """The question C should have asked.

    C varied sqlite's page cache against a *fresh* database and found nothing,
    because nothing spills when the database is small. But `bench_opts.py`
    measured only 1.23x at 4096 writes, and it reached that number by writing
    six repeats into one cache — roughly 196 MB by the end. The variable is
    not the page cache setting, it is how much data is already there. That is
    also the question a user actually has: my cache holds 200k embeddings, does
    this fix still help me?
    """
    c.banner("E. does the transact() win survive an already-large cache?")
    value = c.b64_vector(DIM)
    rows = []

    for population in populations:
        for n in sizes:
            plain, txn = [], []
            for run in range(repeats):
                for transacted, sink in ((False, plain), (True, txn)):
                    with tempfile.TemporaryDirectory() as d:
                        cache = c.fresh_cache(d)
                        if population:
                            c.warm_cache(cache, [f"pre:{i}" for i in range(population)], value)
                        keys = [f"w:{run}:{i}" for i in range(n)]
                        sink.append(_write_batch(cache, keys, value, transacted=transacted))
                        cache.close()
            rows.append(
                {
                    "existing_entries": population,
                    "db_mb": round(population * len(value.encode()) / 1024 / 1024),
                    "n": n,
                    "plain_ms": statistics.median(plain) * 1000,
                    "transact_ms": statistics.median(txn) * 1000,
                    "speedup": statistics.median(plain) / statistics.median(txn),
                }
            )

    print(c.table(rows, ["existing_entries", "db_mb", "n", "plain_ms", "transact_ms", "speedup"]))
    worst = min(rows, key=lambda r: r["speedup"])
    holds = worst["speedup"] > 1.15
    verdict = (
        f"transact() still pays off everywhere (worst case {worst['speedup']:.2f}x at "
        f"{worst['existing_entries']} existing entries, n={worst['n']})"
        if holds
        else f"transact() stops paying off at {worst['existing_entries']} existing entries ({worst['speedup']:.2f}x)"
    )
    print(f"\n-> Claim E: {verdict}")
    return {"rows": rows, "verdict": verdict, "holds": holds}


# --------------------------------------------------------------------------
# D. what a mid-batch write failure costs
# --------------------------------------------------------------------------


def claim_d(n: int, fail_at: int) -> dict:
    c.banner("D. mid-batch write failure: what survives?")
    value = c.b64_vector(DIM)

    class Boom(Exception):
        pass

    def run(transacted: bool) -> int:
        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d)
            keys = [f"d:{i}" for i in range(n)]

            def write_all():
                for i, key in enumerate(keys):
                    if i == fail_at:
                        raise Boom("injected failure mid-batch")
                    cache.set(key, value)

            try:
                if transacted:
                    with cache.transact():
                        write_all()
                else:
                    write_all()
            except Boom:
                pass

            survived = sum(1 for key in keys if cache.get(key) is not None)
            cache.close()
            return survived

    plain_survived = run(transacted=False)
    txn_survived = run(transacted=True)

    rows = [
        {"variant": "plain set() loop", "written_before_failure": fail_at, "survived": plain_survived},
        {"variant": "wrapped in transact()", "written_before_failure": fail_at, "survived": txn_survived},
    ]
    print(c.table(rows, ["variant", "written_before_failure", "survived"]))
    verdict = (
        f"CONFIRMED: transact() discards all {fail_at} already-written entries"
        if txn_survived == 0 and plain_survived == fail_at
        else f"UNEXPECTED: plain={plain_survived}, transact={txn_survived}, expected {fail_at} and 0"
    )
    print(f"\n-> Claim D {verdict}")
    return {"rows": rows, "verdict": verdict, "plain_survived": plain_survived, "transact_survived": txn_survived}


# --------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick

    payload: dict = {"quick": q}
    payload["claim_a"] = await claim_a(concurrency=4 if q else 16, n_texts=64 if q else 512, trials=1 if q else 3)
    payload["claim_b"] = await claim_b(
        n_small=4 if q else 16,
        small_texts=32 if q else 128,
        big_texts=256 if q else 4096,
        trials=1 if q else 3,
    )
    payload["claim_c"] = claim_c(repeats=1 if q else 3, sizes=[128, 512] if q else [512, 2048, 4096])
    payload["claim_e"] = claim_e(
        repeats=1 if q else 3,
        sizes=[128] if q else [512, 4096],
        populations=[0, 2_000] if q else [0, 50_000, 200_000],
    )
    payload["claim_d"] = claim_d(n=32 if q else 512, fail_at=16 if q else 256)

    out = c.save("diag_mechanisms", payload)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
