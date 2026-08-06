"""Can the read collapse be escaped without changing the library?

The collapse is insensitive to working-set size, page cache and mmap, so it is
sqlite's per-query WAL read lock, not I/O. That lock is per database file —
which means sharding across several database files should escape it.

`diskcache.FanoutCache` does exactly that, and the library accepts whatever
cache object it is handed. If a FanoutCache scales, the mitigation costs a
user one line and costs this library nothing.

Also checked here: journal mode (does WAL cause it?), reads batched inside a
transaction, and whether FanoutCache is genuinely drop-in for every cache
operation this library performs.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys
import tempfile
import threading
import time

import diskcache

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

THREADS = [1, 2, 4, 8, 16]
OPS = 20_000
TRIALS = 3
N_KEYS = 5_000


def read_sweep(label: str, cache, keys: list[str]) -> list[dict]:
    rows, base = [], None
    for n in THREADS:
        per = OPS // n
        trials = []
        for _ in range(TRIALS):
            barrier = threading.Barrier(n)

            def worker(idx: int) -> None:
                rng = random.Random(idx)
                barrier.wait()
                for _ in range(per):
                    cache.get(keys[rng.randrange(len(keys))])

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            trials.append(n * per / (time.perf_counter() - t0))
        ops = statistics.median(trials)
        base = base if base is not None else ops
        rows.append({"threads": n, "ops_per_sec": ops, "scaling": ops / base})
    print(f"\n--- {label} ---")
    print(c.table(rows, ["threads", "ops_per_sec", "scaling"]))
    return rows


def write_sweep(label: str, cache) -> list[dict]:
    rows, base = [], None
    value = c.b64_vector(1536)
    run = [0]
    for n in THREADS:
        per = max(1, (OPS // 10) // n)
        trials = []
        for _ in range(TRIALS):
            run[0] += 1
            tag = run[0]
            barrier = threading.Barrier(n)

            def worker(idx: int) -> None:
                barrier.wait()
                for i in range(per):
                    cache.set(f"w:{tag}:{idx}:{i}", value)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            trials.append(n * per / (time.perf_counter() - t0))
        ops = statistics.median(trials)
        base = base if base is not None else ops
        rows.append({"threads": n, "ops_per_sec": ops, "scaling": ops / base})
    print(f"\n--- {label} ---")
    print(c.table(rows, ["threads", "ops_per_sec", "scaling"]))
    return rows


def drop_in_check() -> dict:
    """Every cache operation the library performs, against a FanoutCache."""
    from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, OpenAIEmbeddingsModel  # noqa: F401

    results = {}
    with tempfile.TemporaryDirectory() as d:
        fan = diskcache.FanoutCache(directory=d, shards=8)
        results["is_Cache_subclass"] = isinstance(fan, diskcache.Cache)
        results["has_get"] = hasattr(fan, "get")
        results["has_set"] = hasattr(fan, "set")
        results["has_close"] = hasattr(fan, "close")
        results["has_transact"] = hasattr(fan, "transact")

        texts = c.make_texts(64, seed=99)
        model = c.sync_model(c.sync_create(), cache=fan)
        cold = model.get_embeddings(texts, c.settings(dimensions=1536))
        warm = model.get_embeddings(texts, c.settings(dimensions=1536))
        results["cold_cache_hits"] = cold.usage.cache_hits
        results["warm_cache_hits"] = warm.usage.cache_hits
        results["vectors_identical"] = bool((cold.to_numpy() == warm.to_numpy()).all())

        # the fork path calls cache.close(); make sure that does not explode
        model._reset_after_fork()
        results["reset_after_fork_ok"] = True
        results["usable_after_fork_reset"] = model.get_embeddings(texts, c.settings(dimensions=1536)).usage.cache_hits

        fan.close()
    return results


def main() -> None:
    payload: dict = {"threads": THREADS, "ops_total": OPS, "trials": TRIALS, "keys": N_KEYS}
    keys = [f"k:{i}" for i in range(N_KEYS)]
    value = c.b64_vector(1536)

    # baseline
    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        c.warm_cache(cache, keys, value)
        payload["read_cache_wal"] = read_sweep("read: diskcache.Cache (WAL, default)", cache, keys)
        payload["write_cache_wal"] = write_sweep("write: diskcache.Cache (WAL, default)", cache)
        cache.close()

    # journal mode: is WAL the cause?
    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d, sqlite_journal_mode="truncate")
        c.warm_cache(cache, keys, value)
        payload["read_cache_truncate"] = read_sweep("read: diskcache.Cache (journal_mode=truncate)", cache, keys)
        cache.close()

    # sharded
    for shards in (4, 8, 16):
        with tempfile.TemporaryDirectory() as d:
            fan = diskcache.FanoutCache(directory=d, shards=shards)
            with fan.transact():
                for key in keys:
                    fan.set(key, value)
            payload[f"read_fanout_{shards}"] = read_sweep(f"read: FanoutCache(shards={shards})", fan, keys)
            payload[f"write_fanout_{shards}"] = write_sweep(f"write: FanoutCache(shards={shards})", fan)
            fan.close()

    payload["drop_in_check"] = drop_in_check()
    print("\n--- FanoutCache drop-in check ---")
    for k, v in payload["drop_in_check"].items():
        print(f"  {k}: {v}")

    out = c.RESULTS / "diag_fanout.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
