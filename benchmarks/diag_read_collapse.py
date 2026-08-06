"""Why do concurrent diskcache reads collapse?

bench_cache.py measured 2.4e5 read ops/sec on one thread and 1.9e4 on eight —
a 13x per-operation slowdown from adding threads. Before that goes in a
document, find the mechanism. Four candidates:

  a. the shared `diskcache.Cache` object serialises somewhere
  b. sqlite itself serialises readers
  c. the GIL convoy effect (many short GIL-releasing C calls)
  d. the benchmark is wrong

Each variant below isolates one of them.
"""

from __future__ import annotations

import json
import pathlib
import random
import sqlite3
import statistics
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

THREADS = [1, 2, 4, 8, 16]
OPS = 20_000
TRIALS = 3
N_KEYS = 5_000


def run(n_threads: int, ops: int, work) -> float:
    """`work(idx, rng, ops)` runs one thread's share; returns aggregate ops/sec."""
    barrier = threading.Barrier(n_threads)

    def worker(idx: int) -> None:
        rng = random.Random(idx)
        barrier.wait()
        work(idx, rng, ops)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return n_threads * ops / (time.perf_counter() - t0)


def sweep(label: str, make_work, ops: int = OPS) -> list[dict]:
    rows = []
    base = None
    for n in THREADS:
        trials = [run(n, ops // n if n > 1 else ops, make_work()) for _ in range(TRIALS)]
        # Total work is held constant (each thread does ops//n), so the
        # comparison answers 'do threads finish the same job faster'.
        result = statistics.median(trials)
        if base is None:
            base = result
        rows.append({"threads": n, "ops_per_sec": result, "scaling": result / base})
    print(f"\n--- {label} ---")
    print(c.table(rows, ["threads", "ops_per_sec", "scaling"]))
    return rows


def main() -> None:
    import tempfile

    payload: dict = {"threads": THREADS, "ops_total": OPS, "trials": TRIALS, "keys": N_KEYS}

    with tempfile.TemporaryDirectory() as d:
        keys = [f"k:{i}" for i in range(N_KEYS)]
        value = c.b64_vector(1536)
        shared = c.fresh_cache(d)
        c.warm_cache(shared, keys, value)

        # (a) one shared Cache object, the way the library uses it
        def shared_work():
            def work(idx, rng, ops):
                for _ in range(ops):
                    shared.get(keys[rng.randrange(N_KEYS)])

            return work

        payload["a_shared_cache"] = sweep("a. shared diskcache.Cache (what the library does)", shared_work)

        # (b) one Cache object per thread, same directory
        def per_thread_work():
            def work(idx, rng, ops):
                own = c.fresh_cache(d)
                for _ in range(ops):
                    own.get(keys[rng.randrange(N_KEYS)])
                own.close()

            return work

        payload["b_cache_per_thread"] = sweep("b. one diskcache.Cache per thread, same directory", per_thread_work)

        # (c) raw sqlite SELECT, own connection per thread — the floor diskcache
        #     could ever reach
        db = str(pathlib.Path(d) / "cache.db")

        def raw_work():
            def work(idx, rng, ops):
                con = sqlite3.connect(db, isolation_level=None)
                con.execute("PRAGMA journal_mode = wal")
                for _ in range(ops):
                    con.execute("SELECT value FROM Cache WHERE key = ?", (keys[rng.randrange(N_KEYS)],)).fetchall()
                con.close()

            return work

        payload["c_raw_sqlite"] = sweep("c. raw sqlite3, connection per thread (the floor)", raw_work)

        # (d) pure Python dict lookup — no C call releases the GIL, so this
        #     shows what thread scaling looks like with no GIL handoff at all
        table = dict.fromkeys(keys, value)

        def dict_work():
            def work(idx, rng, ops):
                for _ in range(ops):
                    table[keys[rng.randrange(N_KEYS)]]

            return work

        payload["d_dict_control"] = sweep("d. plain dict (GIL-bound control)", dict_work)

        # (e) shared Cache with a short GIL switch interval — if the collapse is
        #     a convoy, shortening the interval changes it
        for interval in (0.0005, 0.005, 0.05):
            sys.setswitchinterval(interval)
            payload[f"e_switchinterval_{interval}"] = sweep(
                f"e. shared Cache, sys.setswitchinterval({interval})", shared_work
            )
        sys.setswitchinterval(0.005)

        shared.close()

    out = c.RESULTS / "diag_read_collapse.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
