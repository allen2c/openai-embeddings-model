"""Follow-up: the collapse is inside sqlite, but which part?

diag_read_collapse.py ruled out the GIL (a plain dict scales flat to 16
threads), ruled out the shared `Cache` object (a Cache per thread behaves
identically), and reproduced the collapse with raw sqlite3 on per-thread
connections. Remaining candidates:

  a. working set does not fit sqlite's page cache, so every read is real file
     I/O and the threads contend in the filesystem, not in sqlite
  b. sqlite's WAL index (-shm) locking serialises readers regardless of size

Vary the working-set size and the page-cache pragma to tell them apart.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

THREADS = [1, 2, 4, 8]
OPS = 20_000
TRIALS = 3


def sweep(label: str, cache, keys: list[str]) -> list[dict]:
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


def main() -> None:
    payload: dict = {"threads": THREADS, "ops_total": OPS, "trials": TRIALS}

    cases = [
        # label, n_keys, dim -> value bytes, extra Cache kwargs
        ("tiny values (16 B) x 5000 keys, ~0.1 MB db", 5_000, 4, {}),
        ("8 KB values x 200 keys, ~1.6 MB db", 200, 1536, {}),
        ("8 KB values x 5000 keys, ~40 MB db", 5_000, 1536, {}),
        ("8 KB values x 5000 keys, 256 MB sqlite page cache", 5_000, 1536, {"sqlite_cache_size": 262_144}),
        ("8 KB values x 5000 keys, 1 GB mmap", 5_000, 1536, {"sqlite_mmap_size": 1 << 30}),
    ]

    for label, n_keys, dim, kwargs in cases:
        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d, **kwargs)
            keys = [f"k:{i}" for i in range(n_keys)]
            c.warm_cache(cache, keys, c.b64_vector(dim))
            payload[label] = sweep(label, cache, keys)
            cache.close()

    out = c.RESULTS / "diag_read_collapse2.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
