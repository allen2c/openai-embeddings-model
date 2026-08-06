"""Can the cache-hit ceiling be raised, or is it the floor of the design?

Every other experiment points at the same wall: a single process serves about
61k cache hits/sec, threads make it worse, and 53% of that call is
`validate_cached_embedding` — 84% of which is `base64.b64decode`, run once per
entry.

But the async read path already has the whole batch in hand. Base64 is
4-characters-to-3-bytes, so when every entry decodes to the same length and
that length is a multiple of 3, the entries can be concatenated and decoded in
one call, then checked with a single `np.isfinite` over the whole matrix. Two
Python-level loops over n entries become two C-level passes.

That is only worth proposing if it is both faster and still able to reject the
entries the per-entry version rejects. Measure both.
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import sys
import typing

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

from openai_embeddings_model import validate_cached_embedding

DIMS = [256, 768, 1536, 3072]


def per_entry(keys: list[str], values: list[typing.Any], dim: int | None) -> list[str | None]:
    """What the library does today."""
    return [validate_cached_embedding(k, v, dim) for k, v in zip(keys, values, strict=True)]


def batched(keys: list[str], values: list[typing.Any], dim: int | None) -> list[str | None]:
    """One decode and one finiteness check for the whole batch.

    Falls back to the per-entry path the moment anything is irregular, which is
    what keeps it honest: the fast path only runs when every entry is a string
    of one identical, correctly-sized length.
    """
    if not values:
        return []

    first = values[0]
    if not isinstance(first, str):
        return per_entry(keys, values, dim)

    width = len(first)
    # A uniform batch of str is the common case; anything else is rare enough
    # that the slow path costs nothing overall.
    if any(not isinstance(v, str) or len(v) != width for v in values):
        return per_entry(keys, values, dim)

    try:
        raw = base64.b64decode("".join(values), validate=True)
    except Exception:
        return per_entry(keys, values, dim)

    n = len(values)
    if not raw or len(raw) % (4 * n) != 0:
        return per_entry(keys, values, dim)

    entry_dim = len(raw) // 4 // n
    if entry_dim == 0 or (dim is not None and entry_dim != dim):
        return per_entry(keys, values, dim)

    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, entry_dim)
    finite = np.isfinite(arr).all(axis=1)
    if bool(finite.all()):
        return list(values)
    # Rare: re-check only the entries that failed, so the warnings still name
    # the right key.
    return [values[i] if finite[i] else validate_cached_embedding(keys[i], values[i], dim) for i in range(n)]


def equivalence_check() -> list[dict]:
    """The batched path must reject exactly what the per-entry path rejects."""
    good = c.b64_vector(1536, seed=1)
    other = c.b64_vector(1536, seed=2)
    nan = base64.b64encode(np.full(1536, np.nan, dtype=np.float32).tobytes()).decode()
    short = c.b64_vector(768)

    cases = {
        "all valid": [good, other, good],
        "one None": [good, None, other],
        "one non-str": [good, 12345, other],
        "one NaN": [good, nan, other],
        "one wrong dimension": [good, short, other],
        "one bad base64": [good, "!!!not base64!!!" + good[16:], other],
        "empty batch": [],
        "single entry": [good],
        "all NaN": [nan, nan],
    }

    rows = []
    for label, values in cases.items():
        keys = [f"k{i}" for i in range(len(values))]
        for dim in (1536, None):
            expected = per_entry(keys, list(values), dim)
            actual = batched(keys, list(values), dim)
            rows.append({"case": label, "expected_dim": str(dim), "equal": expected == actual})
    return rows


def timing(n: int, repeats: int) -> list[dict]:
    rows = []
    for dim in DIMS:
        values = [c.b64_vector(dim, seed=i % 32) for i in range(n)]
        keys = [f"k{i}" for i in range(n)]

        base = c.timeit(lambda: per_entry(keys, values, dim), repeats=repeats)
        fast = c.timeit(lambda: batched(keys, values, dim), repeats=repeats)
        rows.append(
            {
                "dim": dim,
                "b64_multiple_of_3_bytes": (dim * 4) % 3 == 0,
                "per_entry_us_each": base["median"] * 1e6 / n,
                "batched_us_each": fast["median"] * 1e6 / n,
                "speedup": base["median"] / fast["median"],
            }
        )
    return rows


def ceiling(n: int, repeats: int) -> list[dict]:
    """Translate the saving into the number that matters: hits per second.

    A cache hit costs key generation, `cache.get` and validation. Only the last
    changes here, so hold the other two fixed and recompute the ceiling.
    """
    dim = 1536
    values = [c.b64_vector(dim, seed=i % 32) for i in range(n)]
    keys = [f"k{i}" for i in range(n)]

    base = c.timeit(lambda: per_entry(keys, values, dim), repeats=repeats)["median"] * 1e6 / n
    fast = c.timeit(lambda: batched(keys, values, dim), repeats=repeats)["median"] * 1e6 / n
    # Measured elsewhere: key generation ~1.6 us/text after the 0.6.0 hoist,
    # cache.get ~3.2 us/text at a realistic cache size.
    other = 1.6 + 3.2
    return [
        {"path": "today (per entry)", "us_per_hit": other + base, "hits_per_sec": 1e6 / (other + base)},
        {"path": "batched validation", "us_per_hit": other + fast, "hits_per_sec": 1e6 / (other + fast)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n = 128 if args.quick else 2048
    repeats = 3 if args.quick else 10

    c.banner("Equivalence: does the batched path reject what the per-entry path rejects?")
    eq = equivalence_check()
    print(c.table(eq, ["case", "expected_dim", "equal"]))
    all_equal = all(r["equal"] for r in eq)
    print(f"\n-> all {len(eq)} cases identical: {all_equal}")

    c.banner(f"Validation cost, {n} entries")
    times = timing(n, repeats)
    print(c.table(times, ["dim", "b64_multiple_of_3_bytes", "per_entry_us_each", "batched_us_each", "speedup"]))

    c.banner("What that does to the single-process cache-hit ceiling")
    ceil = ceiling(n, repeats)
    print(c.table(ceil, ["path", "us_per_hit", "hits_per_sec"]))
    print(f"\n-> ceiling moves {ceil[1]['hits_per_sec'] / ceil[0]['hits_per_sec']:.2f}x")

    c.save(
        "diag_validation_ceiling",
        {"n": n, "repeats": repeats, "equivalence": eq, "all_equal": all_equal, "timing": times, "ceiling": ceil},
    )


if __name__ == "__main__":
    main()
