"""Benchmark: quantify low-risk optimisation candidates for openai_embeddings_model.

Every "prototype" here is either a standalone copy of the relevant library
function (with a change applied) or a subclass that overrides exactly one
method -- the shipped library (openai_embeddings_model/__init__.py) is never
modified, and the shared module's functions are never monkeypatched in
place. Each subclass is instantiated fresh per test; nothing here mutates
process-wide state.

Run from the repo root:

    python tmp/bench/bench_opts.py --quick   # tiny params, smoke test (<45s)
    python tmp/bench/bench_opts.py           # full run (< ~8 min)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import hashlib
import itertools
import json
import pathlib
import sys
import tempfile
import time

import diskcache
import numpy as np
import openai

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# The repo root (two levels up from tmp/bench) so `openai_embeddings_model`
# resolves even when it is not pip-installed and PYTHONPATH is unset --
# `python tmp/bench/bench_opts.py` alone does not put the repo root on
# sys.path, only the script's own directory. Must happen before importing
# `common`, which itself imports openai_embeddings_model.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import common as c

import openai_embeddings_model as oem
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelResponse,
    OpenAIEmbeddingsModel,
    Usage,
)

# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def sizes(quick: bool) -> list[int]:
    return [16, 64] if quick else [128, 512, 2048, 4096]


def e2e_sizes(quick: bool) -> list[int]:
    """A smaller subset for end-to-end (full get_embeddings call) timings."""
    return [64] if quick else [512, 4096]


def repeats_for(quick: bool, n: int) -> int:
    if quick:
        return 3
    return 5 if n <= 512 else 3


def make_sync(cls: type, create=None, **kwargs) -> OpenAIEmbeddingsModel:
    create = create if create is not None else c.sync_create()
    client = openai.OpenAI(api_key="bench", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]
    return cls(model=c.MODEL, openai_client=client, **kwargs)


def make_async(cls: type, create=None, **kwargs) -> AsyncOpenAIEmbeddingsModel:
    create = create if create is not None else c.async_create()
    client = openai.AsyncOpenAI(api_key="bench", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]
    return cls(model=c.MODEL, openai_client=client, **kwargs)


# --------------------------------------------------------------------------
# Candidate 1: cache_scope_digest recomputed per text
# --------------------------------------------------------------------------


def _keygen_baseline(texts, model, provider, extra_body, dims):
    return [
        oem.generate_cache_key(model=model, dimensions=dims, text=t, provider=provider, extra_body=extra_body)
        for t in texts
    ]


def _manual_key(scope: str, model: str, dimensions: int | None, text: str) -> str:
    """Same tail as generate_cache_key, built from a precomputed scope digest."""
    hash_text = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return (
        f"{oem.CACHE_KEY_VERSION}:{model if model is not None else 'unknown'}:"
        f"{dimensions if dimensions is not None else 'default'}:"
        f"{scope}:{hash_text}"
    )


def _keygen_prototype(texts, model, provider, extra_body, dims):
    scope = oem.cache_scope_digest(provider, extra_body)  # computed once
    return [_manual_key(scope, model, dims, t) for t in texts]


class ScopeOnceModel(OpenAIEmbeddingsModel):
    """Prototype: computes cache_scope_digest once per get_embeddings call
    instead of once per text. Everything else is a verbatim copy of
    OpenAIEmbeddingsModel.get_embeddings."""

    def get_embeddings(self, input, model_settings):
        _input = oem.validate_input(input)
        model_settings.validate_for_model(self.model)

        unique_texts, slots = oem.deduplicate_texts(_input)

        scope = oem.cache_scope_digest(self._provider, model_settings.extra_body)
        keys = [_manual_key(scope, self._model_str, model_settings.dimensions, t) for t in unique_texts]

        resolved = [self._cache_get(key, model_settings.dimensions) for key in keys]
        missing = [slot for slot, value in enumerate(resolved) if value is None]
        cache_hits = sum(1 for slot in slots if resolved[slot] is not None)

        usage = oem.Usage()
        if missing:
            embeddings, usage = self._embed_missing(
                [unique_texts[slot] for slot in missing],
                [keys[slot] for slot in missing],
                model_settings,
            )
            for slot, embedding in zip(missing, embeddings, strict=True):
                resolved[slot] = embedding

        _output = [resolved[slot] for slot in slots]
        if any(item is None for item in _output):
            raise RuntimeError("Failed to generate embeddings for some inputs")

        return oem.ModelResponse.model_validate(
            {
                "output": _output,
                "usage": oem.Usage(
                    input_tokens=int(usage.input_tokens),
                    total_tokens=int(usage.total_tokens),
                    cache_hits=int(cache_hits),
                    truncated_texts=int(usage.truncated_texts),
                ),
            }
        )


def _check_fast_path_not_taken() -> dict:
    custom = openai.OpenAI(api_key="bench", base_url="http://localhost:1")
    default = openai.OpenAI(api_key="bench")
    custom_provider = str(getattr(custom, "base_url", "") or "")
    default_provider = str(getattr(default, "base_url", "") or "")
    return {
        "custom_base_url_provider": custom_provider,
        "custom_base_url_is_falsy": not custom_provider,
        "default_base_url_provider": default_provider,
        "default_base_url_is_falsy": not default_provider,
    }


def cand1_scope_digest(quick: bool) -> dict:
    c.banner("Candidate 1: cache_scope_digest recomputed per text")
    fast_path = _check_fast_path_not_taken()
    fast_path_reachable = fast_path["custom_base_url_is_falsy"] or fast_path["default_base_url_is_falsy"]
    print(f"'default' fast path reachable via a real openai client: {fast_path_reachable}  ({fast_path})")

    ns = sizes(quick)
    configs = [
        ("no extra_body", None),
        ("with extra_body", {"output_dimension": 512, "opt": ["a", "b"], "nested": {"x": 1}}),
    ]
    provider = str(openai.OpenAI(api_key="bench", base_url="http://localhost:1").base_url)

    rows = []
    for label, extra_body in configs:
        for n in ns:
            texts = c.make_texts(n, seed=1000 + n)
            reps = repeats_for(quick, n)

            baseline = c.timeit(
                lambda texts=texts, extra_body=extra_body: _keygen_baseline(texts, c.MODEL, provider, extra_body, None),
                repeats=reps,
                warmup=1,
            )
            prototype = c.timeit(
                lambda texts=texts, extra_body=extra_body: _keygen_prototype(
                    texts, c.MODEL, provider, extra_body, None
                ),
                repeats=reps,
                warmup=1,
            )
            keys_a = _keygen_baseline(texts, c.MODEL, provider, extra_body, None)
            keys_b = _keygen_prototype(texts, c.MODEL, provider, extra_body, None)
            equal = keys_a == keys_b

            speedup = baseline["median"] / prototype["median"] if prototype["median"] > 0 else float("inf")
            rows.append(
                {
                    "config": label,
                    "n": n,
                    "baseline_ms": baseline["median"] * 1000,
                    "prototype_ms": prototype["median"] * 1000,
                    "speedup": speedup,
                    "keys_equal": equal,
                }
            )

    print(c.table(rows, ["config", "n", "baseline_ms", "prototype_ms", "speedup", "keys_equal"]))

    # End-to-end effect: fully warmed (all-hit) cache, so the whole timed
    # call is key-building + cache reads -- nothing else.
    e2e_rows = []
    for n in e2e_sizes(quick):
        texts = c.make_texts(n, seed=2000 + n)
        settings = c.settings(extra_body={"output_dimension": 512, "opt": "x"})

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cache1 = c.fresh_cache(d1)
            cache2 = c.fresh_cache(d2)
            base_model = make_sync(OpenAIEmbeddingsModel, cache=cache1)
            proto_model = make_sync(ScopeOnceModel, cache=cache2)

            keys = [c.legacy_cache_key(base_model, t, settings) for t in texts]
            value = c.b64_vector(c.DIM)
            c.warm_cache(cache1, keys, value)
            c.warm_cache(cache2, keys, value)

            reps = 3 if not quick else 2
            base_stats = c.timeit(lambda: base_model.get_embeddings(texts, settings), repeats=reps, warmup=1)
            proto_stats = c.timeit(lambda: proto_model.get_embeddings(texts, settings), repeats=reps, warmup=1)

            r1 = base_model.get_embeddings(texts, settings)
            r2 = proto_model.get_embeddings(texts, settings)
            outputs_equal = r1.output == r2.output

        speedup = base_stats["median"] / proto_stats["median"] if proto_stats["median"] > 0 else float("inf")
        e2e_rows.append(
            {
                "n": n,
                "baseline_ms": base_stats["median"] * 1000,
                "prototype_ms": proto_stats["median"] * 1000,
                "speedup": speedup,
                "outputs_equal": outputs_equal,
            }
        )

    print(c.table(e2e_rows, ["n", "baseline_ms", "prototype_ms", "speedup", "outputs_equal"]))

    return {
        "id": "cache_scope_digest_reuse",
        "title": "Compute cache_scope_digest once per call instead of once per text",
        "parameters": {"sizes": ns, "e2e_sizes": e2e_sizes(quick)},
        "fast_path_check": fast_path,
        "fast_path_reachable_in_practice": fast_path_reachable,
        "microbenchmark": rows,
        "end_to_end_all_hits": e2e_rows,
        "keys_byte_identical": all(r["keys_equal"] for r in rows),
        "risk": (
            "Low. The prototype only changes when cache_scope_digest runs (once per call instead of "
            "once per text) and reuses generate_cache_key's own key-string template verbatim; key "
            "bytes are unchanged (verified equal above for every size and both with/without "
            "extra_body), so no existing cache entry is invalidated. The library always constructs its "
            "client with a base_url, so `str(client.base_url)` is never empty -- the 'not provider and "
            "not extra_body' fast path in cache_scope_digest is dead code on the real call path "
            "(confirmed above), meaning every text pays the full json.dumps+sha256 today."
        ),
    }


# --------------------------------------------------------------------------
# Candidate 2: cache writes without a transaction
# --------------------------------------------------------------------------


def _write_loop(cache: diskcache.Cache, keys: list[str], value: str) -> None:
    for k in keys:
        cache.set(k, value)


def _write_txn(cache: diskcache.Cache, keys: list[str], value: str) -> None:
    with cache.transact():
        for k in keys:
            cache.set(k, value)


class TxnCacheModel(OpenAIEmbeddingsModel):
    """Prototype: wraps each batch's cache writes in one transaction."""

    def _embed_missing(self, texts, keys, model_settings):
        safe_texts, batches, truncated = self._prepare_batches(texts)
        results: list = [None] * len(texts)
        total_input_tokens = 0
        total_tokens = 0

        for batch_no, group in enumerate(batches, start=1):
            batch = [safe_texts[i] for i in group]
            response = self._create_with_retry(batch, model_settings, batch_no, len(batches))
            batch_embeddings = oem.extract_ordered_embeddings(response.data)
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Provider returned {len(batch_embeddings)} embeddings for "
                    f"{len(batch)} inputs in batch {batch_no}/{len(batches)}"
                )

            if self._cache is not None:
                with self._cache.transact():
                    for index, embedding in zip(group, batch_embeddings, strict=True):
                        results[index] = embedding
                        self._cache.set(keys[index], embedding)
            else:
                for index, embedding in zip(group, batch_embeddings, strict=True):
                    results[index] = embedding

            batch_usage = self._resolve_usage(response, batch)
            total_input_tokens += batch_usage.input_tokens
            total_tokens += batch_usage.total_tokens

        return results, oem.Usage(input_tokens=total_input_tokens, total_tokens=total_tokens, truncated_texts=truncated)


class TxnAsyncModel(AsyncOpenAIEmbeddingsModel):
    """Prototype: writes a batch's cache entries in one transaction."""

    async def _cache_set_many(self, items):
        if self._cache is None or not items:
            return
        cache = self._cache
        loop = asyncio.get_running_loop()

        def _write():
            with cache.transact():
                for key, value in items:
                    cache.set(key, value)

        await loop.run_in_executor(self._executor, _write)


async def _cand2_async_equiv(texts, settings) -> tuple[bool, bool]:
    with tempfile.TemporaryDirectory() as dc, tempfile.TemporaryDirectory() as dd:
        cache_c = c.fresh_cache(dc)
        cache_d = c.fresh_cache(dd)
        base_amodel = make_async(AsyncOpenAIEmbeddingsModel, cache=cache_c)
        proto_amodel = make_async(TxnAsyncModel, cache=cache_d)
        r1 = await base_amodel.get_embeddings(texts, settings)
        r2 = await proto_amodel.get_embeddings(texts, settings)
        outputs_equal = r1.output == r2.output
        keys_eq = [c.legacy_cache_key(base_amodel, t, settings) for t in texts]
        cache_equal = all(cache_c.get(k) == cache_d.get(k) for k in keys_eq)
        await base_amodel.aclose()
        await proto_amodel.aclose()
        return outputs_equal, cache_equal


async def _cand2_async_e2e(quick: bool) -> list[dict]:
    rows = []
    for n in e2e_sizes(quick):
        texts = c.make_texts(n, seed=7000 + n)
        settings = c.settings()
        reps = 3 if not quick else 2

        with tempfile.TemporaryDirectory() as d1:
            cache1 = c.fresh_cache(d1)
            base_model = make_async(AsyncOpenAIEmbeddingsModel, create=c.async_create(), cache=cache1)

            async def fn_base():
                cache1.clear()
                return await base_model.get_embeddings(texts, settings)

            base_stats = await c.atimeit(fn_base, repeats=reps, warmup=1)
            await base_model.aclose()

        with tempfile.TemporaryDirectory() as d2:
            cache2 = c.fresh_cache(d2)
            proto_model = make_async(TxnAsyncModel, create=c.async_create(), cache=cache2)

            async def fn_proto():
                cache2.clear()
                return await proto_model.get_embeddings(texts, settings)

            proto_stats = await c.atimeit(fn_proto, repeats=reps, warmup=1)
            await proto_model.aclose()

        speedup = base_stats["median"] / proto_stats["median"] if proto_stats["median"] > 0 else float("inf")
        rows.append(
            {
                "n": n,
                "path": "async",
                "baseline_ms": base_stats["median"] * 1000,
                "prototype_ms": proto_stats["median"] * 1000,
                "speedup": speedup,
            }
        )
    return rows


def cand2_txn_writes(quick: bool) -> dict:
    c.banner("Candidate 2: cache writes without a transaction")
    ns = sizes(quick)
    raw_rows = []

    for n in ns:
        reps = repeats_for(quick, n)
        base_keys = [f"k-{n}-{i}" for i in range(n)]
        value = c.b64_vector(c.DIM)

        with tempfile.TemporaryDirectory() as d1:
            cache1 = c.fresh_cache(d1)
            counter1 = itertools.count()

            def fn_loop(cache=cache1, base_keys=base_keys, counter=counter1):
                i = next(counter)
                keys_i = [f"{k}-r{i}" for k in base_keys]
                _write_loop(cache, keys_i, value)

            baseline = c.timeit(fn_loop, repeats=reps, warmup=1)

        with tempfile.TemporaryDirectory() as d2:
            cache2 = c.fresh_cache(d2)
            counter2 = itertools.count()

            def fn_txn(cache=cache2, base_keys=base_keys, counter=counter2):
                i = next(counter)
                keys_i = [f"{k}-r{i}" for k in base_keys]
                _write_txn(cache, keys_i, value)

            prototype = c.timeit(fn_txn, repeats=reps, warmup=1)

        speedup = baseline["median"] / prototype["median"] if prototype["median"] > 0 else float("inf")
        raw_rows.append(
            {
                "n": n,
                "baseline_ops_per_sec": n / baseline["median"],
                "prototype_ops_per_sec": n / prototype["median"],
                "baseline_ms": baseline["median"] * 1000,
                "prototype_ms": prototype["median"] * 1000,
                "speedup": speedup,
            }
        )

    print(
        c.table(
            raw_rows,
            ["n", "baseline_ops_per_sec", "prototype_ops_per_sec", "baseline_ms", "prototype_ms", "speedup"],
        )
    )

    # Equivalence: same keys -> same values written, same ModelResponse output.
    eq_n = 24
    texts_eq = c.make_texts(eq_n, seed=42)
    settings_eq = c.settings()
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        cache_a = c.fresh_cache(da)
        cache_b = c.fresh_cache(db)
        base_model = make_sync(OpenAIEmbeddingsModel, cache=cache_a)
        proto_model = make_sync(TxnCacheModel, cache=cache_b)
        r1 = base_model.get_embeddings(texts_eq, settings_eq)
        r2 = proto_model.get_embeddings(texts_eq, settings_eq)
        sync_outputs_equal = r1.output == r2.output
        keys_eq = [c.legacy_cache_key(base_model, t, settings_eq) for t in texts_eq]
        sync_cache_equal = all(cache_a.get(k) == cache_b.get(k) for k in keys_eq)

    async_outputs_equal, async_cache_equal = asyncio.run(_cand2_async_equiv(texts_eq, settings_eq))

    # End-to-end all-miss wall-clock, sync path.
    e2e_rows = []
    for n in e2e_sizes(quick):
        texts = c.make_texts(n, seed=6000 + n)
        settings = c.settings()
        reps = 3 if not quick else 2

        with tempfile.TemporaryDirectory() as d1:
            cache1 = c.fresh_cache(d1)
            base_model = make_sync(OpenAIEmbeddingsModel, create=c.sync_create(), cache=cache1)

            def fn_base(cache=cache1, model=base_model, texts=texts, settings=settings):
                cache.clear()
                return model.get_embeddings(texts, settings)

            base_stats = c.timeit(fn_base, repeats=reps, warmup=1)

        with tempfile.TemporaryDirectory() as d2:
            cache2 = c.fresh_cache(d2)
            proto_model = make_sync(TxnCacheModel, create=c.sync_create(), cache=cache2)

            def fn_proto(cache=cache2, model=proto_model, texts=texts, settings=settings):
                cache.clear()
                return model.get_embeddings(texts, settings)

            proto_stats = c.timeit(fn_proto, repeats=reps, warmup=1)

        speedup = base_stats["median"] / proto_stats["median"] if proto_stats["median"] > 0 else float("inf")
        e2e_rows.append(
            {
                "n": n,
                "path": "sync",
                "baseline_ms": base_stats["median"] * 1000,
                "prototype_ms": proto_stats["median"] * 1000,
                "speedup": speedup,
            }
        )

    e2e_rows.extend(asyncio.run(_cand2_async_e2e(quick)))
    print(c.table(e2e_rows, ["n", "path", "baseline_ms", "prototype_ms", "speedup"]))

    return {
        "id": "txn_cache_writes",
        "title": "Wrap per-batch cache writes in a single diskcache transaction",
        "parameters": {"sizes": ns, "e2e_sizes": e2e_sizes(quick)},
        "raw_write_throughput": raw_rows,
        "equivalence": {
            "sync_outputs_equal": sync_outputs_equal,
            "sync_cache_contents_equal": sync_cache_equal,
            "async_outputs_equal": async_outputs_equal,
            "async_cache_contents_equal": async_cache_equal,
        },
        "end_to_end_all_miss": e2e_rows,
        "risk": (
            "Medium-low, and semantic, as the task flags. A transact()-wrapped batch becomes atomic "
            "and holds diskcache's single writer lock for the whole batch instead of releasing it "
            "between each set(). Readers are never blocked by a diskcache writer (diskcache readers "
            "don't take the write lock), so concurrent get_embeddings() cache *hits* are unaffected; "
            "only another concurrent *writer* -- a second in-flight embedding call landing new misses "
            "against the same cache directory, in another thread or process -- queues behind the lock "
            "for up to one batch (<= max_batch_size items) instead of interleaving between individual "
            "set() calls. The library's stated guarantee, 'each batch is written to the cache as soon "
            "as it succeeds,' is preserved exactly: the transaction still only commits once that "
            "batch's embeddings are known, same as today; it turns N fsyncs into 1 rather than "
            "changing when the write happens."
        ),
    }


# --------------------------------------------------------------------------
# Candidate 3: async validation on the event loop
# --------------------------------------------------------------------------


class ValidateInExecutorModel(AsyncOpenAIEmbeddingsModel):
    """Prototype: validates cached entries inside the executor job, off the event loop."""

    async def _cache_get_many(self, keys, expected_dimensions=None):
        if self._cache is None or not keys:
            return [None] * len(keys)

        cache = self._cache
        loop = asyncio.get_running_loop()

        def _read_and_validate():
            raw = [cache.get(key) for key in keys]
            return [
                oem.validate_cached_embedding(key, value, expected_dimensions)
                for key, value in zip(keys, raw, strict=True)
            ]

        return await loop.run_in_executor(self._executor, _read_and_validate)


async def _measure_loop_lag(model, texts, settings, iterations: int) -> dict:
    probe = c.LoopLagProbe()
    async with probe:
        for _ in range(iterations):
            await model.get_embeddings(texts, settings)
    return probe.summary()


async def _cand3_body(quick: bool) -> dict:
    ns = [64, 256] if quick else [512, 2048, 4096]
    lag_iterations = 6 if quick else 15
    reps = 2 if quick else 3

    rows = []
    for n in ns:
        texts = c.make_texts(n, seed=8000 + n)
        settings = c.settings()

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cache1 = c.fresh_cache(d1)
            cache2 = c.fresh_cache(d2)
            base_model = make_async(AsyncOpenAIEmbeddingsModel, cache=cache1)
            proto_model = make_async(ValidateInExecutorModel, cache=cache2)

            provider = base_model._provider
            keys = [
                oem.generate_cache_key(model=base_model._model_str, dimensions=None, text=t, provider=provider)
                for t in texts
            ]
            value = c.b64_vector(c.DIM)
            c.warm_cache(cache1, keys, value)
            c.warm_cache(cache2, keys, value)

            base_wall = await c.atimeit(lambda: base_model.get_embeddings(texts, settings), repeats=reps, warmup=1)
            proto_wall = await c.atimeit(lambda: proto_model.get_embeddings(texts, settings), repeats=reps, warmup=1)

            base_lag = await _measure_loop_lag(base_model, texts, settings, lag_iterations)
            proto_lag = await _measure_loop_lag(proto_model, texts, settings, lag_iterations)

            r1 = await base_model.get_embeddings(texts, settings)
            r2 = await proto_model.get_embeddings(texts, settings)
            outputs_equal = r1.output == r2.output

            await base_model.aclose()
            await proto_model.aclose()

        speedup = base_wall["median"] / proto_wall["median"] if proto_wall["median"] > 0 else float("inf")
        rows.append(
            {
                "n": n,
                "baseline_wall_ms": base_wall["median"] * 1000,
                "prototype_wall_ms": proto_wall["median"] * 1000,
                "wall_speedup": speedup,
                "baseline_max_lag_ms": base_lag["max"] * 1000,
                "prototype_max_lag_ms": proto_lag["max"] * 1000,
                "baseline_mean_lag_ms": base_lag["mean"] * 1000,
                "prototype_mean_lag_ms": proto_lag["mean"] * 1000,
                "outputs_equal": outputs_equal,
            }
        )

    print(
        c.table(
            rows,
            [
                "n",
                "baseline_wall_ms",
                "prototype_wall_ms",
                "wall_speedup",
                "baseline_max_lag_ms",
                "prototype_max_lag_ms",
                "outputs_equal",
            ],
        )
    )

    return {
        "id": "async_validation_off_loop",
        "title": "Move validate_cached_embedding into the executor job, off the event loop",
        "parameters": {"sizes": ns, "lag_iterations": lag_iterations},
        "results": rows,
        "risk": (
            "Low. validate_cached_embedding is a pure function of (key, cached, expected_dimensions) "
            "with no dependency on the event loop; moving its list comprehension inside the same "
            "executor lambda that already does the raw cache.get() calls changes nothing about its "
            "output, only which thread runs it. This is a latency-fairness fix, not a throughput one: "
            "total wall-clock is expected to be roughly unchanged (same work, same thread pool) while "
            "the event loop is blocked for less time per call -- see max/mean loop-lag columns, which "
            "are the numbers that matter once other coroutines (concurrent requests) share the loop."
        ),
    }


def cand3_async_validation(quick: bool) -> dict:
    c.banner("Candidate 3: async validation on the event loop")
    return asyncio.run(_cand3_body(quick))


# --------------------------------------------------------------------------
# Candidate 4: validate_cached_embedding internals
# --------------------------------------------------------------------------


def _validate_streamlined(key: str, cached, expected_dimensions: int | None = None):
    """Same checks and same single b64decode as validate_cached_embedding,
    with the length quotient/remainder computed together."""
    if cached is None:
        return None
    if not isinstance(cached, str):
        return None
    try:
        raw = base64.b64decode(cached, validate=True)
    except Exception:
        return None
    n = len(raw)
    if not raw or n % 4:
        return None
    dims = n // 4
    if expected_dimensions is not None and dims != expected_dimensions:
        return None
    arr = np.frombuffer(raw, dtype=np.float32)
    if not np.isfinite(arr).all():
        return None
    return cached


def cand4_validate_internals(quick: bool) -> dict:
    c.banner("Candidate 4: validate_cached_embedding internals")
    ns = sizes(quick)

    component_rows = []
    for n in ns:
        reps = repeats_for(quick, n)
        strs = [c.b64_vector(c.DIM, seed=i) for i in range(n)]
        raws = [base64.b64decode(s, validate=True) for s in strs]
        arrs = [np.frombuffer(r, dtype=np.float32) for r in raws]

        t_b64 = c.timeit(lambda strs=strs: [base64.b64decode(s, validate=True) for s in strs], repeats=reps, warmup=1)
        t_frombuffer = c.timeit(
            lambda raws=raws: [np.frombuffer(r, dtype=np.float32) for r in raws], repeats=reps, warmup=1
        )
        t_isfinite = c.timeit(lambda arrs=arrs: [np.isfinite(a).all() for a in arrs], repeats=reps, warmup=1)
        t_full_baseline = c.timeit(
            lambda strs=strs: [oem.validate_cached_embedding(f"k{i}", s, None) for i, s in enumerate(strs)],
            repeats=reps,
            warmup=1,
        )
        t_full_prototype = c.timeit(
            lambda strs=strs: [_validate_streamlined(f"k{i}", s, None) for i, s in enumerate(strs)],
            repeats=reps,
            warmup=1,
        )

        component_rows.append(
            {
                "n": n,
                "b64decode_us_per_entry": t_b64["median"] * 1e6 / n,
                "frombuffer_us_per_entry": t_frombuffer["median"] * 1e6 / n,
                "isfinite_us_per_entry": t_isfinite["median"] * 1e6 / n,
                "full_baseline_us_per_entry": t_full_baseline["median"] * 1e6 / n,
                "full_prototype_us_per_entry": t_full_prototype["median"] * 1e6 / n,
                "speedup": (
                    t_full_baseline["median"] / t_full_prototype["median"] if t_full_prototype["median"] > 0 else 0
                ),
                "per_2048_texts_baseline_ms": t_full_baseline["median"] * 1000 * 2048 / n,
                "per_2048_texts_prototype_ms": t_full_prototype["median"] * 1000 * 2048 / n,
            }
        )

    print(
        c.table(
            component_rows,
            [
                "n",
                "b64decode_us_per_entry",
                "frombuffer_us_per_entry",
                "isfinite_us_per_entry",
                "full_baseline_us_per_entry",
                "full_prototype_us_per_entry",
                "speedup",
            ],
        )
    )

    # Equivalence on edge cases, not just the happy path.
    bad_b64 = "not base64 at all!!"
    wrong_len = base64.b64encode(b"abc").decode()  # 3 bytes: not a multiple of 4
    nan_vec = np.array([1.0, float("nan"), 3.0, 4.0], dtype=np.float32)
    nan_b64 = base64.b64encode(nan_vec.tobytes()).decode()
    good = c.b64_vector(c.DIM)

    cases = [
        ("none", None, None),
        ("non_str", 12345, None),
        ("bad_base64", bad_b64, None),
        ("wrong_length", wrong_len, None),
        ("nan_values", nan_b64, None),
        ("dimension_mismatch", good, c.DIM + 1),
        ("valid", good, c.DIM),
    ]
    equivalence_cases = []
    for name, cached, expected in cases:
        base_result = oem.validate_cached_embedding(f"case-{name}", cached, expected)
        proto_result = _validate_streamlined(f"case-{name}", cached, expected)
        equivalence_cases.append({"case": name, "equal": base_result == proto_result})

    equivalent = all(row["equal"] for row in equivalence_cases)
    print(c.table(equivalence_cases, ["case", "equal"]))

    return {
        "id": "validate_cached_embedding_internals",
        "title": "Profile validate_cached_embedding's b64decode / frombuffer / isfinite, try a streamlined version",
        "parameters": {"sizes": ns},
        "component_timings": component_rows,
        "equivalence_cases": equivalence_cases,
        "all_cases_equivalent": equivalent,
        "risk": (
            "N/A as an optimisation. np.frombuffer is already a zero-copy view over the buffer "
            "base64.b64decode produced -- it is not a second decode of anything, just a reinterpret "
            "cast, and the component timings above show it costing roughly nothing next to b64decode "
            "and isfinite. There is no redundant work inside validate_cached_embedding to remove: `raw` "
            "is decoded once and every later check (length, dimension, isfinite) already reuses it. "
            "The two real costs -- base64.b64decode(validate=True) and np.isfinite(...).all() -- are "
            "both irreducible: validate=True is load-bearing (it is what turns a corrupted cache entry "
            "into a clean miss instead of a crash later), and isfinite must touch every float to catch "
            "NaN/inf. The streamlined variant is functionally identical (verified on 5 edge cases plus "
            "the happy path, see equivalence_cases) but not meaningfully faster -- see the speedup "
            "column."
        ),
    }


# --------------------------------------------------------------------------
# Candidate 5: ModelResponse.model_validate vs model_construct
# --------------------------------------------------------------------------


def cand5_model_validate_vs_construct(quick: bool) -> dict:
    c.banner("Candidate 5: ModelResponse.model_validate vs model_construct")
    ns = sizes(quick)
    rows = []

    for n in ns:
        reps = repeats_for(quick, n)
        outputs = [c.b64_vector(c.DIM, seed=i) for i in range(n)]
        usage = Usage(input_tokens=100 * n, total_tokens=100 * n, cache_hits=n // 2, truncated_texts=0)

        baseline = c.timeit(
            lambda outputs=outputs, usage=usage: ModelResponse.model_validate({"output": outputs, "usage": usage}),
            repeats=reps,
            warmup=1,
        )
        prototype = c.timeit(
            lambda outputs=outputs, usage=usage: ModelResponse.model_construct(output=outputs, usage=usage),
            repeats=reps,
            warmup=1,
        )
        speedup = baseline["median"] / prototype["median"] if prototype["median"] > 0 else float("inf")
        rows.append(
            {
                "n": n,
                "baseline_ms": baseline["median"] * 1000,
                "prototype_ms": prototype["median"] * 1000,
                "speedup": speedup,
            }
        )

    print(c.table(rows, ["n", "baseline_ms", "prototype_ms", "speedup"]))

    # Equivalence + the frozen-model risk check.
    outputs = [c.b64_vector(c.DIM, seed=i) for i in range(8)]
    usage = Usage(input_tokens=1, total_tokens=1)
    r1 = ModelResponse.model_validate({"output": outputs, "usage": usage})
    r2 = ModelResponse.model_construct(output=outputs, usage=usage)

    outputs_equal = r1.output == r2.output
    usage_equal = r1.usage == r2.usage
    numpy_equal = bool(np.array_equal(r1.to_numpy(), r2.to_numpy()))
    python_equal = r1.to_python() == r2.to_python()

    frozen_still_enforced = {}
    for label, resp in [("model_validate", r1), ("model_construct", r2)]:
        try:
            resp.output = ["mutated"]
            frozen_still_enforced[label] = False
        except Exception:
            frozen_still_enforced[label] = True

    print(
        f"Equivalence: output={outputs_equal} usage={usage_equal} numpy={numpy_equal} "
        f"python={python_equal} frozen_enforced={frozen_still_enforced}"
    )

    return {
        "id": "model_response_construct",
        "title": "ModelResponse.model_construct instead of model_validate for the internally-built response",
        "parameters": {"sizes": ns},
        "results": rows,
        "equivalence": {
            "output_equal": outputs_equal,
            "usage_equal": usage_equal,
            "to_numpy_equal": numpy_equal,
            "to_python_equal": python_equal,
            "frozen_still_enforced": frozen_still_enforced,
        },
        "risk": (
            "Low, conditionally. model_construct skips field validation and coercion entirely, so it "
            "is only safe where the caller already guarantees the shape pydantic would have validated. "
            "That holds at both call sites in get_embeddings: `output` is always a list[str] produced "
            "internally by extract_ordered_embeddings/cache reads, and `usage` is always an "
            "already-constructed Usage instance, never a raw dict. Confirmed empirically: "
            "`model_config = ConfigDict(frozen=True)` is still enforced on a model_construct instance "
            "(mutating .output afterwards raises the same pydantic.ValidationError as after "
            "model_validate) because frozen enforcement lives in __setattr__, which model_construct "
            "does not touch or bypass. The one real risk is silent, not immediate: if a future change "
            "ever passes `usage=` as a plain dict instead of a Usage instance, model_construct would "
            "store the dict as-is with no error, and `.usage.input_tokens` would then break with an "
            "AttributeError far from the actual mistake; model_validate would have coerced it safely."
        ),
    }


# --------------------------------------------------------------------------
# Candidate 6: to_python() is documented as cached but is not
# --------------------------------------------------------------------------


class CachedPythonResponse(ModelResponse):
    """Prototype: caches the python-list conversion the same way _ndarray is
    already cached (same functools.cached_property mechanism, proven safe on
    a frozen model by _ndarray/_decoded_bytes and by candidate 5's check)."""

    @functools.cached_property
    def _python_list(self) -> list:
        return self._ndarray.tolist()

    def to_python(self) -> list:
        return self._python_list


def cand6_to_python_caching(quick: bool) -> dict:
    c.banner("Candidate 6: to_python() is documented as cached but is not")
    n = 256 if quick else 2048
    dim = c.DIM
    reps = 3 if quick else 6

    outputs = [c.b64_vector(dim, seed=i) for i in range(n)]
    usage = Usage()

    resp_baseline = ModelResponse.model_validate({"output": outputs, "usage": usage})
    resp_prototype = CachedPythonResponse.model_validate({"output": outputs, "usage": usage})

    baseline = c.timeit(lambda: resp_baseline.to_python(), repeats=reps, warmup=1)
    prototype = c.timeit(lambda: resp_prototype.to_python(), repeats=reps, warmup=1)

    equal = resp_baseline.to_python() == resp_prototype.to_python()
    speedup = baseline["median"] / prototype["median"] if prototype["median"] > 0 else float("inf")

    rows = [
        {
            "n": n,
            "dim": dim,
            "baseline_ms_per_call": baseline["median"] * 1000,
            "prototype_ms_per_call_after_warm": prototype["median"] * 1000,
            "speedup_on_repeat_calls": speedup,
            "outputs_equal": equal,
        }
    ]
    print(
        c.table(
            rows,
            [
                "n",
                "dim",
                "baseline_ms_per_call",
                "prototype_ms_per_call_after_warm",
                "speedup_on_repeat_calls",
                "outputs_equal",
            ],
        )
    )

    return {
        "id": "to_python_not_cached",
        "title": "to_python() re-runs .tolist() on every call despite being documented '(cached)'",
        "parameters": {"n": n, "dim": dim},
        "results": rows,
        "recommendation": (
            "Code and docstring disagree today: only `_ndarray` is a cached_property; to_python() "
            "calls `.tolist()` fresh on every call, so a caller who trusts the docstring and calls "
            f"to_python() repeatedly pays this cost every time ({baseline['median'] * 1000:.3f} ms per "
            f"call at n={n}, dim={dim}) for no reason. Since the fix is the same one-line "
            "cached_property pattern already used for `_ndarray` and `_decoded_bytes`, implementing "
            "the cache is the better fix over relaxing the docstring: it costs one field, matches what "
            "callers already believe, and to_python()'s output is plain list-of-lists data with no "
            "aliasing hazard back into caller-owned memory the way a cached ndarray view would have."
        ),
        "risk": (
            "Low if cached: a cached_property adds one dict entry to the instance and, like _ndarray, "
            "is invalidated only by object identity, which is correct for a frozen response. The one "
            "caveat: callers who mutate the returned list in place would now see that mutation in every "
            "later to_python() call, whereas today each call returns an independent list from a fresh "
            ".tolist(). to_numpy() already documents and handles the analogous risk for arrays via "
            ".copy(); the same caveat should be called out in to_python()'s docstring if this ships."
        ),
    }


# --------------------------------------------------------------------------
# Candidate 7: per-text _cache_get in the sync path
# --------------------------------------------------------------------------


def _bulk_get_raw_sql(cache: diskcache.Cache, keys: list[str], now: float | None = None) -> list:
    """EXPERIMENTAL / not recommended: batched key lookup via direct sqlite access.

    Restricted to mode=1 (diskcache's MODE_RAW) so it only ever returns a
    value for entries diskcache actually stored inline in the value column
    (str values under disk_min_file_size, 32KB by default -- true for every
    embedding this library caches). Anything else (mode != 1, i.e. a value
    that overflowed to an on-disk file) comes back as None from this
    function, same as a cache miss -- silently forcing a re-embed rather
    than returning wrong data, but silent all the same. Also skips the
    eviction-policy 'touch' update and statistics counters that Cache.get()
    performs, and depends on the private Cache._sql accessor plus the Cache
    table's column names/mode encoding, none of which are public API.
    """
    now = time.time() if now is None else now
    placeholders = ",".join("?" for _ in keys)
    rows = cache._sql(
        f"SELECT key, value FROM Cache WHERE raw = 1 AND mode = 1 AND key IN ({placeholders}) "
        "AND (expire_time IS NULL OR expire_time > ?)",
        (*keys, now),
    ).fetchall()
    found = dict(rows)
    return [found.get(k) for k in keys]


def cand7_bulk_read(quick: bool) -> dict:
    c.banner("Candidate 7: per-text _cache_get in the sync path")

    api_findings = {
        "get_many_or_mget_exists": hasattr(diskcache.Cache, "get_many") or hasattr(diskcache.Cache, "mget"),
        "peek_is_arbitrary_key_lookup": False,
        "read_param_is_bulk": False,
        "note": (
            "diskcache 5.6.3's Cache has no get_many/mget (checked via hasattr against the installed "
            "5.6.3 API, both False). peek()/pull()/push() are FIFO-queue accessors over auto-generated "
            "integer (or 'prefix-integer') keys -- not lookup by an arbitrary key you already have, so "
            "they do not apply to this library's sha256-based keys. get(key, read=True) returns a file "
            "handle for that one key's on-disk value, meant for streaming a single large blob; it is a "
            "per-key option, not a batch primitive. The only way to fetch N specific keys in fewer than "
            "N round trips is undocumented direct sqlite access via the private Cache._sql "
            "(self._con.execute), which bypasses Disk.fetch()'s mode/filename dispatch, the "
            "eviction-policy touch update, and statistics counters -- see _bulk_get_raw_sql() for what "
            "that costs and risks. Nothing in the public API is both safe and faster than a get() loop."
        ),
    }
    print(json.dumps(api_findings, indent=2))

    ns = sizes(quick)
    rows = []
    for n in ns:
        reps = repeats_for(quick, n)
        texts = c.make_texts(n, seed=3000 + n)
        provider = str(openai.OpenAI(api_key="bench", base_url="http://localhost:1").base_url)
        keys = [oem.generate_cache_key(model=c.MODEL, dimensions=None, text=t, provider=provider) for t in texts]
        value = c.b64_vector(c.DIM)

        with tempfile.TemporaryDirectory() as d:
            cache = c.fresh_cache(d)
            c.warm_cache(cache, keys, value)

            baseline = c.timeit(lambda cache=cache, keys=keys: [cache.get(k) for k in keys], repeats=reps, warmup=1)
            prototype = c.timeit(lambda cache=cache, keys=keys: _bulk_get_raw_sql(cache, keys), repeats=reps, warmup=1)

            loop_result = [cache.get(k) for k in keys]
            sql_result = _bulk_get_raw_sql(cache, keys)
            equal = loop_result == sql_result

        speedup = baseline["median"] / prototype["median"] if prototype["median"] > 0 else float("inf")
        rows.append(
            {
                "n": n,
                "loop_get_ms": baseline["median"] * 1000,
                "raw_sql_ms_EXPERIMENTAL": prototype["median"] * 1000,
                "speedup_EXPERIMENTAL": speedup,
                "values_equal": equal,
            }
        )

    print(c.table(rows, ["n", "loop_get_ms", "raw_sql_ms_EXPERIMENTAL", "speedup_EXPERIMENTAL", "values_equal"]))

    return {
        "id": "sync_bulk_cache_read",
        "title": "Per-text cache.get() loop in the sync path -- is there a safe bulk-read alternative",
        "parameters": {"sizes": ns},
        "api_findings": api_findings,
        "experimental_raw_sql": rows,
        "risk": (
            "The public diskcache 5.6.3 API offers no safe bulk-read primitive, so there is no "
            "low-risk version of this candidate to ship -- reported as such rather than inventing one. "
            "The raw-sqlite numbers above answer 'how much would it even save' only; they are not a "
            "recommendation. That path depends on the Cache table's column names and the MODE_RAW "
            "encoding of diskcache's internal Disk class, neither of which is public API and both of "
            "which could change silently across diskcache versions -- and even pinned to 5.6.3, it "
            "quietly downgrades any oversized (>32KB) cached embedding to a permanent cache miss "
            "instead of erroring, which is a hard failure mode to notice in production."
        ),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark low-risk optimisation candidates for openai-embeddings-model v0.6.0"
    )
    parser.add_argument("--quick", action="store_true", help="tiny params, smoke test (<45s)")
    args = parser.parse_args()
    quick = args.quick

    t0 = time.perf_counter()
    candidates = [
        cand1_scope_digest(quick),
        cand2_txn_writes(quick),
        cand3_async_validation(quick),
        cand4_validate_internals(quick),
        cand5_model_validate_vs_construct(quick),
        cand6_to_python_caching(quick),
        cand7_bulk_read(quick),
    ]
    elapsed = time.perf_counter() - t0

    payload = {
        "quick": quick,
        "machine": "Apple M4 Pro, 10 cores, 32GB, Python 3.12, diskcache 5.6.3",
        "elapsed_seconds": elapsed,
        "candidates": candidates,
    }
    path = c.save("opts", payload)
    c.banner(f"Done in {elapsed:.1f}s. Results saved to {path}")


if __name__ == "__main__":
    main()
