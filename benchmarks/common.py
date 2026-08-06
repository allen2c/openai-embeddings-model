"""Shared benchmark harness for openai-embeddings-model v0.6.0.

Rules for every benchmark built on this:

- No network, no API key. A real `openai` client is constructed against
  `base_url="http://localhost:1"` so the library's isinstance checks pass,
  then `embeddings.create` is replaced with a local fake.
- Provider variability is deliberately excluded. The fake's latency is a
  parameter, not a measurement, so every number here is *our* code.
- Responses are built with `model_construct` and carry base64 embeddings, so
  the openai SDK's own parsing does not show up in the timings. That mirrors
  production: the library always requests `encoding_format="base64"`.
- Every script writes `results/<name>.json` so the write-up is assembled from
  data rather than from scrollback.
"""

from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import random
import statistics
import sys
import time
import typing

import diskcache
import numpy as np
import openai
import tiktoken
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage as OpenAIUsage

# Run against the working tree, not whatever is installed: `python
# benchmarks/x.py` puts only `benchmarks/` on the path, so the repo root has to
# be added before the library can be imported.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
    OpenAIEmbeddingsModel,
    generate_cache_key,
)


def legacy_cache_key(model, text: str, settings: ModelSettings) -> str:
    """Key generation as it worked before 0.6.0: one full call per text.

    0.6.0 replaced this with `_cache_keys_for`, which digests the request scope
    once rather than once per text, and dropped the per-text helper. The old
    shape lives on here so the benchmarks that measured the change still have
    a baseline to compare against.
    """
    return generate_cache_key(
        model=model._model_str,
        dimensions=settings.dimensions,
        text=text,
        provider=model._provider,
        extra_body=settings.extra_body,
    )


MODEL = "text-embedding-3-small"
DIM = 1536

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

_ENCODING = tiktoken.encoding_for_model(MODEL)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

_WORDS = [
    "retrieval",
    "augmented",
    "generation",
    "embedding",
    "vector",
    "similarity",
    "search",
    "index",
    "chunk",
    "document",
    "corpus",
    "latency",
    "throughput",
    "cache",
    "disk",
    "memory",
    "tenant",
    "request",
    "batch",
    "token",
    "limit",
    "provider",
    "model",
    "dimension",
    "semantic",
    "ranking",
    "relevance",
    "passage",
    "query",
    "answer",
    "context",
    "window",
    "pipeline",
    "ingestion",
    "evaluation",
]


def make_texts(n: int, *, seed: int = 0, words: int = 180) -> list[str]:
    """Distinct texts of a realistic RAG-chunk length (~180 words, ~230 tokens).

    Distinct matters: `deduplicate_texts` collapses repeats, so a benchmark
    built from one repeated string would measure deduplication, not embedding.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = " ".join(rng.choice(_WORDS) for _ in range(words))
        out.append(f"doc-{seed}-{i} {body}")
    return out


def token_length(text: str) -> int:
    return len(_ENCODING.encode(text))


def b64_vector(dim: int = DIM, *, seed: int = 0) -> str:
    """One realistic base64 float32 embedding string."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(dim).astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode()


# --------------------------------------------------------------------------
# fake provider
# --------------------------------------------------------------------------


def build_response(n: int, *, dim: int = DIM, vector: str | None = None) -> CreateEmbeddingResponse:
    vec = vector if vector is not None else b64_vector(dim)
    return CreateEmbeddingResponse.model_construct(
        data=[Embedding.model_construct(embedding=vec, index=i, object="embedding") for i in range(n)],
        model=MODEL,
        object="list",
        usage=OpenAIUsage(prompt_tokens=n, total_tokens=n),
    )


def sync_create(latency: float = 0.0, dim: int = DIM, calls: list | None = None) -> typing.Callable:
    """Blocking fake `embeddings.create`; `latency` stands in for the provider."""
    vec = b64_vector(dim)

    def create(input, **kwargs):
        items = list(input)
        if calls is not None:
            calls.append(len(items))
        if latency:
            time.sleep(latency)
        return build_response(len(items), dim=dim, vector=vec)

    return create


def async_create(latency: float = 0.0, dim: int = DIM, calls: list | None = None) -> typing.Callable:
    """Awaitable fake `embeddings.create`."""
    vec = b64_vector(dim)

    async def create(input, **kwargs):
        items = list(input)
        if calls is not None:
            calls.append(len(items))
        if latency:
            await asyncio.sleep(latency)
        return build_response(len(items), dim=dim, vector=vec)

    return create


def sync_model(create, *, base_url: str = "http://localhost:1", **kwargs) -> OpenAIEmbeddingsModel:
    client = openai.OpenAI(api_key="bench", base_url=base_url)
    client.embeddings.create = create  # type: ignore[method-assign]
    return OpenAIEmbeddingsModel(model=MODEL, openai_client=client, **kwargs)


def async_model(create, *, base_url: str = "http://localhost:1", **kwargs) -> AsyncOpenAIEmbeddingsModel:
    client = openai.AsyncOpenAI(api_key="bench", base_url=base_url)
    client.embeddings.create = create  # type: ignore[method-assign]
    return AsyncOpenAIEmbeddingsModel(model=MODEL, openai_client=client, **kwargs)


def settings(**kwargs) -> ModelSettings:
    return ModelSettings(**kwargs)


# --------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------


def fresh_cache(path: pathlib.Path | str, **kwargs) -> diskcache.Cache:
    """A cache in its own directory. Callers own the cleanup."""
    return diskcache.Cache(directory=str(path), **kwargs)


def warm_cache(cache: diskcache.Cache, keys: typing.Sequence[str], value: str) -> None:
    with cache.transact():
        for key in keys:
            cache.set(key, value)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def stats(samples: typing.Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def timeit(fn: typing.Callable[[], typing.Any], *, repeats: int = 5, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return stats(samples)


async def atimeit(fn: typing.Callable[[], typing.Awaitable], *, repeats: int = 5, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        await fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        await fn()
        samples.append(time.perf_counter() - t0)
    return stats(samples)


class LoopLagProbe:
    """Measure how long the event loop is unavailable to other tasks.

    A background task asks for a 5 ms sleep in a tight loop. Anything above
    that interval is time the loop spent unable to run it — i.e. blocked by
    synchronous work somewhere else on the loop. This is the number that
    decides whether a library is safe to embed in a web server.
    """

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.lags: list[float] = []
        self._task: asyncio.Task | None = None
        self._stop = False

    async def _run(self) -> None:
        while not self._stop:
            t0 = time.perf_counter()
            await asyncio.sleep(self.interval)
            self.lags.append(time.perf_counter() - t0 - self.interval)

    async def __aenter__(self) -> LoopLagProbe:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc) -> None:
        self._stop = True
        if self._task is not None:
            await self._task

    def summary(self) -> dict[str, float]:
        if not self.lags:
            return {"n": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        return stats(self.lags)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def save(name: str, payload: dict) -> pathlib.Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def table(rows: list[dict], columns: list[str]) -> str:
    """Render rows as a markdown table for eyeballing during a run."""
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)
