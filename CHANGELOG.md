# Changelog

All notable changes to this project are documented here.

## [0.5.2] - 2026-08-06

Correctness release. Every change here turns previously broken behaviour into
correct behaviour — no API removals, no signature changes, and the cache key
format is untouched, so existing caches stay valid.

### Fixed

- **`get_similarity()` crashed with exactly one document.** A single document
  produced a `(1, 1)` similarity matrix, and `squeeze()` collapsed it to a
  non-iterable 0-d array, raising `TypeError: iteration over a 0-d array`.
  Both the sync and async paths were affected.
- **Provider responses were consumed in arrival order, ignoring `index`.**
  OpenAI documents that `data` comes back in request order, but proxies and
  other OpenAI-compatible providers make no such promise. Results are now
  ordered by the `index` each item reports, falling back to arrival order when
  a provider omits it. A reordered response previously misaligned every
  embedding with its text — silently, and the wrong vectors were written to
  the cache.
- **The sync path crashed when a provider omitted `usage`.** The async path
  already fell back to a local tiktoken count; the sync path went straight to
  `response.usage.prompt_tokens` and raised `AttributeError`. Both paths now
  share one implementation, so they can no longer drift apart.
- **`NaN` relevance scores could rank as the best match.** `NaN` compares
  `False` against everything, so sorting on the raw score could leave a `NaN`
  result at position 0 ahead of a perfect `1.0` match. `NaN` now sorts last.
- **`to_numpy()` handed out a read-only view of an internal buffer.** Native
  libraries that write through raw pointers (faiss, for example) ignore
  numpy's read-only flag, so `faiss.normalize_L2(response.to_numpy())`
  silently corrupted the decoded buffer and every later `to_numpy()` /
  `to_python()` call on that response. `to_numpy()` now returns a writable
  copy; the decode itself is still cached.
- **Unusable cache entries were coerced with `str()` and trusted.** An entry
  written by another tool or an older version could pass through as a
  syntactically valid but meaningless vector — in the worst case decoding
  cleanly into garbage floats with no error anywhere. Entries are now
  validated (type, base64, buffer length, expected dimensionality, finiteness)
  and treated as a cache miss when they fail.

### Added

- **`AsyncOpenAIEmbeddingsModel.aclose()` and async context manager support.**
  Each instance owns a `ThreadPoolExecutor` and there was no way to release it
  deterministically. `async with AsyncOpenAIEmbeddingsModel(...) as model:` and
  `await model.aclose()` now shut the pool down; a `__del__` safety net covers
  instances dropped without either. Note that CPython already reaps an
  executor's workers once the executor itself becomes unreachable — the case
  this fixes is a pool that outlives its model because something else still
  references it.
- **`tests/test_regressions.py`** — an offline regression suite covering every
  fix above. It needs no API key and makes no network calls, so these paths
  are verifiable in CI for the first time.
- `[tool.pytest.ini_options]` in `pyproject.toml` pinning `asyncio_mode`.

## [0.5.1] - 2026-03-19

- Dependency updates.

## [0.5.0] - 2026-03-01

- Added `get_similarity(query, documents, model_settings)` returning a
  `SimilarityResponse` sorted by relevance score.
- Added Voyage AI support via the `output_dimension` extra body parameter.
- Added `extra_body` to `ModelSettings`.
- Extracted `_OpenAIEmbeddingsModelBase` shared by the sync and async models.
- Async model gained a dedicated `ThreadPoolExecutor` for cache I/O.
- Added documentation site.
