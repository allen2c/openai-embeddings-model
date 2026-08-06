# Changelog

All notable changes to this project are documented here.

## [0.6.0] - 2026-08-06

Behaviour release. These are the fixes deferred from 0.5.2 because each one
changes what existing code does — silently accepted settings now raise, and
the cache key layout changed.

### ⚠️ Breaking

- **Existing caches are invalidated.** The cache key now includes a version
  prefix, the client's `base_url`, and `extra_body`, so 0.5.x entries are
  ignored rather than misread. **The first run after upgrading re-embeds
  everything, at full provider cost.** Budget for it on large corpora. Old
  entries are not deleted, so a rollback still finds its cache intact; clear
  the directory yourself once you no longer need one.

  This closes a silent-corruption bug: two clients pointing at different
  providers but sharing a cache directory and a model name returned each
  other's vectors, and requests differing only in a provider parameter such as
  Voyage's `output_dimension` collided. Neither was detectable from the
  result.

- **`ModelSettings.validate_for_model` now rejects invalid dimensions.** Its
  `ValueError` raises previously sat inside its own `except ValueError`, so it
  accepted anything for any model. Settings that used to pass silently and
  fail later at the provider — `dimensions` above a model's maximum, or any
  `dimensions` on `text-embedding-ada-002` — now raise before the request.
  Unknown model names are still left alone.

- **Invalid constructor arguments now raise.** `token_limit_usage_percent`
  outside `(0, 100]`, and non-positive `max_batch_size`, `max_input_tokens`,
  `max_tokens_a_request`, or negative `max_retries`, raise `ValueError`.
  `token_limit_usage_percent=0` previously sent empty strings to the provider,
  and a negative value hit Python's negative slicing and kept everything
  *except* the last N tokens — the opposite of a cap.

- **A caller-supplied `encoding=` now takes precedence.** It was previously
  ignored whenever tiktoken recognised the model name, meaning it only applied
  when you least needed it.

- **`extra_body` that cannot be serialised raises `ValueError`** at the call
  site instead of failing from inside the HTTP layer.

- **`ModelResponse` is frozen.** Its decoded array is cached on first access
  and never invalidated, so reassigning `output` left `to_numpy()` returning
  vectors for text the response no longer held. Code that assigns to
  `response.output` or `response.usage` now raises `pydantic.ValidationError`.

### Fixed

- **Requests are no longer built past the provider's token limit.**
  `MAX_TOKENS_A_REQUEST` was declared but never used; batches split on item
  count alone, so 2048 long texts produced a single multi-million-token
  request that any provider rejects. Batches now respect a token budget as
  well, configurable via `max_tokens_a_request`.
- **A failed batch no longer discards work already paid for.** Caching was
  deferred until every batch returned, so one failure at batch 5 of 10 threw
  away four batches of billed embeddings. Each batch is now cached the moment
  it succeeds.
- **Rate limits are retried.** The code logged advice to add exponential
  backoff without implementing it. Transient failures (rate limits, timeouts,
  connection errors, 5xx) now retry with exponential backoff, controlled by
  `max_retries` and `retry_base_delay`. Non-transient errors still fail
  immediately.
- **A failing async batch now cancels its siblings.** `asyncio.gather` left
  them running, so a call that had already raised kept issuing requests the
  caller would never see. Replaced with `asyncio.TaskGroup`; the original
  provider exception is still what propagates.
- **Voyage detection no longer matches on substring.** `my-voyage-clone` was
  treated as a Voyage model and had its `dimensions` rerouted. Detection now
  matches a `voyage` prefix, and `dimensions_parameter` sets it explicitly for
  deployment aliases that hide the underlying model.
- **`extra_body` overriding `dimensions` now warns** instead of silently
  producing vectors of a size the caller never asked for.
- **Text carrying lone surrogates no longer aborts the batch** during cache
  key generation.
- **Non-OpenAI models warn once** that token counts, and therefore truncation
  points, are approximate, rather than silently using gpt-4o's tokenizer.
  A non-`str` model no longer raises an uncaught `AttributeError`.
- **Cache keys no longer collide on falsy values.** `dimensions=0` shared a
  key segment with `dimensions=None`, and `model=""` with the literal
  `"unknown"`, so unrelated entries could be served for one another.
- **`extra_body` with mixed key types no longer crashes the call.** Building
  the cache scope digest sorted the original key objects, so a dict mixing
  `str` and `int` keys raised `TypeError` — even with caching disabled, since
  the key is built before the request.
- **A second, differently-caused batch failure is no longer lost.** Only one
  exception can be raised, but the others are now logged; previously they
  vanished from both the raised error and the traceback.
- **A nested `ExceptionGroup` from a batch is unwrapped to a concrete error**,
  so `except openai.RateLimitError` matches instead of silently missing.
- **Truncated texts are re-measured before batching.** Cutting mid-codepoint
  makes `decode()` insert U+FFFD, which re-encodes to an extra token;
  assuming truncation landed exactly on the limit undercounted the request
  and could overfill a batch past `max_tokens_a_request`.
- **The async cache write is shielded from sibling cancellation**, so a batch
  that was billed is persisted even if another batch fails while its write is
  still queued.
- **Forked children rebuild what the fork invalidated.** Models register an
  `os.register_at_fork` hook that recreates the thread pool, whose workers do
  not survive a fork, and drops the inherited sqlite connection. Constructing
  a model before forking — the gunicorn `preload_app` shape — previously left
  the child submitting work to a pool with no threads.

### Added

- **`Usage.truncated_texts`** reports how many texts were shortened to fit the
  token limit. The default `truncate` policy drops input with no signal in the
  result; this makes the loss visible. Truncation also logs at warning level
  rather than debug.
- **Repeated texts are embedded once.** Duplicates within one call previously
  each cost a provider slot.
- **`max_concurrent_batches`** (async) exposes what was a hardcoded 5.
- `get_default_cache`, `generate_cache_key`, and `CACHE_KEY_VERSION` are
  exported for cache management.

### Performance

Measured, not guessed. The full write-up, method, and the experiments that
rejected four other candidates are in [Benchmarks](benchmarks.md).

- **`executor_max_workers` now defaults to `1`**, not `min(32, cpu_count + 4)`.
  Everything the async model's pool runs is GIL-bound — sqlite through
  diskcache, and tiktoken — so the extra workers were buying contention rather
  than parallelism. Thirty-two concurrent all-hit calls finish **4.3x faster**
  on one worker, and every larger value measured worse, on cache hits and
  misses alike. `aiosqlite` reaches the same design from the same constraint.
  The parameter stays, because one worker is right only for a *local* cache.
  Against a cache that blocks on I/O — remote, or your own object over the
  network — the answer inverts: at 0.5 ms per read, eight workers were 6.9x
  faster than one; at 10 ms, fourteen were 8.0x faster. **Raise it to roughly
  your concurrency if your cache is remote.** `None` restores the stdlib
  default.
- **A batch's cache writes now share one sqlite transaction.** Every `set()`
  was its own `BEGIN`/`COMMIT` round trip. Batching them is **1.6-2.4x** faster
  on the write path — steady from an empty cache to one holding 200,000
  entries — and takes 1.4-1.6x off a full uncached call.

  The trade, in exchange: a write that fails partway through a batch now
  discards that whole batch instead of leaving the earlier entries committed.
  Those embeddings were already paid for, so a mid-batch disk failure costs
  re-embedding the batch rather than just its tail. The caller still receives
  every vector either way — only the cached copy is lost.
- **The cache key digests the request scope once per request**, not once per
  text. It is a `json.dumps` plus a sha256 over the provider and `extra_body`,
  both fixed for the call, and it was two thirds of key generation — 3.3 ms
  wasted on a 2048-text request. **Keys are byte-identical**; no cache is
  invalidated by this.
- `ModelResponse.to_python()` never cached its result despite the docstring
  saying so. The docstring was wrong, not the code — caching it would hand
  every caller the same mutable list, which is what `to_numpy()`'s copy exists
  to prevent. Keep the result if you need it twice.

### Changed

- Async cache reads and writes are batched into one executor job each, instead
  of one cross-thread round trip per key. The speedup scales with key count
  and varies by machine; measurements on 2000 keys ranged from roughly 5x to
  35x.
- Async token counting runs in the executor rather than on the event loop,
  which it previously stalled for the duration of the call (~620ms for 2048
  texts).
- `generate_cache_key` no longer caches on the raw text, which had been
  pinning up to 2048 full documents in memory for the life of the process.
- **`str_or_none` is no longer a dependency.** It was declared as a runtime
  requirement but only ever used by the test suite, so installing this package
  pulled it in for nothing. Nothing in the public API changes.

### Internal

- Line length is now 120. Formatting is black + isort, linting is ruff, and
  every setting lives in `pyproject.toml`; the unused flake8 configuration is
  gone. `make fmt` runs all of it plus a gitleaks scan, and `pyright` is clean.

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
