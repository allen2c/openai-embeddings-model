# Benchmarks

Where this library spends its time, how it behaves under concurrency, and
which knobs actually move the numbers.

Every measurement here replaces the provider with a local fake whose latency
is a parameter. Nothing crosses the network, so nothing below is a claim about
OpenAI — it is a claim about this code.

!!! info "Test bench"

    Apple M4 Pro, 10 cores (4 performance + 6 efficiency), 32 GB,
    macOS (Darwin 25.6) · Python 3.12.13 ·
    diskcache 5.6.3 · numpy 2.4.6 · openai 2.53.0 · tiktoken 0.13.0 ·
    `text-embedding-3-small`, 1536 dimensions, ~185-token texts.

---

## The short version

| Finding | Number | Status |
| --- | --- | --- |
| Threads help only when you are waiting on the provider | **25.7×** at 32 threads on cache misses; **0.22×** on cache hits | Size your pool for provider latency, not for cores |
| The async thread pool was oversized | one worker finishes **4.3× faster** than the old `cpu_count + 4` default | **Fixed in 0.6.0** — `executor_max_workers` now defaults to 1 |
| The cache key digested the request scope once per text | **3.28 ms** wasted per 2048-text request | **Fixed in 0.6.0** — 2.9× on key generation, keys unchanged |
| Cache writes were one sqlite transaction each | **1.6–2.4×** faster batched | **Fixed in 0.6.0** — one transaction per provider batch |
| Cache hits cost more to validate than to read | `validate_cached_embedding` is **54%** of an all-hit call | Kept — it is what makes a foreign cache entry safe |
| A large call blocks the event loop | up to **69 ms**; 40% of a 5 ms request cadence missed | Chunk large inputs, or keep them off the request loop |
| `max_concurrent_batches=5` is the right default | 3.0× over serial; no gain past 5 | Left alone |
| A partly-warm cache costs in proportion to its **misses**, over a floor set by the request size | 4096 texts cost 69 ms even at 100% hits | Send only what you know is new, if you track that |
| The throughput ceiling is one process serving **50,571 cache hits/sec**, and base64 decoding is 65% of it | half that decoding is redundant — worth **1.44×** | Next release; see [section 8](#8-where-the-ceiling-is-and-what-is-left) |

---

## Method

A real `openai` client is built against `base_url="http://localhost:1"` so the
library's `isinstance` checks pass, then `embeddings.create` is replaced with a
local function that returns base64 vectors after an optional `sleep`. Responses
use `model_construct`, so the SDK's own parsing does not pollute the timings —
which matches production, since the library always requests
`encoding_format="base64"` and never pays to parse float arrays.

Caches live in `tempfile` directories. Timings are medians over repeated trials
with a warm-up, `time.perf_counter` throughout. Threads start on a
`threading.Barrier` so the aggregate window is measured rather than the sum of
per-thread times.

---

## 1. The sync model across threads

One `OpenAIEmbeddingsModel` and one `diskcache.Cache` shared by every thread —
the Flask/Django worker-thread shape.

**Cache hits** (16 texts per call, fully warm):

| Threads | texts/sec | median call | p95 | vs 1 thread |
| --- | --- | --- | --- | --- |
| 1 | 61,530 | 0.26 ms | 0.27 ms | 1.00× |
| 2 | 44,600 | 0.70 ms | 1.03 ms | 0.72× |
| 4 | 31,210 | 1.99 ms | 2.80 ms | 0.51× |
| 8 | 15,130 | 8.51 ms | 10.8 ms | 0.25× |
| 16 | 13,270 | 19.2 ms | 23.8 ms | 0.22× |
| 32 | 13,230 | 38.4 ms | 47.7 ms | 0.22× |

**Cache misses**, 50 ms of fake provider latency per batch:

| Threads | texts/sec | median call | vs 1 thread | without cache |
| --- | --- | --- | --- | --- |
| 1 | 137 | 58.2 ms | 1.00× | 1.00× |
| 4 | 554 | 57.2 ms | 4.06× | 3.91× |
| 8 | 1,115 | 56.5 ms | 8.17× | 7.68× |
| 16 | 2,195 | 56.1 ms | 16.1× | 15.4× |
| 32 | 3,514 | 56.2 ms | 25.7× | 30.7× |

Two different worlds. On the miss path threads scale nearly linearly to 32,
because every thread spends its life inside a socket read. On the hit path
adding threads makes the work slower in absolute terms — one thread already
does 61k texts/sec, and thirty-two do 13k.

The gap between the cached and uncached miss columns at 32 threads (25.7× vs
30.7×) is the cache reads and writes beginning to contend. It is real but
second-order next to a 50 ms round trip.

!!! tip "Sizing the pool"

    Threads buy you concurrency against the provider and nothing else. Pick the
    pool size from the provider's latency and rate limit. A thread pool sized
    to the core count is the wrong instinct here.

**Correctness held throughout**: 16 threads × 150 calls, overlapping and
distinct texts on a shared model and shared cache — 2,400 calls, 16,098 vectors
verified against what was cached, zero mismatches, zero exceptions.

---

## 2. Where an all-hit call goes

A single-threaded 512-text call that hits the cache for everything, broken into
its parts (50 repeats):

| Component | µs | % |
| --- | --- | --- |
| `validate_cached_embedding` × 512 | 4,180 | 53.6% |
| raw `cache.get` × 512 | 1,630 | 20.9% |
| key generation × 512 | 1,273 | 16.3% |
| unaccounted | 668 | 8.6% |
| `deduplicate_texts` | 25 | 0.3% |
| `validate_input` | 19 | 0.2% |
| `ModelResponse.model_validate` | 4 | 0.06% |
| **measured call** | **7,799** | **100%** |

Measured before the 0.6.0 fixes, so the key-generation row is the number that
motivated hoisting the scope digest — it is about a third lower now.

Validating a cached entry costs more than three times reading it: 7.97 µs
against 2.69 µs. Inside that, 84% is `base64.b64decode`, 11% is
`np.isfinite(...).all()`, 5% is the type and length checks. This is the price of
the 0.6.0 guarantee that a foreign or corrupted cache entry is treated as a miss
rather than silently decoded into a meaningless vector — worth knowing, and
worth keeping.

The unaccounted 8.6% is `validate_for_model`, the two list comprehensions that
compute `missing` and `cache_hits`, `Usage` construction, and the `logger.debug`
f-strings — which are formatted on every call even though the default log level
throws the record away.

**Tokenisation** is the other CPU cost, and it only appears on the miss path.
`_prepare_batches` runs tiktoken's `encode_batch` at **24–28 µs per text**:

| Texts | `_prepare_batches` | µs/text |
| --- | --- | --- |
| 128 | 3.1 ms | 24.0 |
| 512 | 12.4 ms | 24.3 |
| 2048 | 50.6 ms | 24.7 |
| 4096 | 115.1 ms | 28.1 |

A 4096-text request spends 115 ms tokenising before it sends anything, and
produces just 3 provider batches. It does not parallelise either — running
`_prepare_batches` on 4 threads yields 0.80× the single-thread throughput, on 8
threads 0.78×. The async model already moves this into its executor, which is
the reason that decision exists.

**`get_embeddings_generator`** does not pay for itself on wall-clock: 4096
texts took +38.6% longer in chunks of 256, for −0.9% peak RSS. Chunking splits
one large provider request into many small ones. Reach for it when the caller
genuinely cannot hold every response at once, not for speed.

---

## 3. The async model and the event loop

### How long the loop is unavailable

Maximum stall observed while one `get_embeddings` runs, measured by a task
asking for a 5 ms sleep in a tight loop:

| Texts | all hits | all misses | no cache |
| --- | --- | --- | --- |
| 64 | 0.6 ms | 0.5 ms | 0.3 ms |
| 512 | 3.0 ms | 3.5 ms | 14.9 ms |
| 2048 | 16.1 ms | 14.1 ms | 37.0 ms |
| 4096 | 31.5 ms | 69.3 ms | 27.3 ms |

`_cache_get_many` and `_prepare_batches` do go through the executor, as
designed. What stays on the loop is everything around them: `validate_input`,
`deduplicate_texts`, key generation, the
`validate_cached_embedding` comprehension inside `_cache_get_many` — which
offloads only the raw `cache.get` calls, not the validation of their results —
and the final `ModelResponse` build.

Turned into something a service owner can act on: while a 4096-text all-hit
call ran, a simulated 5 ms request cadence **missed 40% of its ticks, the worst
by 34.6 ms**.

!!! warning "Large calls are not free for your other traffic"

    Anything past a few hundred texts will visibly delay concurrent requests on
    the same loop. Feed large corpora through `get_embeddings_generator`, or
    run them somewhere other than the loop serving user requests.

### `executor_max_workers`

32 concurrent 512-text all-hit calls on one shared model:

| Workers | wall | p50 | p95 |
| --- | --- | --- | --- |
| **1** | **0.273 s** | 145 ms | 226 ms |
| 2 | 0.346 s | 186 ms | 300 ms |
| 4 | 0.489 s | 280 ms | 444 ms |
| 8 | 1.116 s | 796 ms | 1072 ms |
| 14 *(default)* | 1.175 s | 999 ms | 1125 ms |
| 32 | 1.306 s | 1166 ms | 1284 ms |
| 64 | 1.250 s | 1118 ms | 1227 ms |

One worker is **4.3× faster than the default** and monotonically better than
every larger value. `ThreadPoolExecutor(max_workers=None)` resolves to
`min(32, cpu_count + 4)` — 14 here — and every one of those threads past the
first is pure contention.

The obvious objection is that the executor also runs `_prepare_batches`, so a
single worker should make concurrent callers queue behind each other's
tokenisation. Measured across 16 concurrent 512-text calls, it does not:

| Workers | all hits | misses, 0 ms provider | misses, 20 ms provider |
| --- | --- | --- | --- |
| **1** | **1.00×** | **1.00×** | **1.00×** |
| 4 | 1.79× | 1.09× | 1.11× |
| 14 | 3.77× | 1.27× | 1.23× |
| 32 | 4.12× | 1.47× | 1.28× |

(Lower is better; 1.00× is the winner.) One worker wins every workload. Extra
threads never add parallelism to GIL-bound work — see [section 4](#4-is-the-disk-cache-the-bottleneck).

### `max_concurrent_batches`

8192 texts, 41 batches, 50 ms per batch:

| Setting | wall |
| --- | --- |
| 1 | 3.289 s |
| 2 | 1.740 s |
| **5** *(default)* | **1.112 s** |
| 10 | 1.111 s |
| 20 | 1.132 s |
| 50 | 1.201 s |

The default is the knee of the curve: 3.0× over serial, with nothing left on
the table. Past 5 it flattens and then reverses — 41 simultaneous in-flight
batches is more bookkeeping for no more throughput, and in production it is
also the fastest way to a rate-limit storm.

### Concurrency, and what saturates

N concurrent 128-text all-hit calls on one shared model:

| Concurrent calls | texts/sec | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| 1 | 58,520 | 2.1 ms | 2.1 ms | 2.1 ms |
| 4 | 29,750 | 14.6 ms | 16.3 ms | 16.3 ms |
| 16 | 14,070 | 131 ms | 138 ms | 138 ms |
| 64 | 13,010 | 401 ms | 585 ms | 603 ms |
| 256 | 14,160 | 1,138 ms | 2,161 ms | 2,222 ms |

Throughput floors at ~13–14k texts/sec and stays there; latency grows linearly
with the queue. The system has one lane, and past about 16 concurrent calls you
are only lengthening it.

Splitting the load over 4 model instances — 4 separate executors against one
cache directory — made it worse, not better (1.257 s against 1.089 s), while
costing four thread pools. Async narrowly beat an equivalent 32-thread sync
pool in both regimes (1.170 s vs 1.218 s on hits, 1.990 s vs 2.246 s on misses
with latency), so the choice between them is an architectural one, not a
performance one.

**Lifecycle**: create + one call + `aclose()` costs 4.0 ms median. 200
create-without-`aclose` cycles left the thread count flat at baseline with and
without a forced `gc.collect()` — `__del__` reclaims the pool. `aclose()` is
still the right call, because it releases the threads at a moment you choose.

---

## 4. Is the disk cache the bottleneck?

Single-threaded, it is not:

| Cache size | hit | miss |
| --- | --- | --- |
| 1,000 | 337k ops/s (2.96 µs) | 608k ops/s |
| 10,000 | 277k ops/s (3.61 µs) | 577k ops/s |
| 100,000 | 225k ops/s (4.45 µs) | 562k ops/s |

Writes run at 17.8k ops/s one at a time, 25.5k batched in a `transact()` block.
Value size barely matters: at 256, 768, 1536 and 3072 dimensions the base64
payload is 1.3–16 KB, all of it under diskcache's 32 KB `disk_min_file_size`, so
every embedding this library produces is stored inline in sqlite and the
file-spill path is never exercised. Reads range 362k → 254k ops/s across that
span, writes 23.1k → 14.0k.

Under threads it collapses:

| Threads | reads | vs 1 | writes | vs 1 |
| --- | --- | --- | --- | --- |
| 1 | 244k ops/s | 1.00× | 16.9k ops/s | 1.00× |
| 2 | 157k ops/s | 0.64× | 16.2k ops/s | 0.96× |
| 4 | 61k ops/s | 0.25× | 15.1k ops/s | 0.90× |
| 8 | 19k ops/s | 0.08× | 15.5k ops/s | 0.92× |
| 16 | 17k ops/s | 0.07× | 12.1k ops/s | 0.72× |

Writes flattening is expected — sqlite has one write lock, and no
`diskcache.Timeout` was raised anywhere in the run. Reads losing 93% of their
throughput is not, so it was worth chasing down. Every sqlite-side explanation
was tested and eliminated:

| Hypothesis | Test | Result |
| --- | --- | --- |
| The shared `Cache` object serialises | One `Cache` per thread, same directory | Identical collapse |
| diskcache overhead | Raw `sqlite3`, connection per thread | Identical collapse |
| Working set exceeds the page cache | 0.1 MB db vs 40 MB db | Identical collapse |
| Page cache / mmap too small | 256 MB `cache_size`, 1 GB `mmap_size` | No change |
| WAL index locking | `journal_mode=truncate` | Same collapse, and 39% slower on one thread |
| One database file is the contention point | `FanoutCache(shards=4/8/16)` | Reads no better (writes 1.15–1.21× at 2–4 threads) |
| Generic GIL preemption | Plain `dict` lookups | Flat to 16 threads (5.9M → 5.4M ops/s) |
| GIL switch interval | 0.5 ms / 5 ms / 50 ms | No change |
| The GIL handoff itself | Processes instead of threads | **Processes scale; threads collapse** |

That last one is decisive. Identical reads, identical database, workers timing
their own loops so interpreter startup is excluded:

| Workers | threads | vs 1 | processes | vs 1 |
| --- | --- | --- | --- | --- |
| 1 | 224k ops/s | 1.00× | 245k ops/s | 1.00× |
| 2 | 144k ops/s | 0.65× | 459k ops/s | 1.87× |
| 4 | 59k ops/s | 0.27× | 791k ops/s | 3.23× |
| 8 | 21k ops/s | 0.10× | 351k ops/s | 1.43× |

Processes scale to 3.2× on four workers — the machine's four performance cores
— and fall back at eight as the work spills onto efficiency cores. Threads
never scale at all. At eight workers, processes do **16× the throughput of
threads on the same query against the same file**.

What is left is the GIL release and reacquire that Python's `sqlite3` performs
around every query. The calls are ~3 µs each; the handoff around them is not
free, and with several threads doing nothing but short C calls the reacquisition
cost swamps the work. A `dict` never releases the GIL and scales flat; processes
have no shared GIL and scale properly; everything in between collapses.

The practical reading: **a shared disk cache is fast enough on one thread that
parallelising reads across threads is counterproductive.** 244k reads/sec is
already far beyond what any embedding workload asks for. If you genuinely need
more, the unit of scale is the process — a `gunicorn`/`uvicorn` worker per core,
each with its own model — and the library is already built for it, since
`_reset_after_fork` exists precisely to make a forked child's cache and
executor usable again.

!!! danger "Do not reach for `FanoutCache`"

    It works end-to-end here, fork path included, and it buys nothing on reads.
    It also **loses writes silently**. `FanoutCache.set` swallows the timeout
    the sharded `Cache` raises under contention and returns `False` instead:

    ```python
    try:
        return shard.set(key, value, expire, read, tag, retry)
    except Timeout:
        return False
    ```

    This library, like most callers, does not inspect that return value — so a
    dropped write is invisible, and the next request silently re-embeds and
    re-pays for a vector you already bought. A plain `diskcache.Cache` raises
    instead.

---

## 5. A partly-warm cache

Sections 1 to 4 measure 0% and 100% hit rates. Production lives between them:
a corpus grows by a few documents, a query overlaps yesterday's, a re-index
touches a subset.

The per-item work splits in two. Key generation, `cache.get` and
`validate_cached_embedding` run over **every** text. tiktoken, the provider
call and the cache write run over **only the misses**.

| Component | Runs over | Cost |
| --- | --- | --- |
| key generation + cache read + validate | every text | 15.2 µs/text |
| tiktoken (`_prepare_batches`) | misses only | 31.3 µs/text |

So the floor is set by how much you ask for and the slope by how much is
missing. 4096 texts, 20 ms per provider batch:

| Hit rate | Misses | sync | async | µs per text |
| --- | --- | --- | --- | --- |
| 0% | 4096 | 410.8 ms | 365.3 ms | 100.3 |
| 25% | 3072 | 332.8 ms | 308.3 ms | 81.3 |
| 50% | 2048 | 272.1 ms | 224.7 ms | 66.4 |
| 75% | 1024 | 161.3 ms | 151.7 ms | 39.4 |
| 90% | 410 | 125.3 ms | 114.8 ms | 30.6 |
| 99% | 41 | 96.2 ms | 87.1 ms | 23.5 |
| 100% | 0 | 68.9 ms | 61.6 ms | 16.8 |

Linear in the miss count, with no cliff at any batch boundary — scattered
misses collapse into few provider batches, because `_prepare_batches` only
ever sees the missing subset.

**The floor is the thing to notice.** At 100% hits, 4096 texts still cost
69 ms. That is the price of *asking*, paid on every text whether or not it is
cached, and it is why the biggest lever belongs to the caller rather than to
this library: at a 99% hit rate, passing all 4096 texts costs 96 ms, where
passing only the 41 you know are new would cost about 21 ms. If you already
track which documents changed, send those.

### Do the 0.6.0 fixes help here?

Yes, and most where there is most to do — the write transaction scales with
misses, so its contribution fades as the cache warms:

| Hit rate | sync 0.5.x → 0.6.0 | async 0.5.x → 0.6.0 |
| --- | --- | --- |
| 0% | 577.3 → 415.1 ms (**1.39×**) | 542.7 → 363.1 ms (**1.49×**) |
| 50% | 344.0 → 252.2 ms (1.36×) | 339.0 → 240.2 ms (1.41×) |
| 90% | 147.4 → 121.8 ms (1.21×) | 135.3 → 118.3 ms (1.14×) |
| 99% | 107.2 → 94.7 ms (1.13×) | 90.0 → 87.2 ms (1.03×) |
| 100% | 71.5 → 67.8 ms (1.05×) | 63.8 → 60.9 ms (1.05×) |

These are single calls, so `executor_max_workers` barely participates — its
4.3× needs concurrency, which is why sync and async track each other here.

### Loop blocking is worst at the extremes

| Hit rate | 0% | 25% | 50% | 75% | 90% | 99% | 100% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max loop lag | 36.7 ms | 20.6 ms | 15.6 ms | 24.0 ms | 28.9 ms | 34.9 ms | 35.3 ms |

A warm cache does not make the async model gentler on the event loop — it
makes it worse. Validation runs on the loop and only hits pay for it, so the
fully-cached call is close to the most disruptive one. The mid-range is
cheapest because neither the validation pass nor the response build dominates.

---

## 6. What was worth changing

Seven candidates were prototyped by subclassing the model or copying the
function under test — never by editing the library. Each was checked for
functional equivalence before its timing was believed.

### Done in 0.6.0

**Hoist `cache_scope_digest` out of the per-text loop.** Key generation used to
call `generate_cache_key` once per text, and each call re-ran a `json.dumps` and
a sha256 over a provider and `extra_body` that are constant for the whole
request. `_cache_keys_for` now digests the scope once and reuses it.
That digest is 67% of a `generate_cache_key` call. The `not provider and not
extra_body` fast path never fires in practice — a real client always carries a
`base_url`, verified for both the default and a custom one.

| Texts | baseline | hoisted | speedup |
| --- | --- | --- | --- |
| 2048, no `extra_body` | 5.10 ms | 1.74 ms | 2.93× |
| 2048, with `extra_body` | 7.40 ms | 1.77 ms | 4.19× |
| 4096, end-to-end all-hit call | 66.4 ms | 58.1 ms | 1.14× |

Keys verified byte-identical, so no cache is invalidated. **Risk: none.**

**Batch cache writes in a transaction.** Both `_embed_missing` (sync) and
`_cache_set_many` (async) wrote one entry per sqlite transaction. Now one
transaction per provider batch:

| Case | baseline | `transact()` | speedup |
| --- | --- | --- | --- |
| 512 writes | 30.5 ms | 16.2 ms | 1.89× |
| 4096 writes | 250 ms | 120 ms | 2.09× |
| 4096 writes into a 391 MB cache | 262 ms | 165 ms | 1.58× |
| 4096 writes into a 1.5 GB cache | 252 ms | 105 ms | 2.40× |
| 512-text all-miss call, sync | 54.8 ms | 38.5 ms | 1.42× |
| 4096-text all-miss call, async | 569 ms | 367 ms | 1.55× |

!!! note "Correction"

    An earlier run of this page reported the speedup decaying from 2.07× at
    512 writes to 1.23× at 4096, and offered sqlite's page-cache spilling as
    the explanation. Both were wrong. The decay was an artifact of that
    harness: it timed six repeats into a single cache directory, so the last
    repeat was writing into a database the first repeat had grown. Measured
    against a controlled cache the win is flat at **1.6–2.4× from empty to
    200,000 existing entries**, and varying sqlite's page cache from 4 MB to
    512 MB changes nothing on a fresh database.

**Risk: real, and now asserted in the test suite.** A transaction holds the
write lock for the whole batch instead of one entry at a time. WAL means it
does not block readers, and 2048 entries commit in roughly 100 ms — three
orders of magnitude inside diskcache's 60 s timeout — so concurrent writers
are not meaningfully affected.

What genuinely changes is failure granularity. Wrapped in `transact()`, a
write that raises partway through discards the whole batch; unwrapped, the
entries before the failure stayed committed. Verified rather than assumed —
injecting a failure at entry 256 of 512 leaves **256 entries committed
unwrapped and 0 wrapped**. Those embeddings were already paid for, so the cost
of a mid-batch disk failure rises from re-embedding the tail to re-embedding
the batch. It is accepted because cache writes fail only on a full or broken
disk, and the caller receives every vector either way — only the cached copy
is lost. `test_a_batch_of_cache_writes_is_one_transaction` pins this down so
nobody has to rediscover it.

### Not worth doing

| Candidate | Measured | Verdict |
| --- | --- | --- |
| Move `validate_cached_embedding` into the executor | wall-clock 1.00×; max loop lag got **worse** (4.0 → 8.2 ms at 512, 45.8 → 56.3 ms at 4096) | Rejected by its own measurement. The work is GIL-bound wherever it runs, so the loop waits either way and pays a thread handoff on top. |
| Streamline `validate_cached_embedding` internals | 0.99–1.02× | There is no redundant decode to remove: `np.frombuffer` is 0.20 µs and zero-copy against `b64decode`'s 7 µs. The cost *is* the base64 decode. |
| `model_construct` instead of `model_validate` for `ModelResponse` | 37× — on an operation costing 33 µs at 4096 texts | Saves 32 µs on a 60 ms call. Not worth giving up validation for. |
| Bulk read via raw sqlite | 1.6–2.3× | Needs `Cache._sql`, bypassing `Disk.fetch`'s mode dispatch, the eviction touch, and the statistics counters — and silently turns any oversized entry into a permanent miss. diskcache 5.6.3 has no `get_many`; nothing in its public API beats a `get()` loop. |
| Cache `to_python()` | 43 ms → 0.0002 ms on repeat calls | The gain is real, but caching hands every caller the same mutable list — precisely what `to_numpy()`'s `.copy()` exists to prevent. The docstring claimed a caching that was never implemented; **the docstring was fixed, not the code.** |

### What the three together are worth

Not the prototypes — the shipped code, measured against the same tree with the
library change stashed:

| Call | 0.5.x behaviour | 0.6.0 | speedup |
| --- | --- | --- | --- |
| sync, 4096 texts, all hits | 72.3 ms | 64.2 ms | 1.13× |
| sync, 4096 texts, all misses | 536.2 ms | 346.7 ms | 1.55× |
| async, 32 concurrent × 512 texts, all hits | 938.1 ms | 313.7 ms | **2.99×** |

### The new executor default

`executor_max_workers` now defaults to `1` instead of `min(32, cpu_count + 4)`.
Everything the pool runs against a local `diskcache` is GIL-bound — sqlite, and
tiktoken — so extra workers were adding contention rather than parallelism.

**The knob stays.** One worker is right only because the cache is local and
fast, and nothing stops a caller passing a cache backed by a network
filesystem, Redis, or S3. Simulating exactly that — 16 concurrent 64-text
all-hit calls against a cache whose reads block — the answer inverts, hard:

| Cache read latency | Best worker count | vs one worker |
| --- | --- | --- |
| 0 ms (local diskcache) | **1** | — (14 workers is 4.4× *slower*) |
| 0.5 ms | 8 | **6.9× faster** |
| 2 ms | 14 | **7.6× faster** |
| 10 ms | 14 | **8.0× faster** |

Half a millisecond per read is enough to flip it, which is not an exotic
threshold. Hardcoding one worker would hand anyone with a remote cache an 8×
regression and no way out. So: default to 1, and **raise it to roughly your
concurrency if your cache blocks on I/O**. `None` restores the stdlib default.

---

## 7. Were these the right fixes?

Both fixes were found by measurement before they were understood, which is a
good way to ship something that surprises you later. Research produced four
explanations; each was turned into a prediction and tested. **Three failed.**

**"Offloading cache reads to a thread is pointless, since the work is GIL-bound
either way."** Half right, and the half that is wrong is the important one.
Doing the reads inline on the loop is indeed no slower — 0.151 s against
0.155 s for one executor worker, a wash. But it blocks the event loop **7.7×
longer**:

| Variant | wall | max loop lag |
| --- | --- | --- |
| executor, 1 worker | 0.155 s | 19 ms |
| executor, 14 workers | 0.550 s | 23 ms |
| inline on the event loop | 0.151 s | **147 ms** |

The executor is not buying throughput. It is buying the loop the chance to run
somebody else's request while sqlite holds the GIL released. Keep it.

**"One worker means tiktoken blocks cache reads behind it in the queue."** Not
reproduced. Adding a 4096-text uncached call into 16 concurrent hit-path
callers moved the hit-path maximum from 46.4 ms to 47.7 ms — 1.03×. Splitting
cache I/O and tokenisation across two dedicated executors, the recommended fix,
recovered 1.4 ms of that. At realistic load the single worker's queue is
already the dominant latency and one more job is lost in it. **Rejected: not
worth a second thread pool.**

**"The transact() speedup decays at large batches because of sqlite page-cache
spilling."** Refuted twice over. Varying the page cache from 4 MB to 512 MB
against a fresh database changed nothing, and the decay itself turned out to be
a harness artifact — see the correction above.

**"A transacted batch rolls back entirely if one write fails."** Confirmed, and
it is the one real cost of the change. Numbers in the section above.

Two things from the research did survive, and both are worth knowing:
`aiosqlite` gives each connection exactly one dedicated worker thread, and
SQLAlchemy's async sqlite support inherits that — so a single worker is the
settled answer to this problem elsewhere, not a local hack. And reading
diskcache's source explains why `transact()` composes: `_transact` keys an open
transaction to `threading.get_ident()`, so the nested `set()` calls inside the
block detect the outer `BEGIN IMMEDIATE` and skip issuing their own.

---

## 8. Where the ceiling is, and what is left

For planning the next release, the bottleneck depends on which regime you are
in — and only one of the three has anything left in it.

**Miss-dominated** (cold start, corpus ingest): the provider is the bottleneck
and the library gets out of the way. Threads scale 25.7× to 32, and the local
cost — 31 µs/text of tiktoken — is around 1% of a real 200-500 ms round trip.
Nothing here is worth optimising.

**Async under concurrency**: throughput floors at ~14k texts/sec however many
callers you add, which is *lower* than one sync thread. Async buys you
concurrency against the provider, not throughput. Do not reach for it to go
faster on cache hits.

**Hit-dominated** (the steady state of any mature deployment): this is the real
ceiling, and it is **50,571 texts/sec per process**, cache hit through to a
usable numpy array. Threads make it worse; only processes scale it.

### What that 50k is made of

Per text, hit through `to_numpy()` — 19.8 µs total:

| Step | µs | share |
| --- | --- | --- |
| `base64.b64decode` in `validate_cached_embedding` | 6.7 | 34% |
| `base64.b64decode` again in `ModelResponse._ndarray` | 6.0 | 30% |
| `cache.get` | 3.7 | 19% |
| key generation | 1.6 | 8% |
| `np.isfinite(...).all()` | 0.9 | 4% |
| everything else | ~1.0 | 5% |

**Base64 decoding is 65% of the cache-hit path, and half of it is redundant.**
The bytes are decoded once to validate the entry, thrown away, and decoded
again to build the array.

### The next win, sized

Carrying the decoded buffer from validation through to `ModelResponse` instead
of decoding twice:

| | texts/sec |
| --- | --- |
| today | 50,571 |
| decoded once | 72,651 |

**1.44×**, and it is the largest single item left. It is not a free change:
`ModelResponse.output` is a public, frozen `list[str]` of base64, so the buffer
has to travel beside it rather than replace it. That shape makes it a 0.7.0
change, not a patch.

### What will not work — already measured, so nobody retries it

| Idea | Result |
| --- | --- |
| Decode the whole batch in one `b64decode` instead of per entry | **0.99×.** The cost is proportional to bytes, not to call count, so there is no loop overhead to remove. Verified equivalent on all 18 rejection cases first, then found to be pointless. |
| Streamline `validate_cached_embedding`'s internals | 0.99–1.02× |
| More executor threads, or a second executor | negative, and 1.03× respectively |
| `FanoutCache` | no read gain, and it drops writes silently |
| Raw sqlite bulk reads | 1.6–2.3×, at the cost of private API and silent misses on oversized entries |

### The structural answer

Even with the double decode gone, one process tops out near 73k texts/sec and
no amount of threading moves it. **The unit of scale is the process** — a
worker per performance core, each with its own model, which is measured at
3.2× on four. `_reset_after_fork` already exists to make that safe. Anything
past that is a question about how fast Python can move 6 KB of float32 per
text, not a question about this library.

---

## Reproducing

The benchmark scripts live in `benchmarks/` in the repository. They are not
shipped with the package and are not part of the test suite. From the repo
root:

```bash
python benchmarks/bench_cache.py   # disk cache in isolation
python benchmarks/bench_sync.py    # sync model across threads
python benchmarks/bench_async.py   # event loop and thread pool
python benchmarks/bench_opts.py    # optimisation candidates
```

Each writes a self-describing JSON payload to `benchmarks/results/`, and each
takes `--quick` for a fast smoke run. Alongside them, the `diag_*.py` scripts
hold the follow-up experiments: the read-collapse elimination table in section
4 (`diag_read_collapse.py`, `diag_read_collapse2.py`, `diag_fanout.py`,
`diag_proc_vs_thread.py`), the executor trade-off sweep in section 3
(`diag_executor_tradeoff.py`), the partly-warm sweep in section 5
(`diag_partial_hit.py`), the mechanism checks in section 7
(`diag_mechanisms.py`), and the ceiling analysis in section 8
(`diag_validation_ceiling.py`).
