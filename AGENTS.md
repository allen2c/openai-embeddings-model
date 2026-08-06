# AGENTS.md

OpenAI-compatible embeddings client with disk caching, batching, and similarity
search. Sync and async variants.

**The docs are the reference, not this file.** Read them first:

| Read | For |
|---|---|
| [`docs/index.md`](docs/index.md) | Installation, usage, providers, caching, the full API, and the *Good to Know* behaviours users rely on. |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Where time goes, how it behaves under concurrency, what was already tried and rejected, and where the remaining ceiling is. |
| [`CHANGELOG.md`](CHANGELOG.md) | Every user-visible change. There is nowhere else to record one. |

## Layout

| Path | What |
|---|---|
| `openai_embeddings_model/__init__.py` | Everything. Both model classes, the response types, and the module-level helpers. |
| `openai_embeddings_model/normalize.py` | Vector normalisation. |
| `openai_embeddings_model/embedding_model.py` | `EmbeddingModel` literal type. |
| `tests/test_regressions*.py` | Offline suites, one per release. No API key, no network. |
| Other `tests/*` | Hit the **real** API and need `OPENAI_API_KEY` (some need `GEMINI_API_KEY`). |
| `benchmarks/` | Offline performance experiments. `bench_*.py` measure, `diag_*.py` chase a specific result down. |

## Working here

```bash
poetry install --all-extras --all-groups
python -m pytest tests/test_regressions*.py   # offline, always safe to run
make fmt                                      # isort, black, ruff, gitleaks
pyright                                       # must stay at zero errors
```

- Add offline tests to the current release's `tests/test_regressions*.py`, and
  **confirm each new test fails against the previous release** — stash the
  library change and re-run. A regression test that passes on the buggy version
  is testing nothing. Build a real client against `base_url="http://localhost:1"`
  so isinstance checks pass, then replace `client.embeddings.create` with a
  fake. Never add a test that makes a real API call.
- The legacy tests share the repo's real `./.cache`. Use `tmp_path` for anything
  new.
- Style: 120 columns, black + isort (`profile = "black"`), ruff. Everything is
  configured in `pyproject.toml`. A `# noqa` needs a comment saying why.
- Version lives in **two** places — `openai_embeddings_model/VERSION` and
  `pyproject.toml`. They must match.

## Traps

Four things that bite while editing. Everything else is in the docs.

- **Changing the cache key layout costs users money.** It invalidates every
  entry and re-embeds their corpus. Bump `CACHE_KEY_VERSION`, take a minor
  version, and say so loudly in the changelog — never in a patch.
- **More threads make this library slower.** Everything except waiting on the
  provider is GIL-bound, so added workers buy contention, not parallelism. That
  is why `executor_max_workers` defaults to 1. Measure before adding a pool or
  raising a worker count; `docs/benchmarks.md` has the numbers and the list of
  optimisations already tried and rejected.
- **Blocking cache I/O lives on the sync class only.** The async model goes
  through `_cache_get_many` / `_cache_set_many` in its executor. Anything
  CPU-bound or blocking added to the async path needs `run_in_executor`.
- **Anything that does not survive a fork** belongs in `_reset_after_fork`.
