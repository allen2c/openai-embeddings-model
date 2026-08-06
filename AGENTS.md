# AGENTS.md

OpenAI-compatible embeddings client with disk caching, batching, and similarity
search. Sync and async variants.

**Read [`docs/index.md`](docs/index.md) first** — it is the reference for
installation, usage, providers, caching, and the full API. This file only
covers what is not in the docs.

## Layout

| Path | What |
|---|---|
| `openai_embeddings_model/__init__.py` | Everything. Both model classes, the response types, and the module-level helpers. |
| `openai_embeddings_model/normalize.py` | Vector normalisation. |
| `openai_embeddings_model/embedding_model.py` | `EmbeddingModel` literal type. |
| `tests/test_regressions*.py` | Offline suites, one per release. No API key, no network. |
| Other `tests/*` | Hit the **real** API and need `OPENAI_API_KEY` (some need `GEMINI_API_KEY`). |

## Working here

```bash
poetry install --all-extras --all-groups
python -m pytest tests/test_regressions*.py   # offline, always safe to run
make format-all                               # isort + black
```

- Add offline tests to the current release's `tests/test_regressions*.py`.
  Confirm each new test **fails** against the previous release (stash the
  library change and re-run) — a regression test that passes on the buggy
  version is testing nothing. Build a real client against
  `base_url="http://localhost:1"` so isinstance checks pass, then replace
  `client.embeddings.create` with a fake. Never add a test that makes a real
  API call.
- The legacy tests share the repo's real `./.cache` directory. Use `tmp_path`
  for anything new.
- Style: black, isort (`profile = "black"`), flake8 at 88 columns.

## Before committing

- Version lives in **two** places — `openai_embeddings_model/VERSION` and
  `pyproject.toml`. They must match.
- Record user-visible changes in `CHANGELOG.md`. The docs site renders it
  directly, so there is nowhere else to put them.

## Known constraints

- The cache key covers model name, `dimensions`, text, `base_url`, and
  `extra_body`, behind a `CACHE_KEY_VERSION` prefix. Changing its layout
  invalidates every existing cache and makes users re-pay for embeddings, so
  it needs a minor version bump and a prominent changelog note — never a patch
  release.
- Token counts for non-OpenAI models fall back to gpt-4o's tokenizer, so
  truncation points are approximate unless the caller passes `encoding=`.
- `validate_cached_embedding` can only check dimensionality when
  `model_settings.dimensions` is set. Providers configured through
  `extra_body` alone get weaker validation. Not fixable from here — nothing
  in the request states the expected size.
- Models register with `os.register_at_fork` so a forked child rebuilds its
  thread pool and drops the inherited sqlite connection. Anything else added
  to a model that does not survive a fork belongs in `_reset_after_fork`.
- A `ModelResponse` cannot be deep-copied or pickled once `to_numpy()`,
  `to_python()`, or `as_similarity_response()` has run — the cached
  `_decoded_bytes` is a `memoryview`. Pass `to_python()` output across a
  process boundary instead. Long-standing, not specific to any release.
