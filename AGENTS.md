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
make fmt                                      # isort, black, ruff, gitleaks
pyright                                       # must stay at zero errors
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
- Style: 120 columns, black + isort (`profile = "black"`), linted by ruff.
  Every setting lives in `pyproject.toml`; there is no flake8 any more. A
  `# noqa` needs a comment saying why.

## Before committing

- Version lives in **two** places — `openai_embeddings_model/VERSION` and
  `pyproject.toml`. They must match.
- Record user-visible changes in `CHANGELOG.md`. The docs site renders it
  directly, so there is nowhere else to put them.

## Traps

Behaviour users need is in the docs' *Good to Know*. These are the ones that
bite while editing:

- **Changing the cache key layout costs users money.** It invalidates every
  entry and re-embeds their corpus. Bump `CACHE_KEY_VERSION`, take a minor
  version, and say so loudly in the changelog — never in a patch.
- **Blocking cache I/O lives on the sync class only.** The async model uses
  `_cache_get_many` / `_cache_set_many` via its executor. Keep it that way;
  anything CPU-bound or blocking added to the async path needs
  `run_in_executor`.
- **Anything that does not survive a fork** belongs in `_reset_after_fork`.
- **`validate_cached_embedding` can only check dimensionality** when
  `model_settings.dimensions` is set. Not fixable — nothing in an
  `extra_body`-only request states the expected size.
