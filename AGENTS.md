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
| `tests/test_regressions.py` | Offline suite. No API key, no network. |
| Other `tests/*` | Hit the **real** API and need `OPENAI_API_KEY` (some need `GEMINI_API_KEY`). |

## Working here

```bash
poetry install --all-extras --all-groups
python -m pytest tests/test_regressions.py   # offline, always safe to run
make format-all                              # isort + black
```

- Add offline tests to `tests/test_regressions.py`. Build a real client against
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

- The cache key covers model name, `dimensions`, and text — **not** the
  client's `base_url` or `extra_body`. One cache directory per provider.
  Changing the key format invalidates every existing cache, so it needs a
  version bump and a prominent changelog note, never a patch release.
- `MAX_TOKENS_A_REQUEST` is declared but not enforced; batches split by item
  count only.
- `ModelSettings.validate_for_model` currently rejects nothing.
