# OpenAI Embeddings Model

A high-performance Python library for generating embeddings using OpenAI's API and other OpenAI-compatible providers, with intelligent caching, batch processing, and similarity search.

## Installation

```bash
pip install openai-embeddings-model
```

---

## Quick Start

=== "Sync"

    ```python
    import openai
    from openai_embeddings_model import OpenAIEmbeddingsModel, ModelSettings

    client = openai.OpenAI(api_key="your-api-key")
    model = OpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)

    response = model.get_embeddings(
        input=["Hello, world!", "How are you?"],
        model_settings=ModelSettings(dimensions=512)
    )

    embeddings = response.to_numpy()  # shape: (2, 512)
    print(f"Shape: {embeddings.shape}, Tokens: {response.usage.total_tokens}")
    ```

=== "Async"

    ```python
    import asyncio, openai
    from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

    async def main():
        client = openai.AsyncOpenAI(api_key="your-api-key")
        async with AsyncOpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=client
        ) as model:
            response = await model.get_embeddings(
                input=["Hello, world!", "How are you?"],
                model_settings=ModelSettings(dimensions=512)
            )
            print(response.to_numpy().shape)  # (2, 512)

    asyncio.run(main())
    ```

    !!! tip "Release the thread pool"

        `AsyncOpenAIEmbeddingsModel` owns a `ThreadPoolExecutor` for cache I/O.
        Use it as an async context manager, or call `await model.aclose()`, so
        the worker threads are released deterministically rather than at
        garbage-collection time.

---

## Similarity Search

Rank documents against a query — results sorted by relevance score descending.

=== "Sync"

    ```python
    query = "What is the capital of France?"
    documents = [
        "The capital of Germany is Berlin.",
        "The capital of France is Paris.",
        "The capital of Italy is Rome.",
    ]

    response = model.get_similarity(
        query, documents, model_settings=ModelSettings(dimensions=512)
    )

    for result in response.results:
        print(f"[{result.index}] {result.relevance_score:.4f}  {documents[result.index]}")
    ```

=== "Async"

    ```python
    response = await model.get_similarity(
        query, documents, model_settings=ModelSettings(dimensions=512)
    )
    ```

!!! tip
    `response.results` is always sorted by `relevance_score` from highest to lowest.

---

## Supported Providers

| Provider         | Example model                                      |
|------------------|----------------------------------------------------|
| **OpenAI**       | `text-embedding-3-small`, `text-embedding-3-large` |
| **Azure OpenAI** | `text-embedding-3-small` via `AzureOpenAI` client  |
| **Gemini**       | `text-embedding-004`                               |
| **Voyage AI**    | `voyage-3`, `voyage-3-lite`                        |
| **Self-hosted**  | `nomic-embed-text` via Ollama / LocalAI            |

=== "Azure OpenAI"

    ```python
    from openai import AzureOpenAI
    from openai_embeddings_model import OpenAIEmbeddingsModel

    client = AzureOpenAI(
        api_key="your-azure-key",
        api_version="2023-05-15",
        azure_endpoint="https://your-resource.openai.azure.com/"
    )
    model = OpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)
    ```

=== "Gemini"

    ```python
    client = openai.OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="your-gemini-key"
    )
    model = OpenAIEmbeddingsModel(model="text-embedding-004", openai_client=client)
    ```

=== "Voyage AI"

    ```python
    client = openai.OpenAI(
        base_url="https://api.voyageai.com/v1",
        api_key="your-voyage-key"
    )
    model = OpenAIEmbeddingsModel(model="voyage-3-lite", openai_client=client)
    ```

=== "Self-hosted (Ollama)"

    ```python
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model = OpenAIEmbeddingsModel(model="nomic-embed-text", openai_client=client)
    ```

---

## Caching

Disk-based caching prevents redundant API calls. Cache hits are free and tracked in `usage.cache_hits`.

```python
import diskcache
from openai_embeddings_model import get_default_cache

# Default location: ./.cache/embeddings.cache, relative to the working directory
cache = get_default_cache()

# Or specify a path
cache = diskcache.Cache('/path/to/cache')

model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client,
    cache=cache
)
```

Entries are validated on read. Anything that does not decode into an embedding
of the expected shape is discarded and re-fetched, so a cache directory shared
with another tool cannot feed you a malformed vector.

The cache key covers the model name, `dimensions`, the text, the client's
`base_url`, and `extra_body`. Two clients pointing at different providers can
therefore share one cache directory safely, and requests differing only in a
provider parameter — Voyage's `output_dimension`, a task type — no longer
collide.

Repeated texts within a single call are embedded once and the vector shared,
so passing the same string many times costs one slot, not many.

!!! warning "0.6.0 invalidates existing caches"

    The key layout changed, so entries written by 0.5.x are ignored rather
    than misread. The first run after upgrading re-embeds everything. Old
    entries are not deleted — clear the directory yourself once you no longer
    need to roll back.

---

## Good to Know

**Token counts for non-OpenAI models are approximate.** `tiktoken` has no
encoding for Voyage, Gemini, or a local model, so the count falls back to
gpt-4o's tokenizer and the library warns once. That makes truncation points
approximate — pass `encoding=` to make them exact.

**Truncation is silent by default.** `token_limit_policy` defaults to
`"truncate"`, which drops input to fit the token limit. Check
`usage.truncated_texts`, or set the policy to `"raise"`.

**A `ModelResponse` cannot be pickled or deep-copied** once `to_numpy()`,
`to_python()`, or `as_similarity_response()` has run — the cached decode is a
`memoryview`. Send `to_python()` output across a process boundary instead.

**Forking is handled.** A model built before `os.fork()` — the gunicorn
`preload_app` shape — rebuilds its thread pool and cache connection in the
child automatically.

---

## API Reference

### Classes

| Class                        | Description                                                   |
|------------------------------|---------------------------------------------------------------|
| `OpenAIEmbeddingsModel`      | Synchronous model                                             |
| `AsyncOpenAIEmbeddingsModel` | Async model with dedicated `ThreadPoolExecutor` for cache I/O |

### Constructor Parameters

| Parameter                   | Type                           | Default      | Description                                        |
|-----------------------------|--------------------------------|--------------|----------------------------------------------------|
| `model`                     | `str \| EmbeddingModel`        | —            | Model name                                         |
| `openai_client`             | `OpenAI \| AsyncOpenAI \| ...` | —            | OpenAI-compatible client                           |
| `cache`                     | `diskcache.Cache \| None`      | `None`       | Embedding cache                                    |
| `encoding`                  | `tiktoken.Encoding \| None`    | `None`       | Tokenizer override; wins over auto-detection       |
| `max_batch_size`            | `int`                          | `2048`       | Max texts per API call                             |
| `max_input_tokens`          | `int`                          | `8191`       | Max tokens per text                                |
| `max_tokens_a_request`      | `int`                          | `300000`     | Max tokens per API call                            |
| `token_limit_policy`        | `str`                          | `"truncate"` | `"raise"`, `"warn"`, `"ignore"`, `"truncate"`      |
| `token_limit_usage_percent` | `float`                        | `85`         | Share of `max_input_tokens` to use; must be in (0, 100] |
| `dimensions_parameter`      | `str \| None`                  | `None`       | `"dimensions"` or `"output_dimension"`; auto-detected |
| `max_retries`               | `int`                          | `2`          | Retries for rate limits and transient failures     |
| `retry_base_delay`          | `float`                        | `1.0`        | Seconds before the first retry, doubling each time |
| `executor_max_workers`      | `int \| None`                  | `None`       | Async only — thread pool size                      |
| `max_concurrent_batches`    | `int`                          | `5`          | Async only — batches in flight at once. A batch keeps its slot while backing off, so a rate-limit storm can fill them all |

### Methods

| Method                                                            | Returns                    | Description                       |
|-------------------------------------------------------------------|----------------------------|-----------------------------------|
| `get_embeddings(input, model_settings)`                           | `ModelResponse`            | Embed one or more texts           |
| `get_embeddings_generator(input, model_settings, chunk_size=100)` | `Generator[ModelResponse]` | Stream results for large datasets |
| `get_similarity(query, documents, model_settings)`                | `SimilarityResponse`       | Rank documents by query relevance |
| `aclose()`                                                        | `None`                     | Async model only — release the cache-I/O thread pool |

### ModelSettings

| Parameter    | Type            | Default | Description                                        |
|--------------|-----------------|---------|----------------------------------------------------|
| `dimensions` | `int \| None`   | `None`  | Custom output dimensions                           |
| `timeout`    | `float \| None` | `None`  | Request timeout (seconds)                          |
| `extra_body` | `dict \| None`  | `None`  | Provider-specific parameters merged into the request |

### Response Types

**`ModelResponse`**

| Attribute / Method   | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `to_numpy()`         | `NDArray[np.float32]` — shape `(n, dims)`, a writable copy    |
| `to_python()`        | `List[List[float]]`                                          |
| `usage.input_tokens` | Tokens from input texts                                      |
| `usage.total_tokens` | Total tokens billed                                          |
| `usage.cache_hits`   | Number of cache hits                                         |
| `usage.truncated_texts` | Texts shortened to fit the token limit — non-zero means input was dropped |

**`SimilarityResponse`**

| Attribute | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `results` | `list[SimilarityResult]` sorted by `relevance_score` descending |
| `usage`   | Same usage stats as `ModelResponse`                             |

**`SimilarityResult`**

| Attribute         | Description             |
|-------------------|-------------------------|
| `index`           | Original document index |
| `relevance_score` | Cosine similarity score |

---

## Requirements

- Python 3.11+
- OpenAI API key (or compatible provider)

## License

MIT — [Allen Chou](mailto:f1470891079@gmail.com)
