# OpenAI Embeddings Model

A high-performance Python library for generating embeddings using OpenAI's API and other OpenAI-compatible providers, with intelligent caching, batch processing, and similarity search.

## Installation

```bash
pip install openai-embeddings-model
```

## Quick Start

```python
import openai
from openai_embeddings_model import OpenAIEmbeddingsModel, ModelSettings

client = openai.OpenAI(api_key="your-api-key")
model = OpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)

response = model.get_embeddings(
    input=["Hello, world!", "How are you?"],
    model_settings=ModelSettings(dimensions=512)
)

embeddings = response.to_numpy()  # NumPy array
print(f"Shape: {embeddings.shape}, Tokens: {response.usage.total_tokens}")
```

### Async

```python
import asyncio, openai
from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

async def main():
    client = openai.AsyncOpenAI(api_key="your-api-key")
    async with AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small", openai_client=client
    ) as model:
        response = await model.get_embeddings(
            input=["Hello, world!"],
            model_settings=ModelSettings(dimensions=512)
        )
        print(response.to_numpy().shape)

asyncio.run(main())
```

`AsyncOpenAIEmbeddingsModel` owns a `ThreadPoolExecutor` for cache I/O. Use it as
an async context manager, or call `await model.aclose()`, so the worker threads
are released deterministically instead of at garbage-collection time.

## Similarity Search

Find the most relevant documents for a query — results are sorted by relevance score:

```python
query = "What is the capital of France?"
documents = [
    "The capital of Germany is Berlin.",
    "The capital of France is Paris.",
    "The capital of Italy is Rome.",
]

response = model.get_similarity(query, documents, model_settings=ModelSettings(dimensions=512))

for result in response.results:
    print(f"[{result.index}] score={result.relevance_score:.4f}  {documents[result.index]}")
```

Also available as `await model.get_similarity(...)` on `AsyncOpenAIEmbeddingsModel`.

## Supported Providers

| Provider         | Example model                                      |
|------------------|----------------------------------------------------|
| **OpenAI**       | `text-embedding-3-small`, `text-embedding-3-large` |
| **Azure OpenAI** | `text-embedding-3-small` via AzureOpenAI client    |
| **Gemini**       | `text-embedding-004`                               |
| **Voyage AI**    | `voyage-3`, `voyage-3-lite`                        |
| **Self-hosted**  | `nomic-embed-text` via Ollama / LocalAI            |

### Voyage AI

```python
import openai
from openai_embeddings_model import OpenAIEmbeddingsModel, ModelSettings

client = openai.OpenAI(
    base_url="https://api.voyageai.com/v1",
    api_key="your-voyage-api-key"
)
model = OpenAIEmbeddingsModel(model="voyage-3-lite", openai_client=client)
response = model.get_embeddings(input=["Hello"], model_settings=ModelSettings(dimensions=512))
```

## Caching

```python
import diskcache
from openai_embeddings_model import get_default_cache

# Default cache
cache = get_default_cache()

# Custom location
cache = diskcache.Cache('/path/to/cache')

model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client,
    cache=cache
)
```

Cache hits are tracked in `response.usage.cache_hits` and never re-billed. The
cache key covers the model name, `dimensions`, the text, the client's
`base_url`, and `extra_body`, so different providers can share one directory
safely. Entries that do not decode into an embedding of the expected shape are
discarded and re-fetched.

> **0.6.0 invalidates existing caches.** The key layout changed, so entries
> written by 0.5.x are ignored rather than misread, and the first run after
> upgrading re-embeds everything. Old entries are not deleted — clear the
> directory yourself once you no longer need to roll back.

## API Reference

### Classes

| Class                        | Description                                                |
|------------------------------|------------------------------------------------------------|
| `OpenAIEmbeddingsModel`      | Synchronous model                                          |
| `AsyncOpenAIEmbeddingsModel` | Async model (dedicated `ThreadPoolExecutor` for cache I/O) |

### Methods

| Method                                                            | Returns                    |
|-------------------------------------------------------------------|----------------------------|
| `get_embeddings(input, model_settings)`                           | `ModelResponse`            |
| `get_embeddings_generator(input, model_settings, chunk_size=100)` | `Generator[ModelResponse]` |
| `get_similarity(query, documents, model_settings)`                | `SimilarityResponse`       |
| `aclose()` — async model only, releases the cache-I/O thread pool | `None`                     |

### ModelSettings

| Parameter    | Type            | Default | Description                                          |
|--------------|-----------------|---------|------------------------------------------------------|
| `dimensions` | `int \| None`   | `None`  | Custom output dimensions                             |
| `timeout`    | `float \| None` | `None`  | Request timeout (seconds)                            |
| `extra_body` | `dict \| None`  | `None`  | Provider-specific parameters merged into the request |

### Constructor Parameters

Beyond `model` and `openai_client`: `cache`, `encoding`, `max_batch_size`,
`max_input_tokens`, `max_tokens_a_request`, `token_limit_policy`,
`token_limit_usage_percent`, `dimensions_parameter`, `max_retries`,
`retry_base_delay`, and — async only — `executor_max_workers` and
`max_concurrent_batches`. See the
[documentation](https://allen2c.github.io/openai-embeddings-model/) for types
and defaults.

### Responses

**`ModelResponse`**

- `to_numpy()` → `NDArray[np.float32]` — a writable copy
- `to_python()` → `List[List[float]]`
- `usage.input_tokens`, `usage.total_tokens`, `usage.cache_hits`
- `usage.truncated_texts` — texts shortened to fit the token limit; non-zero
  means input was dropped before embedding

`ModelResponse` is immutable.

**`SimilarityResponse`**

- `results: list[SimilarityResult]` — sorted by `relevance_score` descending
- `usage` — same as `ModelResponse`

**`SimilarityResult`**

- `index: int` — original document index
- `relevance_score: float`

## Requirements

- Python 3.11+
- OpenAI API key (or compatible provider)

## License

MIT — Allen Chou &lt;<f1470891079@gmail.com>&gt;
