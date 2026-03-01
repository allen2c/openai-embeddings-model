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
        model = AsyncOpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)

        response = await model.get_embeddings(
            input=["Hello, world!", "How are you?"],
            model_settings=ModelSettings(dimensions=512)
        )
        print(response.to_numpy().shape)  # (2, 512)

    asyncio.run(main())
    ```

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

# Use default XDG cache directory
cache = get_default_cache()

# Or specify a path
cache = diskcache.Cache('/path/to/cache')

model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client,
    cache=cache
)
```

---

## API Reference

### Classes

| Class                        | Description                                                   |
|------------------------------|---------------------------------------------------------------|
| `OpenAIEmbeddingsModel`      | Synchronous model                                             |
| `AsyncOpenAIEmbeddingsModel` | Async model with dedicated `ThreadPoolExecutor` for cache I/O |

### Constructor Parameters

| Parameter              | Type                           | Default      | Description                                   |
|------------------------|--------------------------------|--------------|-----------------------------------------------|
| `model`                | `str \| EmbeddingModel`        | —            | Model name                                    |
| `openai_client`        | `OpenAI \| AsyncOpenAI \| ...` | —            | OpenAI-compatible client                      |
| `cache`                | `diskcache.Cache \| None`      | `None`       | Embedding cache                               |
| `max_batch_size`       | `int`                          | `2048`       | Max texts per API call                        |
| `token_limit_policy`   | `str`                          | `"truncate"` | `"raise"`, `"warn"`, `"ignore"`, `"truncate"` |
| `executor_max_workers` | `int \| None`                  | `None`       | Async only — thread pool size                 |

### Methods

| Method                                                            | Returns                    | Description                       |
|-------------------------------------------------------------------|----------------------------|-----------------------------------|
| `get_embeddings(input, model_settings)`                           | `ModelResponse`            | Embed one or more texts           |
| `get_embeddings_generator(input, model_settings, chunk_size=100)` | `Generator[ModelResponse]` | Stream results for large datasets |
| `get_similarity(query, documents, model_settings)`                | `SimilarityResponse`       | Rank documents by query relevance |

### ModelSettings

| Parameter    | Type            | Default | Description               |
|--------------|-----------------|---------|---------------------------|
| `dimensions` | `int \| None`   | `None`  | Custom output dimensions  |
| `timeout`    | `float \| None` | `None`  | Request timeout (seconds) |

### Response Types

**`ModelResponse`**

| Attribute / Method   | Description                               |
|----------------------|-------------------------------------------|
| `to_numpy()`         | `NDArray[np.float32]` — shape `(n, dims)` |
| `to_python()`        | `List[List[float]]`                       |
| `usage.input_tokens` | Tokens from input texts                   |
| `usage.total_tokens` | Total tokens billed                       |
| `usage.cache_hits`   | Number of cache hits                      |

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
