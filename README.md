# OpenAI Embeddings Model

A high-performance Python library for generating embeddings using OpenAI's API and other OpenAI-compatible providers with intelligent caching and batch processing.

## Features

- **🚀 High Performance**: Optimized batch processing
- **🔄 Smart Caching**: Intelligent disk-based caching
- **⚡ Async Support**: Full async/await support
- **🧠 Memory Efficient**: Lazy decoding with zero-copy views
- **📊 Usage Tracking**: Token usage and cache statistics
- **🛡️ Thread Safe**: Concurrent processing support
- **🌐 Multi-Provider**: OpenAI, Gemini, and other compatible APIs
- **📈 Scalable**: Generator support for large datasets

## Installation

```bash
pip install openai-embeddings-model
```

## Quick Start

### Basic Usage

```python
import openai
from openai_embeddings_model import OpenAIEmbeddingsModel, ModelSettings

# Initialize
client = openai.OpenAI(api_key="your-api-key")
model = OpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)

# Generate embeddings
response = model.get_embeddings(
    input="Hello, world!",
    model_settings=ModelSettings(dimensions=512)
)

# Access results
embeddings = response.to_numpy()  # NumPy array
embeddings_list = response.to_python()  # Python lists
print(f"Tokens used: {response.usage.input_tokens}")
```

### Async Usage

```python
import asyncio
import openai
from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

async def main():
    client = openai.AsyncOpenAI(api_key="your-api-key")
    model = AsyncOpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)

    response = await model.get_embeddings(
        input=["Hello, world!", "How are you?"],
        model_settings=ModelSettings(dimensions=512)
    )

    embeddings = response.to_numpy()
    print(f"Shape: {embeddings.shape}")

asyncio.run(main())
```

## Supported Providers

### OpenAI

```python
client = openai.OpenAI(api_key="your-api-key")
model = OpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)
```

**Available models:**

- `text-embedding-3-small` (up to 1536 dimensions)
- `text-embedding-3-large` (up to 3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions, fixed)

### Gemini

```python
client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="your-gemini-api-key"
)
model = OpenAIEmbeddingsModel(model="text-embedding-004", openai_client=client)
```

### Azure OpenAI

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-azure-api-key",
    api_version="2023-05-15",
    azure_endpoint="https://your-resource.openai.azure.com/"
)
model = OpenAIEmbeddingsModel(model="text-embedding-3-small", openai_client=client)
```

### Self-Hosted (Ollama, LocalAI, etc.)

```python
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real API key
)
model = OpenAIEmbeddingsModel(model="nomic-embed-text", openai_client=client)
```

## Advanced Features

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3", ...]

# All at once
response = model.get_embeddings(input=texts, model_settings=ModelSettings(dimensions=512))

# Or use generator for large datasets
for chunk in model.get_embeddings_generator(input=texts, chunk_size=100):
    process_chunk(chunk.to_numpy())
```

### Custom Caching

```python
import diskcache

# Custom cache location
cache = diskcache.Cache('/path/to/cache')
model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client,
    cache=cache
)

# Or use default cache
from openai_embeddings_model import get_default_cache
cache = get_default_cache()
```

### Model Configuration

```python
settings = ModelSettings(
    dimensions=1024,    # Custom dimensions (if supported)
    timeout=30.0       # Request timeout in seconds
)

response = model.get_embeddings(input="Your text", model_settings=settings)
```

## API Reference

### Models

- **OpenAIEmbeddingsModel** - Synchronous embedding generation
- **AsyncOpenAIEmbeddingsModel** - Asynchronous embedding generation

### Methods

- `get_embeddings(input, model_settings)` → `ModelResponse`
- `get_embeddings_generator(input, model_settings, chunk_size=100)` → `Generator[ModelResponse]`

### Configuration

- **ModelSettings**
    - `dimensions: int | None = None` - Custom embedding dimensions
    - `timeout: float | None = None` - Request timeout

### Response

- **ModelResponse**
    - `to_numpy()` → `NDArray[np.float32]` - NumPy array
    - `to_python()` → `List[List[float]]` - Python lists
    - `usage.input_tokens` - Input tokens used
    - `usage.total_tokens` - Total tokens used
    - `usage.cache_hits` - Cache hits

## Error Handling

```python
try:
    response = model.get_embeddings(input="Your text", model_settings=settings)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

## Performance Tips

1. **Enable caching** - Avoid redundant API calls
2. **Use batch processing** - Process multiple texts together
3. **Choose appropriate dimensions** - Smaller = faster + cheaper
4. **Use async for I/O-bound work** - Better concurrency
5. **Use generators for large datasets** - Memory efficient

## Requirements

- Python 3.11+
- OpenAI API key (or compatible provider)

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests or open issues.

## Author

Allen Chou - <f1470891079@gmail.com>
