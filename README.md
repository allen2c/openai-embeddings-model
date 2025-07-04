# OpenAI Embeddings Model

A high-performance, thread-safe Python library for generating embeddings using OpenAI's embedding models and other OpenAI-compatible APIs with intelligent caching and batch processing.

## Features

- **🚀 High Performance**: Optimized batch processing with configurable batch sizes
- **🔄 Smart Caching**: Intelligent disk-based caching to avoid redundant API calls
- **⚡ Async Support**: Both synchronous and asynchronous implementations
- **🧠 Memory Efficient**: Lazy decoding with zero-copy memory views
- **📊 Usage Tracking**: Comprehensive token usage and cache hit statistics
- **🛡️ Thread Safe**: Concurrent processing with proper error handling
- **📈 Scalable**: Generator support for processing large datasets
- **🎯 Model Validation**: Automatic validation of model capabilities and constraints

## Supported Models

### OpenAI Official Models

- `text-embedding-3-small` (up to 1536 dimensions)
- `text-embedding-3-large` (up to 3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions, no custom dimensions)

### OpenAI-Compatible APIs

- Any embedding model accessible through OpenAI-compatible endpoints
- Self-hosted solutions like Ollama, LocalAI, etc.
- Custom embedding services with OpenAI-compatible interfaces

## Installation

```bash
pip install openai-embeddings-model
```

## Requirements

- Python 3.11+
- OpenAI API key

## Quick Start

### Synchronous Usage

```python
import openai
from openai_embeddings_model import OpenAIEmbeddingsModel, ModelSettings

# Initialize OpenAI client
client = openai.OpenAI(api_key="your-api-key")

# Create embedding model
model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client
)

# Generate embeddings
response = model.get_embeddings(
    input="Hello, world!",
    model_settings=ModelSettings(dimensions=512)
)

# Access embeddings
embeddings = response.to_numpy()  # NumPy array
embeddings_list = response.to_python()  # Python lists

# Check usage statistics
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Cache hits: {response.usage.cache_hits}")
```

### Asynchronous Usage

```python
import asyncio
import openai
from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

async def main():
    # Initialize async OpenAI client
    client = openai.AsyncOpenAI(api_key="your-api-key")

    # Create async embedding model
    model = AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=client
    )

    # Generate embeddings
    response = await model.get_embeddings(
        input=["Hello, world!", "How are you?"],
        model_settings=ModelSettings(dimensions=512)
    )

    embeddings = response.to_numpy()
    print(f"Generated embeddings: {embeddings.shape}")

asyncio.run(main())
```

## Advanced Usage

### Batch Processing

```python
# Process large datasets efficiently
texts = ["Text 1", "Text 2", "Text 3", ...]

# All at once
response = model.get_embeddings(
    input=texts,
    model_settings=ModelSettings(dimensions=512)
)

# Or use generator for memory efficiency
for chunk_response in model.get_embeddings_generator(
    input=texts,
    model_settings=ModelSettings(dimensions=512),
    chunk_size=100
):
    process_chunk(chunk_response.to_numpy())
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
model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client,
    cache=get_default_cache()
)
```

### Model Configuration

```python
# Configure model settings
settings = ModelSettings(
    dimensions=1024,    # Custom dimensions (if supported)
    timeout=30.0       # Request timeout in seconds
)

response = model.get_embeddings(
    input="Your text here",
    model_settings=settings
)
```

### Azure OpenAI Support

```python
from openai import AzureOpenAI

# Azure OpenAI client
client = AzureOpenAI(
    api_key="your-azure-api-key",
    api_version="2023-05-15",
    azure_endpoint="https://your-resource.openai.azure.com/"
)

model = OpenAIEmbeddingsModel(
    model="text-embedding-3-small",
    openai_client=client
)
```

### Self-Hosted and OpenAI-Compatible APIs

```python
import openai

# Ollama (self-hosted)
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real API key
)

model = OpenAIEmbeddingsModel(
    model="nomic-embed-text",  # Or any model available in Ollama
    openai_client=client
)

# Other OpenAI-compatible endpoints
client = openai.OpenAI(
    base_url="https://your-custom-endpoint.com/v1",
    api_key="your-api-key"
)

model = OpenAIEmbeddingsModel(
    model="your-custom-model",
    openai_client=client
)
```

## API Reference

### OpenAIEmbeddingsModel

Main class for synchronous embedding generation.

#### OpenAIEmbeddingsModel Methods

- `get_embeddings(input, model_settings) -> ModelResponse`
    - Generate embeddings for input text(s)
    - **input**: `str` or `List[str]` - Text(s) to embed
    - **model_settings**: `ModelSettings` - Configuration options
    - **Returns**: `ModelResponse` with embeddings and usage stats

- `get_embeddings_generator(input, model_settings, chunk_size=100) -> Generator[ModelResponse, None, None]`
    - Generate embeddings in chunks for large datasets
    - **chunk_size**: `int` - Number of texts per chunk

### AsyncOpenAIEmbeddingsModel

Asynchronous version with the same interface but async methods.

#### AsyncOpenAIEmbeddingsModel Methods

- `async get_embeddings(input, model_settings) -> ModelResponse`
- `async get_embeddings_generator(input, model_settings, chunk_size=100) -> AsyncGenerator[ModelResponse, None]`

### ModelSettings

Configuration for embedding requests.

#### ModelSettings Attributes

- `dimensions: int | None = None` - Custom embedding dimensions
- `timeout: float | None = None` - Request timeout in seconds

### ModelResponse

Response object containing embeddings and metadata.

#### ModelResponse Methods

- `to_numpy() -> NDArray[np.float32]` - Get embeddings as NumPy array
- `to_python() -> List[List[float]]` - Get embeddings as Python lists

#### ModelResponse Attributes

- `usage: Usage` - Token usage statistics
    - `input_tokens: int` - Number of input tokens
    - `total_tokens: int` - Total tokens used
    - `cache_hits: int` - Number of cache hits

## Error Handling

The library provides comprehensive error handling:

```python
try:
    response = model.get_embeddings(
        input="Your text",
        model_settings=ModelSettings(dimensions=512)
    )
except ValueError as e:
    print(f"Invalid input or settings: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

## Performance Tips

1. **Use caching**: Enable caching to avoid redundant API calls
2. **Batch processing**: Process multiple texts at once for better throughput
3. **Custom dimensions**: Use smaller dimensions when possible to reduce costs
4. **Async for I/O**: Use async version for I/O-bound applications
5. **Generators**: Use generators for large datasets to manage memory usage

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Author

Allen Chou - <f1470891079@gmail.com>
