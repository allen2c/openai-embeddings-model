import asyncio
import base64
import enum
import functools
import hashlib
import logging
import pathlib
import time
import typing

import diskcache
import numpy as np
import openai
import pydantic

from .embedding_model import EmbeddingModel

__all__ = ["ModelSettings", "OpenAIEmbeddingsModel", "AsyncOpenAIEmbeddingsModel"]
__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

# Constants
MAX_BATCH_SIZE = 2048  # OpenAI's batch size limit
MAX_INPUT_TOKENS = 8191  # Maximum tokens per input


@functools.lru_cache(maxsize=MAX_BATCH_SIZE)
def generate_cache_key(
    model: str | None = None, dimensions: int | None = None, text: str | None = None
) -> str:
    """Generate a cache key."""
    if text is None:
        raise ValueError("text is required")
    hash_text = hashlib.sha256(text.encode()).hexdigest()
    return f"{model or 'unknown'}:{dimensions or 'default'}:{hash_text}"


def validate_input(input: str | typing.List[str]) -> typing.List[str]:
    """Validate and normalize input, converting strings to lists.

    Raises ValueError for empty inputs, TypeError for invalid types.
    """
    if isinstance(input, str):
        if not input.strip():
            raise ValueError("Input string cannot be empty")
        return [input]
    elif isinstance(input, list):
        if not input:
            raise ValueError("Input list cannot be empty")
        if not all(isinstance(item, str) for item in input):
            raise TypeError("All input items must be strings")
        if not all(item.strip() for item in input):
            raise ValueError("All input items must be non-empty strings")
        return input
    else:
        raise TypeError(f"Input must be str or List[str], got {type(input)}")


def get_default_cache() -> diskcache.Cache:
    """Get default cache instance."""
    return diskcache.Cache(directory="./.cache/embeddings.cache")


def convert_float_list_to_base64(float_list: typing.List[float]) -> str:
    """Convert a list of python floats to base64-encoded numpy float32 array."""
    array = np.array(float_list, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("utf-8")


class EmbeddingModelType(enum.StrEnum):
    """Supported embedding model types with their constraints."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @property
    def max_dimensions(self) -> int | None:
        """Maximum allowed dimensions for this model."""
        return {
            self.TEXT_EMBEDDING_3_SMALL: 1536,
            self.TEXT_EMBEDDING_3_LARGE: 3072,
            self.TEXT_EMBEDDING_ADA_002: 1536,
        }.get(self)

    @property
    def supports_dimensions(self) -> bool:
        """Whether this model supports custom dimensions."""
        return self in {self.TEXT_EMBEDDING_3_SMALL, self.TEXT_EMBEDDING_3_LARGE}


class ModelSettings(pydantic.BaseModel):
    """Configuration for embedding model requests."""

    dimensions: int | None = None
    timeout: float | None = None

    def validate_for_model(self, model: str | EmbeddingModel) -> None:
        """Validate settings are appropriate for the given model."""
        model_str = str(model)

        # Check if model supports dimensions
        try:
            model_type = EmbeddingModelType(model_str)
            if self.dimensions is not None:
                if not model_type.supports_dimensions:
                    raise ValueError(
                        f"Model {model_str} does not support custom dimensions"
                    )
                max_dims = model_type.max_dimensions
                if max_dims and not (1 <= self.dimensions <= max_dims):
                    raise ValueError(
                        f"Dimensions must be between 1 and {max_dims} for {model_str}, "
                        f"got {self.dimensions}"
                    )
        except ValueError:
            # Unknown model type, skip validation
            logger.debug(
                f"Unknown model type: {model_str}, skipping dimension validation"
            )


class Usage(pydantic.BaseModel):
    """Token usage statistics."""

    input_tokens: int = 0
    total_tokens: int = 0
    cache_hits: int = 0


class ModelResponse(pydantic.BaseModel):
    """Response from embedding model with lazy decoding."""

    output: list[typing.Text]
    usage: Usage

    @functools.cached_property
    def _decoded_bytes(self) -> memoryview:
        """
        Lazily decode *all* embeddings in one pass and expose them
        as a zero-copy memoryview to avoid duplicating data.
        """
        return memoryview(b"".join(base64.b64decode(s) for s in self.output))

    @functools.cached_property
    def _ndarray(self) -> np.ndarray:
        """
        Materialize the NumPy array once and cache it.  Later calls to
        `to_numpy()` or `to_python()` return the cached view.
        """
        if not self.output:  # Handle empty response.
            return np.empty((0, 0), dtype=np.float32)

        # Each embedding has the same dimensionality; derive it from the first.
        dim = len(base64.b64decode(self.output[0])) // 4  # 4 bytes per float32
        arr = np.frombuffer(self._decoded_bytes, dtype=np.float32)
        return arr.reshape(len(self.output), dim)

    def to_numpy(self) -> np.typing.NDArray[np.float32]:
        """Return embeddings as an (n, d) float32 ndarray (cached)."""
        return self._ndarray

    def to_python(self) -> list[list[float]]:
        """Return embeddings as ordinary Python lists (cached)."""
        return self._ndarray.tolist()


class OpenAIEmbeddingsModel:
    """Thread-safe OpenAI embeddings model with caching and batch processing."""

    def __init__(
        self,
        model: str | EmbeddingModel,
        openai_client: openai.OpenAI | openai.AzureOpenAI,
        cache: diskcache.Cache | None = None,
    ) -> None:
        self.model = model
        self._client = openai_client
        self._cache = cache

        # Validate model
        self._model_str = str(model)
        logger.debug(f"Initialized OpenAIEmbeddingsModel with model: {self._model_str}")

    def _batch_api_calls(
        self,
        texts: typing.List[str],
        model_settings: ModelSettings,
    ) -> typing.Tuple[typing.List[str], Usage]:
        """
        Process texts in batches to respect API limits.

        Args:
            texts: List of texts to embed
            model_settings: Model configuration

        Returns:
            Tuple of (List of base64-encoded embeddings, Usage statistics)

        Raises:
            RuntimeError: If API call fails
        """
        embeddings: typing.List[str] = []
        total_input_tokens = 0
        total_tokens = 0
        total_batches = (len(texts) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

        for batch_idx in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[batch_idx : batch_idx + MAX_BATCH_SIZE]
            current_batch = batch_idx // MAX_BATCH_SIZE + 1

            logger.debug(
                f"Processing batch {current_batch}/{total_batches} "
                f"({len(batch)} texts)"
            )

            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=(
                        model_settings.dimensions
                        if model_settings.dimensions is not None
                        else openai.NOT_GIVEN
                    ),
                    encoding_format="base64",
                    timeout=model_settings.timeout,
                )
                embeddings.extend(
                    [
                        (
                            data.embedding
                            if isinstance(data.embedding, str)
                            else convert_float_list_to_base64(data.embedding)
                        )
                        for data in response.data
                    ]
                )

                # Accumulate actual token usage from API response
                total_input_tokens += response.usage.prompt_tokens
                total_tokens += response.usage.total_tokens

            except openai.RateLimitError as e:
                logger.error(f"Rate limit hit on batch {current_batch}: {str(e)}")
                raise RuntimeError(
                    f"Rate limit exceeded while processing batch "
                    f"{current_batch}/{total_batches}. "
                    f"Consider implementing exponential backoff or reducing batch size."
                ) from e

            except openai.APIError as e:
                logger.error(f"API error on batch {current_batch}: {str(e)}")
                raise RuntimeError(
                    f"Failed to generate embeddings for batch "
                    f"{current_batch}/{total_batches} using model {self.model}: "
                    f"{str(e)}"
                ) from e

            except Exception as e:
                logger.error(f"Unexpected error on batch {current_batch}: {str(e)}")
                raise RuntimeError(
                    f"Unexpected error processing batch "
                    f"{current_batch}/{total_batches}: {str(e)}"
                ) from e

        return embeddings, Usage(
            input_tokens=total_input_tokens,
            total_tokens=total_tokens,
        )

    def get_embeddings(
        self,
        input: str | typing.List[str],
        model_settings: ModelSettings,
    ) -> ModelResponse:
        """
        Get embeddings for the input text with caching and batch processing.

        Args:
            input: Single string or list of strings to embed
            model_settings: Model configuration including dimensions and timeout

        Returns:
            ModelResponse containing embeddings and usage statistics

        Raises:
            ValueError: If input is invalid or model settings are incompatible
            TypeError: If input type is incorrect
            RuntimeError: If API calls fail
        """
        start_time = time.time()

        # Validate input
        _input = validate_input(input)

        # Validate model settings
        model_settings.validate_for_model(self.model)

        logger.debug(f"Processing {len(_input)} texts for embedding")

        # Initialize output and tracking
        _output: typing.List[typing.Text | None] = [None] * len(_input)
        _missing_idx: typing.List[int] = []
        cache_hits = 0

        # Check cache for existing embeddings
        if self._cache is not None:
            for i, item in enumerate(_input):
                cache_key = generate_cache_key(
                    model=self._model_str,
                    dimensions=model_settings.dimensions,
                    text=item,
                )
                cached_item = self._cache.get(cache_key)

                if cached_item is None:
                    _missing_idx.append(i)
                else:
                    _output[i] = str(cached_item)
                    cache_hits += 1
        else:
            _missing_idx = list(range(len(_input)))

        # Log cache statistics
        if self._cache is not None and _input:
            cache_hit_rate = cache_hits / len(_input)
            logger.debug(
                f"Cache hit rate: {cache_hit_rate:.2%}, "
                f"Processing {len(_missing_idx)} new embeddings"
            )

        # Process missing embeddings
        total_tokens = 0
        input_tokens = 0

        if _missing_idx:
            missing_texts = [_input[i] for i in _missing_idx]

            try:
                embeddings, usage = self._batch_api_calls(missing_texts, model_settings)

                # Use actual token counts from API response
                input_tokens = usage.input_tokens
                total_tokens = usage.total_tokens

                # Store results and update cache
                for missing_idx_pos, embedding in zip(_missing_idx, embeddings):
                    _output[missing_idx_pos] = embedding

                    if self._cache is not None:
                        cache_key = generate_cache_key(
                            model=self._model_str,
                            dimensions=model_settings.dimensions,
                            text=_input[missing_idx_pos],
                        )
                        self._cache.set(cache_key, embedding)

            except Exception as e:
                logger.error(f"Failed to process embeddings: {str(e)}")
                raise

        # Ensure all outputs are filled
        if any(item is None for item in _output):
            raise RuntimeError("Failed to generate embeddings for some inputs")

        elapsed_time = time.time() - start_time
        logger.debug(
            f"Embeddings generated in {elapsed_time:.3f}s "
            f"({len(_input)} texts, {len(_missing_idx)} API calls)"
        )

        return ModelResponse.model_validate(
            {
                "output": _output,
                "usage": Usage(
                    input_tokens=int(input_tokens),
                    total_tokens=int(total_tokens),
                    cache_hits=int(cache_hits),
                ),
            }
        )

    def get_embeddings_generator(
        self,
        input: typing.List[str],
        model_settings: ModelSettings,
        chunk_size: int = 100,
    ) -> typing.Generator[ModelResponse, None, None]:
        """
        Generate embeddings in chunks to manage memory for large datasets.

        Args:
            input: List of strings to embed
            model_settings: Model configuration
            chunk_size: Number of texts to process per chunk

        Yields:
            ModelResponse for each chunk

        Raises:
            ValueError: If chunk_size is invalid
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        # Validate all input first
        validated_input = validate_input(input)

        total_chunks = (len(validated_input) + chunk_size - 1) // chunk_size
        logger.debug(
            f"Processing {len(validated_input)} texts in {total_chunks} chunks "
            f"of size {chunk_size}"
        )

        for i in range(0, len(validated_input), chunk_size):
            chunk = validated_input[i : i + chunk_size]
            logger.debug(f"Processing chunk {i // chunk_size + 1}/{total_chunks}")
            yield self.get_embeddings(chunk, model_settings)


class AsyncOpenAIEmbeddingsModel:
    """Async version of OpenAI embeddings model with caching and batch processing."""

    def __init__(
        self,
        model: str | EmbeddingModel,
        openai_client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI,
        cache: diskcache.Cache | None = None,
    ) -> None:
        self.model = model
        self._client = openai_client
        self._cache = cache

        # Validate model
        self._model_str = str(model)
        logger.debug(
            f"Initialized AsyncOpenAIEmbeddingsModel with model: {self._model_str}"
        )

    async def _batch_api_calls(
        self,
        texts: typing.List[str],
        model_settings: ModelSettings,
    ) -> typing.Tuple[typing.List[str], Usage]:
        """Async version of batch API calls."""
        embeddings = []
        total_input_tokens = 0
        total_tokens = 0
        total_batches = (len(texts) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

        # Process batches concurrently with controlled concurrency
        max_concurrent_batches = 5  # Adjust based on rate limits
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def process_batch(
            batch_idx: int, batch: typing.List[str]
        ) -> typing.Tuple[typing.List[str], Usage]:
            async with semaphore:
                current_batch = batch_idx // MAX_BATCH_SIZE + 1
                logger.debug(
                    f"Processing batch {current_batch}/{total_batches} "
                    f"({len(batch)} texts)"
                )

                try:
                    response = await self._client.embeddings.create(
                        input=batch,
                        model=self.model,
                        dimensions=(
                            model_settings.dimensions
                            if model_settings.dimensions is not None
                            else openai.NOT_GIVEN
                        ),
                        encoding_format="base64",
                        timeout=model_settings.timeout,
                    )
                    batch_embeddings = [
                        (
                            data.embedding
                            if isinstance(data.embedding, str)
                            else convert_float_list_to_base64(data.embedding)
                        )
                        for data in response.data
                    ]
                    batch_usage = Usage(
                        input_tokens=response.usage.prompt_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                    return batch_embeddings, batch_usage

                except openai.RateLimitError as e:
                    logger.error(f"Rate limit hit on batch {current_batch}: {str(e)}")
                    raise RuntimeError(
                        f"Rate limit exceeded while processing batch "
                        f"{current_batch}/{total_batches}. "
                        "Consider implementing exponential backoff or "
                        "reducing batch size."
                    ) from e

                except openai.APIError as e:
                    logger.error(f"API error on batch {current_batch}: {str(e)}")
                    raise RuntimeError(
                        f"Failed to generate embeddings for batch "
                        f"{current_batch}/{total_batches} "
                        f"using model {self.model}: {str(e)}"
                    ) from e

                except Exception as e:
                    logger.error(f"Unexpected error on batch {current_batch}: {str(e)}")
                    raise RuntimeError(
                        f"Unexpected error processing batch "
                        f"{current_batch}/{total_batches}: {str(e)}"
                    ) from e

        # Create tasks for all batches
        tasks = []
        for batch_idx in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[batch_idx : batch_idx + MAX_BATCH_SIZE]
            tasks.append(process_batch(batch_idx, batch))

        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)

        # Flatten results and accumulate usage
        for batch_embeddings, batch_usage in batch_results:
            embeddings.extend(batch_embeddings)
            total_input_tokens += batch_usage.input_tokens
            total_tokens += batch_usage.total_tokens

        return embeddings, Usage(
            input_tokens=total_input_tokens,
            total_tokens=total_tokens,
        )

    async def get_embeddings(
        self,
        input: str | typing.List[str],
        model_settings: ModelSettings,
    ) -> ModelResponse:
        """
        Async version of get_embeddings with concurrent batch processing.

        Args:
            input: Single string or list of strings to embed
            model_settings: Model configuration including dimensions and timeout

        Returns:
            ModelResponse containing embeddings and usage statistics
        """
        start_time = time.time()

        # Validate input
        _input = validate_input(input)

        # Validate model settings
        model_settings.validate_for_model(self.model)

        logger.debug(f"Processing {len(_input)} texts for embedding (async)")

        # Initialize output and tracking
        _output: typing.List[typing.Text | None] = [None] * len(_input)
        _missing_idx: typing.List[int] = []
        cache_hits = 0

        # Check cache for existing embeddings
        if self._cache is not None:
            for i, item in enumerate(_input):
                cache_key = generate_cache_key(
                    model=self._model_str,
                    dimensions=model_settings.dimensions,
                    text=item,
                )
                cached_item = await asyncio.to_thread(self._cache.get, cache_key)

                if cached_item is None:
                    _missing_idx.append(i)
                else:
                    _output[i] = str(cached_item)
                    cache_hits += 1
        else:
            _missing_idx = list(range(len(_input)))

        # Log cache statistics
        if self._cache is not None and _input:
            cache_hit_rate = cache_hits / len(_input)
            logger.debug(
                f"Cache hit rate: {cache_hit_rate:.2%}, "
                f"Processing {len(_missing_idx)} new embeddings"
            )

        # Process missing embeddings
        total_tokens = 0
        input_tokens = 0

        if _missing_idx:
            missing_texts = [_input[i] for i in _missing_idx]

            try:
                embeddings, usage = await self._batch_api_calls(
                    missing_texts, model_settings
                )

                # Use actual token counts from API response
                input_tokens = usage.input_tokens
                total_tokens = usage.total_tokens

                # Store results and update cache
                for missing_idx_pos, embedding in zip(_missing_idx, embeddings):
                    _output[missing_idx_pos] = embedding

                    if self._cache is not None:
                        cache_key = generate_cache_key(
                            model=self._model_str,
                            dimensions=model_settings.dimensions,
                            text=_input[missing_idx_pos],
                        )
                        await asyncio.to_thread(self._cache.set, cache_key, embedding)

            except Exception as e:
                logger.error(f"Failed to process embeddings: {str(e)}")
                raise

        # Ensure all outputs are filled
        if any(item is None for item in _output):
            raise RuntimeError("Failed to generate embeddings for some inputs")

        elapsed_time = time.time() - start_time
        logger.debug(
            f"Embeddings generated in {elapsed_time:.3f}s "
            f"({len(_input)} texts, {len(_missing_idx)} API calls)"
        )

        return ModelResponse.model_validate(
            {
                "output": _output,
                "usage": Usage(
                    input_tokens=int(input_tokens),
                    total_tokens=int(total_tokens),
                    cache_hits=int(cache_hits),
                ),
            }
        )

    async def get_embeddings_generator(
        self,
        input: typing.List[str],
        model_settings: ModelSettings,
        chunk_size: int = 100,
    ) -> typing.AsyncGenerator[ModelResponse, None]:
        """
        Async generator for processing embeddings in chunks.

        Args:
            input: List of strings to embed
            model_settings: Model configuration
            chunk_size: Number of texts to process per chunk

        Yields:
            ModelResponse for each chunk
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        # Validate all input first
        validated_input = validate_input(input)

        total_chunks = (len(validated_input) + chunk_size - 1) // chunk_size
        logger.debug(
            f"Processing {len(validated_input)} texts in {total_chunks} chunks "
            f"of size {chunk_size}"
        )

        for i in range(0, len(validated_input), chunk_size):
            chunk = validated_input[i : i + chunk_size]
            logger.debug(f"Processing chunk {i // chunk_size + 1}/{total_chunks}")
            yield await self.get_embeddings(chunk, model_settings)
