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
import tiktoken

from .embedding_model import EmbeddingModel

__all__ = ["ModelSettings", "OpenAIEmbeddingsModel", "AsyncOpenAIEmbeddingsModel"]
__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

# Constants
MAX_BATCH_SIZE = 2048  # OpenAI's batch size limit
MAX_INPUT_TOKENS = 8191  # Maximum tokens per input
MAX_TOKENS_A_REQUEST = 300_000  # Maximum tokens per request


@functools.lru_cache(maxsize=MAX_BATCH_SIZE)
def generate_cache_key(
    model: str | None = None, dimensions: int | None = None, text: str | None = None
) -> str:
    """Generate a unique cache key for embedding storage.

    Combines model name, dimensions, and text hash to create a unique identifier.
    """
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
    """Get the default disk cache instance for embedding storage.

    Creates a cache directory at './.cache/embeddings.cache' if it doesn't exist.
    """
    return diskcache.Cache(directory="./.cache/embeddings.cache")


def py_float_list_to_b64_np32_array(float_list: typing.List[float]) -> str:
    """Convert a list of python floats to base64-encoded numpy float32 array."""
    array = np.array(float_list, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("utf-8")


def b64_np32_array_to_py_float_list(b64_np32_array: str) -> typing.List[float]:
    """Convert a base64-encoded numpy float32 array to a list of python floats."""
    return np.frombuffer(base64.b64decode(b64_np32_array), dtype=np.float32).tolist()


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count the number of tokens in a text using a given encoding."""
    return len(encoding.encode(text))


def count_tokens_in_batch(
    texts: typing.List[str], encoding: tiktoken.Encoding
) -> typing.List[int]:
    """Count the number of tokens in a batch of texts using a given encoding."""
    token_sequences = encoding.encode_batch(texts)
    return [len(tokens) for tokens in token_sequences]


def truncate_text(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> str:
    """Truncate a text to a maximum number of tokens using a given encoding."""
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text


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
        Decode all embeddings in one pass as a zero-copy memoryview.
        Avoids data duplication by returning a memory view of decoded bytes.
        """
        return memoryview(b"".join(base64.b64decode(s) for s in self.output))

    @functools.cached_property
    def _ndarray(self) -> np.ndarray:
        """
        Materialize the NumPy array once and cache it.
        Later calls to `to_numpy()` or `to_python()` return the cached view.
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
        *,
        cache: diskcache.Cache | None = None,
        encoding: tiktoken.Encoding | None = None,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        token_limit_policy: typing.Literal[
            "raise", "warn", "ignore", "truncate"
        ] = "truncate",
        token_limit_usage_percent: typing.Annotated[float, "Range: 1 to 100"] = 85,
    ) -> None:
        self.model = model
        self._client = openai_client

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug(
                f"Encoding for model {model} not found, "
                + "using default encoding gpt-4o"
            )
            self._encoding = encoding or tiktoken.encoding_for_model("gpt-4o")

        self._cache = cache
        self._max_batch_size = max_batch_size
        self._max_input_tokens = max_input_tokens
        self._token_limit_policy = token_limit_policy
        self._token_limit_usage_percent = token_limit_usage_percent

        # Calculate effective token limit
        self._effective_token_limit = int(
            self._max_input_tokens * self._token_limit_usage_percent / 100
        )

        # Validate model
        self._model_str = str(model)
        logger.debug(f"Initialized OpenAIEmbeddingsModel with model: {self._model_str}")

    def _handle_token_limits(self, texts: typing.List[str]) -> typing.List[str]:
        """
        Apply token limit policy to process texts within limits.
        Handles truncation, warnings, or errors based on configured policy.

        Args:
            texts: List of texts to process

        Returns:
            List of processed texts according to policy

        Raises:
            ValueError: If policy is "raise" and token limit exceeded
        """
        # Count tokens for each text
        token_counts = count_tokens_in_batch(texts, self._encoding)

        # Check if any text exceeds limit
        over_limit_indices = [
            i
            for i, count in enumerate(token_counts)
            if count > self._effective_token_limit
        ]

        if not over_limit_indices:
            return texts  # All texts within limit

        # Apply policy
        if self._token_limit_policy == "raise":
            max_tokens = max(token_counts[i] for i in over_limit_indices)
            raise ValueError(
                f"Token limit exceeded: {max_tokens} tokens > "
                f"{self._effective_token_limit} limit. "
                f"Consider using 'truncate' policy or increasing "
                f"token_limit_usage_percent."
            )

        elif self._token_limit_policy == "warn":
            max_tokens = max(token_counts[i] for i in over_limit_indices)
            logger.warning(
                f"Token limit exceeded: {max_tokens} tokens > "
                f"{self._effective_token_limit} limit. "
                f"Sending to provider anyway. "
                f"({len(over_limit_indices)} texts affected)"
            )
            return texts

        elif self._token_limit_policy == "ignore":
            return texts

        elif self._token_limit_policy == "truncate":
            # Truncate texts that exceed limit
            processed_texts = texts.copy()
            for i in over_limit_indices:
                processed_texts[i] = truncate_text(
                    texts[i], self._encoding, self._effective_token_limit
                )

            logger.debug(
                f"Truncated {len(over_limit_indices)} texts to "
                f"{self._effective_token_limit} tokens"
            )
            return processed_texts

        return texts  # Fallback

    def _batch_api_calls(
        self,
        texts: typing.List[str],
        model_settings: ModelSettings,
    ) -> typing.Tuple[typing.List[str], Usage]:
        """
        Process texts in batches to respect OpenAI API limits.
        Handles rate limiting, errors, and usage tracking across batches.

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
        total_batches = (len(texts) + self._max_batch_size - 1) // self._max_batch_size

        for batch_idx in range(0, len(texts), self._max_batch_size):
            batch = texts[batch_idx : batch_idx + self._max_batch_size]
            current_batch = batch_idx // self._max_batch_size + 1

            logger.debug(
                f"Processing batch {current_batch}/{total_batches} "
                f"({len(batch)} texts)"
            )

            # Apply token limit handling
            safe_batch = (
                batch
                if self._token_limit_policy == "ignore"
                else self._handle_token_limits(batch)
            )

            try:
                response = self._client.embeddings.create(
                    input=safe_batch,
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
                            else py_float_list_to_b64_np32_array(data.embedding)
                        )
                        for data in response.data
                    ]
                )

                # Accumulate actual token usage from API response
                total_input_tokens += response.usage.prompt_tokens
                total_tokens += response.usage.total_tokens

            except openai.RateLimitError as e:
                logger.error(f"Rate limit hit on batch {current_batch}: {str(e)}")
                logger.error(
                    f"Rate limit exceeded while processing batch "
                    f"{current_batch}/{total_batches}. "
                    f"Consider implementing exponential backoff or reducing batch size."
                )
                raise e

            except openai.NotFoundError as e:
                logger.error(f"Model not found on batch {current_batch}: {str(e)}")
                logger.error(
                    f"Model {self.model} not found while processing batch "
                    f"{current_batch}/{total_batches}. "
                    f"Consider using a different model."
                )
                raise e

            except openai.APIError as e:
                logger.error(f"API error on batch {current_batch}: {str(e)}")
                logger.error(
                    f"Failed to generate embeddings for batch "
                    f"{current_batch}/{total_batches} using model {self.model}: "
                    f"{str(e)}"
                )
                raise e

            except Exception as e:
                logger.error(f"Unexpected error on batch {current_batch}: {str(e)}")
                logger.error(
                    f"Unexpected error processing batch "
                    f"{current_batch}/{total_batches}: {str(e)}"
                )
                raise e

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
        Generate embeddings with intelligent caching and batch processing.
        Validates inputs, checks cache, and processes missing embeddings efficiently.

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
        Generate embeddings in chunks for memory-efficient processing.
        Ideal for large datasets that don't fit in memory at once.

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
        *,
        cache: diskcache.Cache | None = None,
        encoding: tiktoken.Encoding | None = None,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        token_limit_policy: typing.Literal[
            "raise", "warn", "ignore", "truncate"
        ] = "truncate",
        token_limit_usage_percent: typing.Annotated[float, "Range: 1 to 100"] = 85,
    ) -> None:
        self.model = model
        self._client = openai_client

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug(
                f"Encoding for model {model} not found, "
                + "using default encoding gpt-4o"
            )
            self._encoding = encoding or tiktoken.encoding_for_model("gpt-4o")

        self._cache = cache
        self._max_batch_size = max_batch_size
        self._max_input_tokens = max_input_tokens
        self._token_limit_policy = token_limit_policy
        self._token_limit_usage_percent = token_limit_usage_percent

        # Calculate effective token limit
        self._effective_token_limit = int(
            self._max_input_tokens * self._token_limit_usage_percent / 100
        )

        # Validate model
        self._model_str = str(model)
        logger.debug(
            f"Initialized AsyncOpenAIEmbeddingsModel with model: {self._model_str}"
        )

    def _handle_token_limits(self, texts: typing.List[str]) -> typing.List[str]:
        """
        Apply token limit policy to process texts within limits.
        Handles truncation, warnings, or errors based on configured policy.

        Args:
            texts: List of texts to process

        Returns:
            List of processed texts according to policy

        Raises:
            ValueError: If policy is "raise" and token limit exceeded
        """
        # Count tokens for each text
        token_counts = count_tokens_in_batch(texts, self._encoding)

        # Check if any text exceeds limit
        over_limit_indices = [
            i
            for i, count in enumerate(token_counts)
            if count > self._effective_token_limit
        ]

        if not over_limit_indices:
            return texts  # All texts within limit

        # Apply policy
        if self._token_limit_policy == "raise":
            max_tokens = max(token_counts[i] for i in over_limit_indices)
            raise ValueError(
                f"Token limit exceeded: {max_tokens} tokens > "
                f"{self._effective_token_limit} limit. "
                f"Consider using 'truncate' policy or increasing "
                f"token_limit_usage_percent."
            )

        elif self._token_limit_policy == "warn":
            max_tokens = max(token_counts[i] for i in over_limit_indices)
            logger.warning(
                f"Token limit exceeded: {max_tokens} tokens > "
                f"{self._effective_token_limit} limit. "
                f"Sending to provider anyway. "
                f"({len(over_limit_indices)} texts affected)"
            )
            return texts

        elif self._token_limit_policy == "ignore":
            return texts

        elif self._token_limit_policy == "truncate":
            # Truncate texts that exceed limit
            processed_texts = texts.copy()
            for i in over_limit_indices:
                processed_texts[i] = truncate_text(
                    texts[i], self._encoding, self._effective_token_limit
                )

            logger.debug(
                f"Truncated {len(over_limit_indices)} texts to "
                f"{self._effective_token_limit} tokens"
            )
            return processed_texts

        return texts  # Fallback

    async def _batch_api_calls(
        self,
        texts: typing.List[str],
        model_settings: ModelSettings,
    ) -> typing.Tuple[typing.List[str], Usage]:
        """
        Process texts in batches with concurrent API calls.
        Handles rate limiting and errors with controlled concurrency.
        """
        embeddings = []
        total_input_tokens = 0
        total_tokens = 0
        total_batches = (len(texts) + self._max_batch_size - 1) // self._max_batch_size

        # Process batches concurrently with controlled concurrency
        max_concurrent_batches = 5  # Adjust based on rate limits
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def process_batch(
            batch_idx: int, batch: typing.List[str]
        ) -> typing.Tuple[typing.List[str], Usage]:
            async with semaphore:
                current_batch = batch_idx // self._max_batch_size + 1
                logger.debug(
                    f"Processing batch {current_batch}/{total_batches} "
                    f"({len(batch)} texts)"
                )

                # Apply token limit handling
                safe_batch = (
                    batch
                    if self._token_limit_policy == "ignore"
                    else self._handle_token_limits(batch)
                )

                try:
                    response = await self._client.embeddings.create(
                        input=safe_batch,
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
                            else py_float_list_to_b64_np32_array(data.embedding)
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
                    logger.error(
                        "Rate limit exceeded while processing batch "
                        f"{current_batch}/{total_batches}. "
                        "Consider implementing exponential backoff or "
                        "reducing batch size."
                    )
                    raise e

                except openai.NotFoundError as e:
                    logger.error(f"Model not found on batch {current_batch}: {str(e)}")
                    logger.error(
                        f"Model {self.model} not found while processing batch "
                        f"{current_batch}/{total_batches}. "
                        f"Consider using a different model."
                    )
                    raise e

                except openai.APIError as e:
                    logger.error(f"API error on batch {current_batch}: {str(e)}")
                    logger.error(
                        f"Failed to generate embeddings for batch "
                        f"{current_batch}/{total_batches} using model {self.model}: "
                        f"{str(e)}"
                    )
                    raise e

                except Exception as e:
                    logger.error(f"Unexpected error on batch {current_batch}: {str(e)}")
                    logger.error(
                        f"Unexpected error processing batch "
                        f"{current_batch}/{total_batches}: {str(e)}"
                    )
                    raise e

        # Create tasks for all batches
        tasks = []
        for batch_idx in range(0, len(texts), self._max_batch_size):
            batch = texts[batch_idx : batch_idx + self._max_batch_size]
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
        Generate embeddings asynchronously with caching and concurrent processing.
        Processes multiple texts concurrently for improved performance.

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
        Generate embeddings in chunks asynchronously for memory-efficient processing.
        Processes large datasets in manageable chunks to avoid memory issues.

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
