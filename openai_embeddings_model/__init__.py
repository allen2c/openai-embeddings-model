import asyncio
import base64
import concurrent.futures
import enum
import functools
import hashlib
import json
import logging
import math
import os
import pathlib
import time
import typing
import weakref

import diskcache
import numpy as np
import openai
import pydantic
import tiktoken

from .embedding_model import EmbeddingModel

__all__ = [
    "CACHE_KEY_VERSION",
    "AsyncOpenAIEmbeddingsModel",
    "EmbeddingModel",
    "ModelResponse",
    "ModelSettings",
    "OpenAIEmbeddingsModel",
    "SimilarityResponse",
    "SimilarityResult",
    "Usage",
    "generate_cache_key",
    "get_default_cache",
]
__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

# Constants
MAX_BATCH_SIZE = 2048  # OpenAI's batch size limit
MAX_INPUT_TOKENS = 8191  # Maximum tokens per input
MAX_TOKENS_A_REQUEST = 300_000  # Maximum tokens per request

# Bumped whenever the key layout changes, so old entries are ignored rather
# than misread. v1 keys covered only model, dimensions and text.
CACHE_KEY_VERSION = "v2"

# Model names already warned about for approximate token counting, so the
# warning fires once per model rather than once per instance.
_warned_tokenizers: set[str] = set()

# Live models, so a forked child can rebuild what the fork invalidated.
_fork_sensitive_models: "weakref.WeakSet" = weakref.WeakSet()


def _reset_all_after_fork() -> None:
    """Rebuild per-process resources in a forked child.

    fork() does not carry threads into the child, so it inherits an executor
    whose workers no longer exist and a diskcache sqlite connection that may
    have been mid-transaction in the parent. Left alone, the child's first
    cache write blocks on a lock nothing will ever release — the classic
    gunicorn `preload_app` hang.
    """
    for model in list(_fork_sensitive_models):
        try:
            model._reset_after_fork()
        except Exception:  # a child that cannot reset must not die here
            logger.debug("Post-fork reset failed", exc_info=True)


if hasattr(os, "register_at_fork"):  # not available on Windows
    os.register_at_fork(after_in_child=_reset_all_after_fork)


def _canonical_for_digest(value: typing.Any) -> typing.Any:
    """Stringify dict keys recursively so the payload can be sorted.

    `json.dumps(sort_keys=True)` compares the original key objects, so a dict
    mixing key types raises TypeError before serialisation. JSON stringifies
    keys anyway, so doing it first loses nothing and makes any dict sortable.
    """
    if isinstance(value, dict):
        return {str(k): _canonical_for_digest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_for_digest(v) for v in value]
    return value


def cache_scope_digest(provider: str | None = None, extra_body: dict | None = None) -> str:
    """Digest the request context that changes an embedding but not its text.

    Two requests for the same text under the same model can still produce
    different vectors: a different provider is a different model behind the
    same name, and `extra_body` carries provider parameters such as Voyage's
    `output_dimension`. Both belong in the cache key.
    """
    if not provider and not extra_body:
        return "default"
    payload = json.dumps(
        _canonical_for_digest({"provider": provider or "", "extra_body": extra_body or {}}),
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _hash_text(text: str) -> str:
    """Digest one text for the cache key.

    surrogatepass: text carrying lone surrogates from a mis-decoded source
    should not abort the whole batch.
    """
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _format_cache_key(model: str | None, dimensions: int | None, scope: str, hash_text: str) -> str:
    """Assemble a cache key from parts already computed.

    The single place the key layout is written down, so the per-text and
    per-request paths cannot drift apart. Changing this layout invalidates
    every user's cache and re-embeds their corpus — see `CACHE_KEY_VERSION`.
    """
    # `is not None` rather than truthiness: `dimensions=0` and `model=""` are
    # distinct requests, and coalescing them onto the shared 'default' and
    # 'unknown' segments makes them collide with unrelated entries.
    return (
        f"{CACHE_KEY_VERSION}:{model if model is not None else 'unknown'}:"
        f"{dimensions if dimensions is not None else 'default'}:"
        f"{scope}:{hash_text}"
    )


def generate_cache_key(
    model: str | None = None,
    dimensions: int | None = None,
    text: str | None = None,
    *,
    provider: str | None = None,
    extra_body: dict | None = None,
) -> str:
    """Generate a unique cache key for embedding storage.

    Combines the key version, model name, dimensions, a digest of the request
    scope (provider and `extra_body`), and a hash of the text.
    """
    if text is None:
        raise ValueError("text is required")
    return _format_cache_key(model, dimensions, cache_scope_digest(provider, extra_body), _hash_text(text))


def _write_cache_entries(cache: diskcache.Cache, items: typing.Sequence[tuple[str, str]]) -> None:
    """Write a batch of entries in one sqlite transaction.

    Left unwrapped, every `set()` is its own `BEGIN`/`COMMIT` round trip.
    Batching them is 1.6-2.4x faster on the write path, and stays that fast
    whatever the batch size and however full the cache already is.

    The trade is a real one: `transact()` rolls the *whole* batch back if any
    single write raises, where an unwrapped loop leaves the earlier entries
    committed. Those embeddings have already been paid for, so such a failure
    costs re-embedding the batch rather than just its tail. It is accepted
    because a cache write fails only on a full or broken disk, and the caller
    still receives every vector either way — only the cached copy is lost.
    """
    transact = getattr(cache, "transact", None)
    if transact is None:  # a cache-like object that is not a diskcache.Cache
        for key, value in items:
            cache.set(key, value)
        return
    with transact():
        for key, value in items:
            cache.set(key, value)


def validate_input(input: str | list[str]) -> list[str]:
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


def py_float_list_to_b64_np32_array(float_list: list[float]) -> str:
    """Convert a list of python floats to base64-encoded numpy float32 array."""
    array = np.array(float_list, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("utf-8")


def b64_np32_array_to_py_float_list(b64_np32_array: str) -> list[float]:
    """Convert a base64-encoded numpy float32 array to a list of python floats."""
    return np.frombuffer(base64.b64decode(b64_np32_array), dtype=np.float32).tolist()


def validate_cached_embedding(key: str, cached: typing.Any, expected_dimensions: int | None = None) -> str | None:
    """Validate a raw cache entry, returning None when it is unusable.

    A cache directory can accumulate entries written by another tool or an
    older version of this library. Passing those through `str()` silently
    produces a syntactically valid but meaningless vector, so anything that
    does not decode into a plausible float32 embedding is treated as a miss
    and re-fetched from the provider instead.
    """
    if cached is None:
        return None

    if not isinstance(cached, str):
        logger.warning(f"Discarding cache entry {key}: expected str, got {type(cached).__name__}")
        return None

    try:
        raw = base64.b64decode(cached, validate=True)
    except Exception:
        logger.warning(f"Discarding cache entry {key}: not valid base64")
        return None

    if not raw or len(raw) % 4 != 0:
        logger.warning(f"Discarding cache entry {key}: {len(raw)} bytes is not a positive multiple of 4")
        return None

    if expected_dimensions is not None and len(raw) // 4 != expected_dimensions:
        logger.warning(
            f"Discarding cache entry {key}: decodes to {len(raw) // 4} dimensions, expected {expected_dimensions}"
        )
        return None

    if not np.isfinite(np.frombuffer(raw, dtype=np.float32)).all():
        logger.warning(f"Discarding cache entry {key}: contains NaN or inf")
        return None

    return cached


def extract_ordered_embeddings(
    data: typing.Sequence[typing.Any],
) -> list[str]:
    """Extract base64 embeddings from response data, ordered by provider index.

    OpenAI documents that `data` comes back in request order, but proxies and
    other OpenAI-compatible providers make no such promise. `index` is the
    authoritative position, so it is used whenever every item supplies one.
    """
    items = list(data)
    if items and all(isinstance(getattr(d, "index", None), int) for d in items):
        items.sort(key=lambda d: d.index)
    return [
        (d.embedding if isinstance(d.embedding, str) else py_float_list_to_b64_np32_array(d.embedding)) for d in items
    ]


# Transient failures worth retrying. Everything else (a missing model, a
# malformed request) will fail again identically, so retrying only delays it.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def deduplicate_texts(
    texts: typing.Sequence[str],
) -> tuple[list[str], list[int]]:
    """Collapse repeated texts, keeping first-seen order.

    Identical texts have identical embeddings, so sending each copy is paying
    the provider more than once for the same vector.

    Returns:
        Tuple of (unique texts, slot in `unique` for each original text)
    """
    unique: list[str] = []
    slot_of: dict[str, int] = {}
    slots: list[int] = []

    for text in texts:
        slot = slot_of.get(text)
        if slot is None:
            slot = len(unique)
            slot_of[text] = slot
            unique.append(text)
        slots.append(slot)

    return unique, slots


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count the number of tokens in a text using a given encoding."""
    return len(encoding.encode(text))


def count_tokens_in_batch(texts: list[str], encoding: tiktoken.Encoding) -> list[int]:
    """Count the number of tokens in a batch of texts using a given encoding."""
    token_sequences = encoding.encode_batch(texts)
    return [len(tokens) for tokens in token_sequences]


def truncate_text(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> str:
    """Truncate a text to a maximum number of tokens using a given encoding."""
    # Clamp first: a negative bound would slice from the end, keeping all but
    # the last N tokens — the opposite of a cap.
    max_tokens = max(0, max_tokens)
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
    extra_body: dict | None = None

    def validate_for_model(self, model: str | EmbeddingModel) -> None:
        """Validate settings are appropriate for the given model.

        Raises ValueError when `dimensions` is not usable with a known model.
        Unknown model names are left alone — only the provider can judge them.
        """
        model_str = str(model)

        # Scoped narrowly to the lookup: widening it would swallow the
        # validation errors raised below and silently accept anything.
        try:
            model_type = EmbeddingModelType(model_str)
        except ValueError:
            logger.debug(f"Unknown model type: {model_str}, skipping dimension validation")
            return

        if self.dimensions is None:
            return

        if not model_type.supports_dimensions:
            raise ValueError(f"Model {model_str} does not support custom dimensions")

        max_dims = model_type.max_dimensions
        if max_dims and not (1 <= self.dimensions <= max_dims):
            raise ValueError(f"Dimensions must be between 1 and {max_dims} for {model_str}, got {self.dimensions}")


class Usage(pydantic.BaseModel):
    """Token usage statistics."""

    input_tokens: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    truncated_texts: int = 0
    """Texts shortened to fit the token limit. Non-zero means input was
    dropped before embedding, which the vectors themselves cannot show."""


class ModelResponse(pydantic.BaseModel):
    """Response from embedding model with lazy decoding."""

    # Frozen because the decoded array is cached on first access and never
    # invalidated; a mutable `output` would leave `to_numpy()` silently
    # returning vectors for text the response no longer holds.
    model_config = pydantic.ConfigDict(frozen=True)

    output: list[str]
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
        """Return embeddings as a writable (n, d) float32 ndarray.

        The decoded buffer is cached, but each call returns a fresh copy of
        it. The cached array is a read-only view over that buffer, and native
        libraries that write through raw pointers (faiss, for example) ignore
        numpy's read-only flag — handing out the view directly would let them
        corrupt every later `to_numpy()` / `to_python()` result.
        """
        return self._ndarray.copy()

    def to_python(self) -> list[list[float]]:
        """Return embeddings as ordinary Python lists.

        The decoded array behind this is cached, but the list itself is built
        fresh on every call — about 43 ms for 2048 x 1536. Caching it would
        hand every caller the same mutable list, which is exactly what
        `to_numpy()`'s copy exists to prevent, so keep the result if you need
        it twice.
        """
        return self._ndarray.tolist()

    def as_similarity_response(self) -> "SimilarityResponse":
        from openai_embeddings_model.normalize import normalize

        embeddings = normalize(self.to_numpy())

        similarity_matrix = embeddings[0:1, :] @ embeddings[1:, :].T  # Shape: (1, length)
        # reshape rather than squeeze: a single document yields a (1, 1)
        # matrix, and squeeze() would collapse that to a non-iterable 0-d array
        relevance_scores = similarity_matrix.reshape(-1)  # Shape: (length, )

        similarity_response = SimilarityResponse.model_validate(
            {
                "results": [
                    SimilarityResult(index=i, relevance_score=score) for i, score in enumerate(relevance_scores)
                ],
                "usage": Usage.model_validate_json(self.usage.model_dump_json()),
            }
        )
        # NaN compares False against everything, so sorting on the raw score
        # can leave a NaN result sitting at the top as the "best" match.
        similarity_response.results.sort(
            key=lambda x: (-math.inf if math.isnan(x.relevance_score) else x.relevance_score),
            reverse=True,
        )

        return similarity_response


class SimilarityResult(pydantic.BaseModel):
    index: int
    relevance_score: float = 0.0


class SimilarityResponse(pydantic.BaseModel):
    """Response from similarity model with lazy decoding."""

    results: list[SimilarityResult]
    usage: Usage


class _OpenAIEmbeddingsModelBase:
    def __init__(
        self,
        model: str | EmbeddingModel,
        openai_client: openai.OpenAI | openai.AzureOpenAI | openai.AsyncOpenAI | openai.AsyncAzureOpenAI,
        *args,
        cache: diskcache.Cache | None = None,
        encoding: tiktoken.Encoding | None = None,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_tokens_a_request: int = MAX_TOKENS_A_REQUEST,
        token_limit_policy: typing.Literal["raise", "warn", "ignore", "truncate"] = "truncate",
        token_limit_usage_percent: typing.Annotated[float, "Range: 1 to 100"] = 85,
        dimensions_parameter: typing.Literal["dimensions", "output_dimension"] | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        **kwargs,
    ) -> None:
        self.model = model
        self._client = openai_client
        self._model_str = str(model)

        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_input_tokens < 1:
            raise ValueError(f"max_input_tokens must be >= 1, got {max_input_tokens}")
        if max_tokens_a_request < 1:
            raise ValueError(f"max_tokens_a_request must be >= 1, got {max_tokens_a_request}")
        if not 0 < token_limit_usage_percent <= 100:
            raise ValueError(f"token_limit_usage_percent must be in (0, 100], got {token_limit_usage_percent}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        # An explicit encoding always wins: auto-detection cannot know the
        # tokenizer of a model tiktoken has never heard of.
        if encoding is not None:
            self._encoding = encoding
        else:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model_str)
            except Exception:
                if self._model_str not in _warned_tokenizers:
                    _warned_tokenizers.add(self._model_str)
                    logger.warning(
                        f"No tiktoken encoding for {self._model_str}; falling back "
                        "to gpt-4o. Token counts, and therefore truncation points, "
                        "are approximate. Pass encoding= to make them exact."
                    )
                self._encoding = tiktoken.encoding_for_model("gpt-4o")

        self._cache = cache
        self._max_batch_size = max_batch_size
        self._max_input_tokens = max_input_tokens
        self._max_tokens_a_request = max_tokens_a_request
        self._token_limit_policy = token_limit_policy
        self._token_limit_usage_percent = token_limit_usage_percent
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

        # Calculate effective token limit
        self._effective_token_limit = max(1, int(self._max_input_tokens * self._token_limit_usage_percent / 100))

        # Which request parameter carries custom dimensions. Voyage models take
        # `output_dimension` in extra_body; everything else takes `dimensions`.
        # Auto-detection matches a `voyage` prefix, so a model merely
        # containing the word is not misread; pass this explicitly for
        # deployment aliases that hide the underlying model.
        self._dimensions_parameter = dimensions_parameter or (
            "output_dimension" if self._model_str.lower().startswith("voyage") else "dimensions"
        )

        # Identifies the provider in cache keys: the same model name behind a
        # different base_url is a different model.
        self._provider = str(getattr(openai_client, "base_url", "") or "")

        _fork_sensitive_models.add(self)

        logger.debug(f"Initialized {self.__class__.__name__} with model: {self._model_str}")

    def _reset_after_fork(self) -> None:
        """Drop resources a forked child inherited but cannot use.

        The sqlite connection diskcache opened in the parent is not valid in
        the child; closing it makes diskcache open a fresh one on next use.
        """
        if self._cache is not None:
            self._cache.close()

    def _handle_token_limits(self, texts: list[str], token_counts: list[int]) -> tuple[list[str], list[int]]:
        """
        Apply token limit policy to process texts within limits.
        Handles truncation, warnings, or errors based on configured policy.

        Args:
            texts: List of texts to process
            token_counts: Token count per text, already measured

        Returns:
            Tuple of (processed texts, indices that were truncated)

        Raises:
            ValueError: If policy is "raise" and token limit exceeded
        """
        over_limit_indices = [i for i, count in enumerate(token_counts) if count > self._effective_token_limit]

        if not over_limit_indices:
            return texts, []

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
            return texts, []

        elif self._token_limit_policy == "ignore":
            return texts, []

        elif self._token_limit_policy == "truncate":
            processed_texts = texts.copy()
            for i in over_limit_indices:
                processed_texts[i] = truncate_text(texts[i], self._encoding, self._effective_token_limit)

            logger.warning(
                f"Truncated {len(over_limit_indices)} of {len(texts)} texts to "
                f"{self._effective_token_limit} tokens; input was dropped. "
                "See usage.truncated_texts."
            )
            return processed_texts, over_limit_indices

        return texts, []  # Fallback

    def _prepare_batches(self, texts: list[str]) -> tuple[list[str], list[list[int]], int]:
        """Apply the token limit policy, then group texts into requests.

        Batches respect both `max_batch_size` and `max_tokens_a_request`;
        splitting on item count alone lets a batch of long texts build a
        request far past any provider's limit.

        Token counts are needed for both jobs, so they are measured once here
        and reused. This is the CPU-heavy step — the async model runs it in
        its executor rather than on the event loop.

        Returns:
            Tuple of (processed texts, batches as index groups, truncated count)
        """
        token_counts = count_tokens_in_batch(texts, self._encoding)
        safe_texts, truncated_indices = self._handle_token_limits(texts, token_counts)

        if truncated_indices:
            # Re-measure the truncated texts rather than assuming they land on
            # exactly the limit. Slicing at a token boundary can cut a
            # multi-byte character in half, and decode() replaces the remnant
            # with U+FFFD, which re-encodes to an extra token. Assuming the
            # limit would undercount the real request. Only the changed texts
            # are re-encoded.
            token_counts = list(token_counts)
            for index in truncated_indices:
                token_counts[index] = len(self._encoding.encode(safe_texts[index]))

        batches: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0

        for index, count in enumerate(token_counts):
            exceeds_items = len(current) >= self._max_batch_size
            exceeds_tokens = current_tokens + count > self._max_tokens_a_request
            if current and (exceeds_items or exceeds_tokens):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(index)
            current_tokens += count

        if current:
            batches.append(current)

        return safe_texts, batches, len(truncated_indices)

    def _cache_keys_for(self, texts: typing.Sequence[str], model_settings: ModelSettings) -> list[str]:
        """Cache keys for a whole request, digesting the scope once.

        `cache_scope_digest` is a `json.dumps` plus a sha256 over the provider
        and `extra_body`, both fixed for the whole request. Calling
        `generate_cache_key` per text recomputed it every time, which was two
        thirds of key generation on a large call. The keys are byte-identical
        either way.
        """
        scope = cache_scope_digest(self._provider, model_settings.extra_body)
        dimensions = model_settings.dimensions
        return [_format_cache_key(self._model_str, dimensions, scope, _hash_text(text)) for text in texts]

    def _resolve_usage(self, response: typing.Any, safe_batch: list[str]) -> Usage:
        """Read token usage from a provider response.

        Not every OpenAI-compatible provider populates `usage`; when it is
        missing the counts are recomputed locally with tiktoken so both the
        sync and async paths return a usable `Usage` instead of failing.
        """
        usage = response.usage
        if usage is None:
            logger.debug(
                f"Provider {self._client.base_url} does not support usage "
                "information. Using self tiktoken calculation."
            )
            batch_tokens = sum(count_tokens_in_batch(safe_batch, self._encoding))
            return Usage(input_tokens=batch_tokens, total_tokens=batch_tokens)

        return Usage(
            input_tokens=(usage.prompt_tokens if usage.prompt_tokens is not None else usage.total_tokens),
            total_tokens=usage.total_tokens,
        )

    def _build_extra_kwargs(self, model_settings: ModelSettings) -> dict[str, typing.Any]:
        """Build the provider-specific request kwargs.

        Raises ValueError if `extra_body` cannot be serialised, rather than
        letting the failure surface from inside the HTTP layer.
        """
        result: dict[str, typing.Any] = {}
        extra_body: dict[str, typing.Any] = {}

        if model_settings.extra_body is not None:
            try:
                # Round-tripped to detach from the caller's dict and to fail
                # here rather than mid-request. Note this is what the wire
                # format does anyway: tuples become lists, keys become strings.
                extra_body = json.loads(json.dumps(model_settings.extra_body))
            except (TypeError, ValueError) as e:
                raise ValueError(f"model_settings.extra_body must be JSON-serialisable: {e}") from e

        if self._dimensions_parameter == "dimensions":
            result["dimensions"] = (
                model_settings.dimensions if model_settings.dimensions is not None else openai.NOT_GIVEN
            )
        elif model_settings.dimensions is not None:
            derived = model_settings.dimensions
            override = extra_body.get("output_dimension")
            if override is not None and override != derived:
                # Silently letting one win produced vectors of a size the
                # caller never asked for.
                logger.warning(
                    f"extra_body['output_dimension']={override} overrides "
                    f"model_settings.dimensions={derived} for {self._model_str}"
                )
            extra_body.setdefault("output_dimension", derived)

        if extra_body:
            result["extra_body"] = extra_body

        return result


class OpenAIEmbeddingsModel(_OpenAIEmbeddingsModelBase):
    """Thread-safe OpenAI embeddings model with caching and batch processing."""

    # Cache I/O lives here rather than on the base class: these calls block,
    # and the async model must never reach them. It uses the batched
    # `_cache_get_many` / `_cache_set_many`, which go through its executor.
    def _cache_get(self, key: str, expected_dimensions: int | None = None) -> str | None:
        if self._cache is None:
            return None
        return validate_cached_embedding(key, self._cache.get(key), expected_dimensions)

    def _cache_set_many(self, items: typing.Sequence[tuple[str, str]]) -> None:
        """Write one batch's entries in a single transaction."""
        if self._cache is not None and items:
            _write_cache_entries(self._cache, items)

    @property
    def client(self) -> openai.OpenAI | openai.AzureOpenAI:
        if not isinstance(self._client, (openai.OpenAI, openai.AzureOpenAI)):
            raise TypeError(f"Expected a sync OpenAI client, got {type(self._client).__name__}")
        return self._client

    def _create_with_retry(
        self,
        batch: list[str],
        model_settings: ModelSettings,
        batch_no: int,
        total_batches: int,
    ) -> typing.Any:
        """Call the provider for one batch, retrying transient failures."""
        extra_kwargs = self._build_extra_kwargs(model_settings)
        attempt = 0

        while True:
            try:
                return self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    encoding_format="base64",
                    timeout=model_settings.timeout,
                    **extra_kwargs,
                )
            except RETRYABLE_ERRORS as e:
                if attempt >= self._max_retries:
                    logger.error(f"Batch {batch_no}/{total_batches} failed after {attempt + 1} attempt(s): {e}")
                    raise
                delay = self._retry_base_delay * (2**attempt)
                logger.warning(f"Batch {batch_no}/{total_batches} hit {type(e).__name__}, retrying in {delay:.1f}s")
                time.sleep(delay)
                attempt += 1
            except Exception as e:
                logger.error(f"Batch {batch_no}/{total_batches} failed on model {self.model}: {e}")
                raise

    def _embed_missing(
        self,
        texts: list[str],
        keys: list[str],
        model_settings: ModelSettings,
    ) -> tuple[list[str], Usage]:
        """
        Embed texts that were not cached, in batches within provider limits.

        Each batch is written to the cache as soon as it succeeds, so a later
        batch failing does not discard embeddings already paid for.

        Args:
            texts: Texts to embed
            keys: Cache key for each text, same order
            model_settings: Model configuration

        Returns:
            Tuple of (List of base64-encoded embeddings, Usage statistics)
        """
        safe_texts, batches, truncated = self._prepare_batches(texts)
        results: list[str | None] = [None] * len(texts)
        total_input_tokens = 0
        total_tokens = 0

        for batch_no, group in enumerate(batches, start=1):
            batch = [safe_texts[i] for i in group]
            logger.debug(f"Processing batch {batch_no}/{len(batches)} ({len(batch)} texts)")

            response = self._create_with_retry(batch, model_settings, batch_no, len(batches))
            batch_embeddings = extract_ordered_embeddings(response.data)

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Provider returned {len(batch_embeddings)} embeddings for "
                    f"{len(batch)} inputs in batch {batch_no}/{len(batches)}"
                )

            written: list[tuple[str, str]] = []
            for index, embedding in zip(group, batch_embeddings, strict=True):
                results[index] = embedding
                written.append((keys[index], embedding))
            # One transaction per batch, not per embedding. The batch is
            # already the unit of durability here: it is written only once the
            # provider call it came from has succeeded.
            self._cache_set_many(written)

            batch_usage = self._resolve_usage(response, batch)
            total_input_tokens += batch_usage.input_tokens
            total_tokens += batch_usage.total_tokens

        return typing.cast(list[str], results), Usage(
            input_tokens=total_input_tokens,
            total_tokens=total_tokens,
            truncated_texts=truncated,
        )

    def get_embeddings(
        self,
        input: str | list[str],
        model_settings: ModelSettings,
    ) -> ModelResponse:
        """
        Generate embeddings with intelligent caching and batch processing.
        Validates inputs, checks cache, and processes missing embeddings efficiently.

        Repeated texts are embedded once and the result shared, so passing the
        same string several times costs one provider call, not several.

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

        _input = validate_input(input)
        model_settings.validate_for_model(self.model)

        logger.debug(f"Processing {len(_input)} texts for embedding")

        unique_texts, slots = deduplicate_texts(_input)
        keys = self._cache_keys_for(unique_texts, model_settings)
        resolved: list[str | None] = [self._cache_get(key, model_settings.dimensions) for key in keys]

        missing = [slot for slot, value in enumerate(resolved) if value is None]
        # Counted per input item, so a text repeated twice and served from
        # cache still reports two hits.
        cache_hits = sum(1 for slot in slots if resolved[slot] is not None)

        if self._cache is not None and _input:
            logger.debug(f"Cache hit rate: {cache_hits / len(_input):.2%}, Processing {len(missing)} new embeddings")

        usage = Usage()
        if missing:
            try:
                embeddings, usage = self._embed_missing(
                    [unique_texts[slot] for slot in missing],
                    [keys[slot] for slot in missing],
                    model_settings,
                )
            except Exception as e:
                logger.error(f"Failed to process embeddings: {e!s}")
                raise

            for slot, embedding in zip(missing, embeddings, strict=True):
                resolved[slot] = embedding

        _output = [resolved[slot] for slot in slots]
        if any(item is None for item in _output):
            raise RuntimeError("Failed to generate embeddings for some inputs")

        elapsed_time = time.time() - start_time
        logger.debug(f"Embeddings generated in {elapsed_time:.3f}s ({len(_input)} texts, {len(missing)} embedded)")

        return ModelResponse.model_validate(
            {
                "output": _output,
                "usage": Usage(
                    input_tokens=int(usage.input_tokens),
                    total_tokens=int(usage.total_tokens),
                    cache_hits=int(cache_hits),
                    truncated_texts=int(usage.truncated_texts),
                ),
            }
        )

    def get_embeddings_generator(
        self,
        input: list[str],
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
        logger.debug(f"Processing {len(validated_input)} texts in {total_chunks} chunks of size {chunk_size}")

        for i in range(0, len(validated_input), chunk_size):
            chunk = validated_input[i : i + chunk_size]
            logger.debug(f"Processing chunk {i // chunk_size + 1}/{total_chunks}")
            yield self.get_embeddings(chunk, model_settings)

    def get_similarity(
        self,
        query: str,
        documents: list[str],
        *,
        model_settings: ModelSettings,
    ) -> SimilarityResponse:
        if not documents:
            raise ValueError("documents is required")
        if not query:
            raise ValueError("query is required")

        # `+` deliberately, not `[query, *documents]`: concatenation rejects a
        # bare `str` for `documents`, where unpacking would silently spread it
        # into single characters and rank letters.
        embeddings_res = self.get_embeddings([query] + documents, model_settings)  # noqa: RUF005

        return embeddings_res.as_similarity_response()


class AsyncOpenAIEmbeddingsModel(_OpenAIEmbeddingsModelBase):
    """Async version of OpenAI embeddings model with caching and batch processing."""

    def __init__(
        self,
        *args,
        executor_max_workers: int | None = 1,
        max_concurrent_batches: int = 5,
        **kwargs,
    ) -> None:
        """
        Args:
            executor_max_workers: Threads for cache I/O and tokenisation.
                One, deliberately, because a local `diskcache` makes this pool
                GIL-bound — sqlite queries and tiktoken — so extra workers buy
                contention rather than parallelism. 32 concurrent all-hit calls
                finish 4.3x faster on one worker than on the
                `min(32, cpu_count + 4)` default, and every larger value
                measured worse. `aiosqlite` reaches the same design from the
                same constraint, with one dedicated thread per connection.

                **Raise it to roughly your concurrency if your cache blocks on
                I/O** — anything remote, or a cache-like object of your own
                over the network. Threads waiting on a socket really do
                overlap, and the answer inverts sharply: at 0.5 ms per cache
                read, eight workers were 6.9x faster than one; at 10 ms,
                fourteen were 8.0x faster. Half a millisecond is enough to
                flip it. `None` restores the stdlib default.
            max_concurrent_batches: Provider requests in flight at once.
        """
        super().__init__(*args, **kwargs)
        if max_concurrent_batches < 1:
            raise ValueError(f"max_concurrent_batches must be >= 1, got {max_concurrent_batches}")
        self._max_concurrent_batches = max_concurrent_batches
        self._executor_max_workers = executor_max_workers
        self._executor = self._new_executor()

    def _new_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=self._executor_max_workers,
            thread_name_prefix=f"openai-emb-async-{id(self)}",
        )

    def _reset_after_fork(self) -> None:
        """Rebuild the thread pool alongside the base class's cache reset.

        The child inherits an executor whose worker threads did not survive
        the fork, so anything submitted to it would never run.
        """
        super()._reset_after_fork()
        self._executor = self._new_executor()

    async def aclose(self) -> None:
        """Shut down the dedicated cache-I/O thread pool.

        Each instance owns a `ThreadPoolExecutor`. CPython reaps an executor's
        workers once the executor itself becomes unreachable, so this is about
        releasing them at a point you choose rather than whenever the garbage
        collector gets there — and about the case where something outlives the
        model still holding the pool. Safe to call repeatedly; the model should
        not be used afterwards.
        """
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def __aenter__(self) -> "AsyncOpenAIEmbeddingsModel":
        return self

    async def __aexit__(self, *exc_info: typing.Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        # Safety net for instances dropped without aclose(): ends the workers
        # even when something else still references the pool.
        executor = getattr(self, "_executor", None)
        if executor is None:
            return
        # `contextlib.suppress` is a global lookup, and module globals are
        # already being torn down by the time `__del__` runs at interpreter
        # shutdown. A bare try/except needs nothing from outside.
        try:  # noqa: SIM105
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # interpreter shutdown can make this unreliable
            pass

    @property
    def client(self) -> openai.AsyncOpenAI | openai.AsyncAzureOpenAI:
        if not isinstance(self._client, (openai.AsyncOpenAI, openai.AsyncAzureOpenAI)):
            raise TypeError(f"Expected an async OpenAI client, got {type(self._client).__name__}")
        return self._client

    async def _cache_get_many(self, keys: list[str], expected_dimensions: int | None = None) -> list[str | None]:
        """Read many keys in one executor job.

        One `run_in_executor` per key costs a cross-thread round trip each
        time, which dominates the work itself for anything but tiny inputs.
        """
        if self._cache is None or not keys:
            return [None] * len(keys)

        cache = self._cache
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(self._executor, lambda: [cache.get(key) for key in keys])
        return [
            validate_cached_embedding(key, value, expected_dimensions) for key, value in zip(keys, raw, strict=True)
        ]

    async def _cache_set_many(self, items: typing.Sequence[tuple[str, str]]) -> None:
        """Write many entries in one executor job."""
        if self._cache is None or not items:
            return

        cache = self._cache
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _write_cache_entries, cache, items)

    async def _resolve_usage_async(self, response: typing.Any, batch: list[str]) -> Usage:
        """Resolve usage, offloading the tiktoken fallback off the loop."""
        if response.usage is not None:
            return self._resolve_usage(response, batch)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._resolve_usage, response, batch)

    async def _create_with_retry(
        self,
        batch: list[str],
        model_settings: ModelSettings,
        batch_no: int,
        total_batches: int,
    ) -> typing.Any:
        """Call the provider for one batch, retrying transient failures."""
        extra_kwargs = self._build_extra_kwargs(model_settings)
        attempt = 0

        while True:
            try:
                return await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    encoding_format="base64",
                    timeout=model_settings.timeout,
                    **extra_kwargs,
                )
            except RETRYABLE_ERRORS as e:
                if attempt >= self._max_retries:
                    logger.error(f"Batch {batch_no}/{total_batches} failed after {attempt + 1} attempt(s): {e}")
                    raise
                delay = self._retry_base_delay * (2**attempt)
                logger.warning(f"Batch {batch_no}/{total_batches} hit {type(e).__name__}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                attempt += 1
            except Exception as e:
                logger.error(f"Batch {batch_no}/{total_batches} failed on model {self.model}: {e}")
                raise

    async def _embed_missing(
        self,
        texts: list[str],
        keys: list[str],
        model_settings: ModelSettings,
    ) -> tuple[list[str], Usage]:
        """
        Embed uncached texts with concurrent, size-limited batches.

        Each batch is cached as soon as it succeeds, and a failure cancels its
        siblings rather than leaving them to finish requests nobody will read.

        A retrying batch holds its concurrency slot while it backs off. Rate
        limits are usually global to the provider, so not dispatching new
        requests while one is backing off is the intent — but it does mean a
        rate-limit storm can fill every slot with waiting batches.
        """
        loop = asyncio.get_running_loop()
        # Token counting is CPU-bound and would otherwise stall the loop for
        # the whole call on a large input.
        safe_texts, batches, truncated = await loop.run_in_executor(self._executor, self._prepare_batches, texts)

        results: list[str | None] = [None] * len(texts)
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        async def process_batch(batch_no: int, group: list[int]) -> Usage:
            async with semaphore:
                batch = [safe_texts[i] for i in group]
                logger.debug(f"Processing batch {batch_no}/{len(batches)} ({len(batch)} texts)")

                response = await self._create_with_retry(batch, model_settings, batch_no, len(batches))
                batch_embeddings = extract_ordered_embeddings(response.data)

                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        f"Provider returned {len(batch_embeddings)} embeddings "
                        f"for {len(batch)} inputs in batch "
                        f"{batch_no}/{len(batches)}"
                    )

                for index, embedding in zip(group, batch_embeddings, strict=True):
                    results[index] = embedding

                # Shielded: this batch has been billed. If a sibling fails
                # while the write is still queued in the executor, cancelling
                # the await would cancel the queued job too and lose work the
                # caller already paid for.
                await asyncio.shield(self._cache_set_many([(keys[i], results[i]) for i in group]))  # type: ignore[misc]
                return await self._resolve_usage_async(response, batch)

        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(process_batch(batch_no, group)) for batch_no, group in enumerate(batches, start=1)
                ]
        except BaseExceptionGroup as eg:
            # TaskGroup wraps failures in a group; callers expect the original
            # provider error (RateLimitError and friends), so unwrap the first.
            primary, *also_failed = eg.exceptions
            for other in also_failed:
                # Only one exception can be raised. Logging the rest keeps a
                # second, differently-caused failure from vanishing entirely —
                # `from None` below hides it from the traceback too.
                logger.error(f"Additional batch failure: {other!r}")
            while isinstance(primary, BaseExceptionGroup):
                # A batch's own transport may itself raise a group. Keep
                # unwrapping so the caller gets a concrete error their
                # `except openai.RateLimitError` can actually match.
                primary = primary.exceptions[0]
            raise primary from None

        total_input_tokens = sum(task.result().input_tokens for task in tasks)
        total_tokens = sum(task.result().total_tokens for task in tasks)

        return typing.cast(list[str], results), Usage(
            input_tokens=total_input_tokens,
            total_tokens=total_tokens,
            truncated_texts=truncated,
        )

    async def get_embeddings(
        self,
        input: str | list[str],
        model_settings: ModelSettings,
    ) -> ModelResponse:
        """
        Generate embeddings asynchronously with caching and concurrent processing.
        Processes multiple texts concurrently for improved performance.

        Repeated texts are embedded once and the result shared, so passing the
        same string several times costs one provider call, not several.

        Args:
            input: Single string or list of strings to embed
            model_settings: Model configuration including dimensions and timeout

        Returns:
            ModelResponse containing embeddings and usage statistics
        """
        start_time = time.time()

        _input = validate_input(input)
        model_settings.validate_for_model(self.model)

        logger.debug(f"Processing {len(_input)} texts for embedding (async)")

        unique_texts, slots = deduplicate_texts(_input)
        keys = self._cache_keys_for(unique_texts, model_settings)
        resolved: list[str | None] = await self._cache_get_many(keys, model_settings.dimensions)

        missing = [slot for slot, value in enumerate(resolved) if value is None]
        # Counted per input item, so a text repeated twice and served from
        # cache still reports two hits.
        cache_hits = sum(1 for slot in slots if resolved[slot] is not None)

        if self._cache is not None and _input:
            logger.debug(f"Cache hit rate: {cache_hits / len(_input):.2%}, Processing {len(missing)} new embeddings")

        usage = Usage()
        if missing:
            try:
                embeddings, usage = await self._embed_missing(
                    [unique_texts[slot] for slot in missing],
                    [keys[slot] for slot in missing],
                    model_settings,
                )
            except Exception as e:
                logger.error(f"Failed to process embeddings: {e!s}")
                raise

            for slot, embedding in zip(missing, embeddings, strict=True):
                resolved[slot] = embedding

        _output = [resolved[slot] for slot in slots]
        if any(item is None for item in _output):
            raise RuntimeError("Failed to generate embeddings for some inputs")

        elapsed_time = time.time() - start_time
        logger.debug(f"Embeddings generated in {elapsed_time:.3f}s ({len(_input)} texts, {len(missing)} embedded)")

        return ModelResponse.model_validate(
            {
                "output": _output,
                "usage": Usage(
                    input_tokens=int(usage.input_tokens),
                    total_tokens=int(usage.total_tokens),
                    cache_hits=int(cache_hits),
                    truncated_texts=int(usage.truncated_texts),
                ),
            }
        )

    async def get_embeddings_generator(
        self,
        input: list[str],
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
        logger.debug(f"Processing {len(validated_input)} texts in {total_chunks} chunks of size {chunk_size}")

        for i in range(0, len(validated_input), chunk_size):
            chunk = validated_input[i : i + chunk_size]
            logger.debug(f"Processing chunk {i // chunk_size + 1}/{total_chunks}")
            yield await self.get_embeddings(chunk, model_settings)

    async def get_similarity(
        self,
        query: str,
        documents: list[str],
        *,
        model_settings: ModelSettings,
    ) -> SimilarityResponse:
        if not documents:
            raise ValueError("documents is required")
        if not query:
            raise ValueError("query is required")

        # `+` deliberately, not `[query, *documents]`: concatenation rejects a
        # bare `str` for `documents`, where unpacking would silently spread it
        # into single characters and rank letters.
        embeddings_res = await self.get_embeddings([query] + documents, model_settings)  # noqa: RUF005

        return embeddings_res.as_similarity_response()
