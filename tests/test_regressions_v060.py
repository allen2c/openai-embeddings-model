"""Offline regression tests for the behaviour changes in 0.6.0.

Same rules as `test_regressions.py`: no network, no API key. A real client is
constructed so the library's isinstance checks pass, then `embeddings.create`
is replaced with a fake. Each test fails on 0.5.2 and passes on 0.6.0.
"""

import asyncio
import os
import signal
import threading
import time
import traceback
import typing

import diskcache
import httpx
import openai
import pydantic
import pytest
import tiktoken
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage as OpenAIUsage

from openai_embeddings_model import (
    CACHE_KEY_VERSION,
    AsyncOpenAIEmbeddingsModel,
    ModelResponse,
    ModelSettings,
    OpenAIEmbeddingsModel,
    Usage,
    cache_scope_digest,
    deduplicate_texts,
    generate_cache_key,
    py_float_list_to_b64_np32_array,
    truncate_text,
)

MODEL = "text-embedding-3-small"
DIM3 = [1.0, 0.0, 0.0]


def build_response(n: int, *, usage: OpenAIUsage | None = None):
    return CreateEmbeddingResponse.model_construct(
        data=[
            Embedding.model_construct(
                embedding=py_float_list_to_b64_np32_array(DIM3),
                index=i,
                object="embedding",
            )
            for i in range(n)
        ],
        model=MODEL,
        object="list",
        usage=usage or OpenAIUsage(prompt_tokens=n, total_tokens=n),
    )


def rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(
        429, request=httpx.Request("POST", "http://localhost:1/embeddings")
    )
    return openai.RateLimitError("slow down", response=response, body=None)


def sync_model(create, *, base_url="http://localhost:1", **kwargs):
    client = openai.OpenAI(api_key="test", base_url=base_url)
    client.embeddings.create = create  # type: ignore[method-assign]
    return OpenAIEmbeddingsModel(model=MODEL, openai_client=client, **kwargs)


def async_model(create, *, base_url="http://localhost:1", **kwargs):
    client = openai.AsyncOpenAI(api_key="test", base_url=base_url)
    client.embeddings.create = create  # type: ignore[method-assign]
    return AsyncOpenAIEmbeddingsModel(model=MODEL, openai_client=client, **kwargs)


def recording_create(calls: list):
    def create(input, **kwargs):
        calls.append(list(input))
        return build_response(len(input))

    return create


# --- cache key scope (was: providers and extra_body collided) ---


def test_cache_key_is_versioned():
    key = generate_cache_key(model=MODEL, dimensions=512, text="hi")
    assert key.startswith(f"{CACHE_KEY_VERSION}:")


def test_extra_body_changes_the_cache_key():
    a = generate_cache_key(
        model="voyage-3", text="hi", extra_body={"output_dimension": 512}
    )
    b = generate_cache_key(
        model="voyage-3", text="hi", extra_body={"output_dimension": 1024}
    )

    assert a != b


def test_provider_changes_the_cache_key():
    a = generate_cache_key(
        model=MODEL, text="hi", provider="https://api.openai.com/v1/"
    )
    b = generate_cache_key(model=MODEL, text="hi", provider="https://example.test/v1/")

    assert a != b


def test_same_scope_produces_the_same_key():
    def key() -> str:
        return generate_cache_key(
            model=MODEL, text="hi", provider="https://a/", extra_body={"b": 1}
        )

    assert key() == key()


def test_scope_digest_is_order_insensitive():
    a = cache_scope_digest("https://a/", {"x": 1, "y": 2})
    b = cache_scope_digest("https://a/", {"y": 2, "x": 1})

    assert a == b


def test_scope_digest_survives_unserialisable_values():
    """Must not raise: the digest only needs to be stable, not reversible."""
    assert cache_scope_digest("https://a/", {"when": object()})


def test_two_providers_sharing_a_cache_do_not_share_vectors(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "shared"))
    calls_a: list = []
    calls_b: list = []

    model_a = sync_model(
        recording_create(calls_a), base_url="http://provider-a:1", cache=cache
    )
    model_b = sync_model(
        recording_create(calls_b), base_url="http://provider-b:1", cache=cache
    )

    settings = ModelSettings(dimensions=3)
    model_a.get_embeddings(["shared text"], model_settings=settings)
    model_b.get_embeddings(["shared text"], model_settings=settings)

    assert calls_a == [["shared text"]]
    assert calls_b == [["shared text"]], "provider B must not reuse provider A's vector"


def test_differing_extra_body_does_not_share_vectors(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "shared"))
    calls: list = []
    model = sync_model(recording_create(calls), cache=cache)

    model.get_embeddings(
        ["t"], model_settings=ModelSettings(dimensions=3, extra_body={"task": "query"})
    )
    model.get_embeddings(
        ["t"],
        model_settings=ModelSettings(dimensions=3, extra_body={"task": "document"}),
    )

    assert len(calls) == 2


def test_lone_surrogate_text_does_not_crash_key_generation():
    assert generate_cache_key(model=MODEL, text="bad \ud800 text")


# --- validate_for_model (was: rejected nothing, ever) ---


@pytest.mark.parametrize(
    "model, dimensions",
    [
        ("text-embedding-3-small", 99999),
        ("text-embedding-3-large", -5),
        ("text-embedding-3-small", 0),
        ("text-embedding-ada-002", 512),
    ],
)
def test_invalid_dimensions_are_rejected(model, dimensions):
    with pytest.raises(ValueError):
        ModelSettings(dimensions=dimensions).validate_for_model(model)


@pytest.mark.parametrize(
    "model, dimensions",
    [
        ("text-embedding-3-small", 512),
        ("text-embedding-3-large", 3072),
        ("text-embedding-ada-002", None),
        ("some-unknown-model", 4096),  # unknown: only the provider can judge
    ],
)
def test_valid_dimensions_are_accepted(model, dimensions):
    ModelSettings(dimensions=dimensions).validate_for_model(model)


def test_get_embeddings_rejects_bad_dimensions_before_calling_the_provider():
    calls: list = []
    model = sync_model(recording_create(calls))

    with pytest.raises(ValueError):
        model.get_embeddings(["hi"], model_settings=ModelSettings(dimensions=99999))

    assert calls == []


# --- token limit settings (was: 0 sent empty strings, negatives reversed) ---


@pytest.mark.parametrize("percent", [0, -10, 101, 1000])
def test_invalid_token_limit_percent_is_rejected(percent):
    with pytest.raises(ValueError):
        sync_model(recording_create([]), token_limit_usage_percent=percent)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch_size": 0},
        {"max_input_tokens": 0},
        {"max_tokens_a_request": 0},
        {"max_retries": -1},
    ],
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sync_model(recording_create([]), **kwargs)


def test_truncate_text_clamps_negative_budgets():
    encoding = tiktoken.encoding_for_model("gpt-4o")
    text = "word " * 500

    assert truncate_text(text, encoding, -100) == ""
    assert truncate_text(text, encoding, 0) == ""


# --- per-request token budget (was: MAX_TOKENS_A_REQUEST unenforced) ---


def test_batches_respect_the_token_budget():
    calls: list = []
    model = sync_model(
        recording_create(calls),
        max_tokens_a_request=100,
        token_limit_policy="ignore",
    )

    # Distinct, or deduplication collapses them before batching is reached.
    texts = [f"t{i} " + " ".join(["word"] * 40) for i in range(10)]
    model.get_embeddings(texts, model_settings=ModelSettings())

    assert len(calls) > 1, "a 400-token input must not go out as one request"
    encoding = tiktoken.encoding_for_model("gpt-4o")
    for batch in calls:
        assert sum(len(encoding.encode(t)) for t in batch) <= 100


def test_a_single_oversized_text_still_goes_out_alone():
    """It cannot be split further, so it must not deadlock the batcher."""
    calls: list = []
    model = sync_model(
        recording_create(calls), max_tokens_a_request=10, token_limit_policy="ignore"
    )

    model.get_embeddings([" ".join(["word"] * 100)], model_settings=ModelSettings())

    assert len(calls) == 1


def test_item_count_limit_still_applies():
    calls: list = []
    model = sync_model(recording_create(calls), max_batch_size=3)

    model.get_embeddings([f"t{i}" for i in range(10)], model_settings=ModelSettings())

    assert [len(b) for b in calls] == [3, 3, 3, 1]


# --- truncation is now reported ---


def test_truncation_is_reported_in_usage():
    model = sync_model(
        recording_create([]), max_input_tokens=20, token_limit_usage_percent=50
    )

    res = model.get_embeddings(
        [" ".join(["word"] * 200), "short"], model_settings=ModelSettings()
    )

    assert res.usage.truncated_texts == 1


def test_no_truncation_reports_zero():
    model = sync_model(recording_create([]))

    res = model.get_embeddings(["short text"], model_settings=ModelSettings())

    assert res.usage.truncated_texts == 0


# --- per-batch persistence and retry (was: billed work discarded) ---


def test_successful_batches_are_cached_even_when_a_later_batch_fails(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "c"))
    seen: list = []

    def create(input, **kwargs):
        seen.append(list(input))
        if len(seen) == 2:
            raise RuntimeError("batch 2 exploded")
        return build_response(len(input))

    model = sync_model(create, cache=cache, max_batch_size=2)

    with pytest.raises(RuntimeError):
        model.get_embeddings(
            ["a", "b", "c", "d"], model_settings=ModelSettings(dimensions=3)
        )

    assert len(cache) == 2, "batch 1 was paid for and must survive the failure"


def test_rate_limit_is_retried_with_backoff():
    attempts: list = []

    def create(input, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise rate_limit_error()
        return build_response(len(input))

    model = sync_model(create, max_retries=3, retry_base_delay=0.001)
    res = model.get_embeddings(["a"], model_settings=ModelSettings())

    assert len(attempts) == 3
    assert res.to_numpy().shape == (1, 3)


def test_retries_are_bounded():
    attempts: list = []

    def create(input, **kwargs):
        attempts.append(1)
        raise rate_limit_error()

    model = sync_model(create, max_retries=2, retry_base_delay=0.001)

    with pytest.raises(openai.RateLimitError):
        model.get_embeddings(["a"], model_settings=ModelSettings())

    assert len(attempts) == 3, "one initial attempt plus max_retries"


def test_non_transient_errors_are_not_retried():
    attempts: list = []

    def create(input, **kwargs):
        attempts.append(1)
        raise ValueError("bad request")

    model = sync_model(create, max_retries=5, retry_base_delay=0.001)

    with pytest.raises(ValueError):
        model.get_embeddings(["a"], model_settings=ModelSettings())

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_rate_limit_is_retried_on_the_async_path():
    attempts: list = []

    async def create(input, **kwargs):
        attempts.append(1)
        if len(attempts) < 2:
            raise rate_limit_error()
        return build_response(len(input))

    async with async_model(create, max_retries=3, retry_base_delay=0.001) as model:
        res = await model.get_embeddings(["a"], model_settings=ModelSettings())

    assert len(attempts) == 2
    assert res.to_numpy().shape == (1, 3)


# --- tokenizer selection ---


def test_explicit_encoding_wins_over_auto_detection():
    encoding = tiktoken.get_encoding("o200k_base")
    model = sync_model(recording_create([]), encoding=encoding)

    assert model._encoding.name == "o200k_base"


def test_unknown_model_type_does_not_raise():
    client = openai.OpenAI(api_key="test", base_url="http://localhost:1")

    weird_model: typing.Any = object()
    model = OpenAIEmbeddingsModel(model=weird_model, openai_client=client)

    assert model._encoding is not None


# --- _build_extra_kwargs ---


def test_voyage_prefix_routes_dimensions_to_extra_body():
    model = sync_model(recording_create([]))
    model.model = "voyage-3"
    model._model_str = "voyage-3"
    model._dimensions_parameter = "output_dimension"

    kwargs = model._build_extra_kwargs(ModelSettings(dimensions=512))

    assert kwargs == {"extra_body": {"output_dimension": 512}}


def test_a_model_merely_containing_voyage_is_not_treated_as_voyage():
    client = openai.OpenAI(api_key="test", base_url="http://localhost:1")
    model = OpenAIEmbeddingsModel(model="my-voyage-clone", openai_client=client)

    kwargs = model._build_extra_kwargs(ModelSettings(dimensions=512))

    assert kwargs["dimensions"] == 512
    assert "extra_body" not in kwargs


def test_dimensions_parameter_can_be_set_explicitly():
    """Deployment aliases hide the underlying model, so auto-detection cannot help."""
    client = openai.OpenAI(api_key="test", base_url="http://localhost:1")
    model = OpenAIEmbeddingsModel(
        model="prod-embed-eu",
        openai_client=client,
        dimensions_parameter="output_dimension",
    )

    kwargs = model._build_extra_kwargs(ModelSettings(dimensions=256))

    assert kwargs == {"extra_body": {"output_dimension": 256}}


def test_unserialisable_extra_body_raises_a_clear_error():
    model = sync_model(recording_create([]))

    with pytest.raises(ValueError, match="JSON-serialisable"):
        model._build_extra_kwargs(ModelSettings(extra_body={"when": object()}))


def test_extra_body_is_not_mutated_by_the_caller():
    model = sync_model(recording_create([]))
    original = {"task": "query"}

    kwargs = model._build_extra_kwargs(ModelSettings(extra_body=original))
    kwargs["extra_body"]["task"] = "changed"

    assert original == {"task": "query"}


# --- deduplication ---


def test_deduplicate_texts_maps_back_correctly():
    unique, slots = deduplicate_texts(["a", "b", "a", "c", "b"])

    assert unique == ["a", "b", "c"]
    assert slots == [0, 1, 0, 2, 1]
    assert [unique[s] for s in slots] == ["a", "b", "a", "c", "b"]


def test_repeated_texts_are_embedded_once():
    calls: list = []
    model = sync_model(recording_create(calls))

    res = model.get_embeddings(
        ["a", "b", "a", "a", "b"], model_settings=ModelSettings()
    )

    assert calls == [["a", "b"]], "each distinct text costs one slot, not one per copy"
    assert len(res.to_python()) == 5


def test_duplicates_still_map_to_the_right_vectors():
    def create(input, **kwargs):
        return CreateEmbeddingResponse.model_construct(
            data=[
                Embedding.model_construct(
                    embedding=py_float_list_to_b64_np32_array([float(i), 0.0, 0.0]),
                    index=i,
                    object="embedding",
                )
                for i, _ in enumerate(input)
            ],
            model=MODEL,
            object="list",
            usage=OpenAIUsage(prompt_tokens=1, total_tokens=1),
        )

    model = sync_model(create)
    res = model.get_embeddings(["x", "y", "x"], model_settings=ModelSettings())

    vectors = res.to_python()
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_cache_hits_are_counted_per_input_item(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "c"))
    model = sync_model(recording_create([]), cache=cache)
    settings = ModelSettings(dimensions=3)

    model.get_embeddings(["a"], model_settings=settings)
    res = model.get_embeddings(["a", "a", "a"], model_settings=settings)

    assert res.usage.cache_hits == 3


@pytest.mark.asyncio
async def test_repeated_texts_are_embedded_once_async():
    calls: list = []

    async def create(input, **kwargs):
        calls.append(list(input))
        return build_response(len(input))

    async with async_model(create) as model:
        res = await model.get_embeddings(
            ["a", "b", "a"], model_settings=ModelSettings()
        )

    assert calls == [["a", "b"]]
    assert len(res.to_python()) == 3


# --- async: work must leave the event loop ---


@pytest.mark.asyncio
async def test_token_preparation_runs_off_the_event_loop():
    loop_thread = threading.current_thread()
    ran_on: list = []
    original = AsyncOpenAIEmbeddingsModel._prepare_batches

    def spy(self, texts):
        ran_on.append(threading.current_thread())
        return original(self, texts)

    async def create(input, **kwargs):
        return build_response(len(input))

    model = async_model(create)
    try:
        model._prepare_batches = spy.__get__(model)  # type: ignore[method-assign]
        await model.get_embeddings(["a", "b"], model_settings=ModelSettings())
    finally:
        await model.aclose()

    assert ran_on and all(t is not loop_thread for t in ran_on)


@pytest.mark.asyncio
async def test_cache_reads_are_batched_into_one_executor_job(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "c"))
    submits: list = []

    async def create(input, **kwargs):
        return build_response(len(input))

    model = async_model(create, cache=cache)
    original_submit = model._executor.submit

    def counting_submit(fn, *a, **kw):
        submits.append(fn)
        return original_submit(fn, *a, **kw)

    try:
        model._executor.submit = counting_submit  # type: ignore[method-assign]
        await model.get_embeddings(
            [f"t{i}" for i in range(50)], model_settings=ModelSettings(dimensions=3)
        )
    finally:
        await model.aclose()

    assert (
        len(submits) < 10
    ), f"50 texts should not mean 50+ executor round trips, saw {len(submits)}"


@pytest.mark.asyncio
async def test_sibling_batches_are_cancelled_when_one_fails():
    started: list = []
    finished: list = []

    async def create(input, **kwargs):
        started.append(input[0])
        if input[0] == "a":
            await asyncio.sleep(0)
            raise RuntimeError("batch 1 exploded")
        await asyncio.sleep(0.5)
        finished.append(input[0])
        return build_response(len(input))

    model = async_model(create, max_batch_size=1, max_concurrent_batches=5)
    try:
        with pytest.raises(RuntimeError):
            await model.get_embeddings(["a", "b", "c"], model_settings=ModelSettings())

        # Long enough for the siblings to finish if they were left running.
        # Asserting immediately would pass either way, since they simply have
        # not reached their own completion yet.
        await asyncio.sleep(0.6)
    finally:
        await model.aclose()

    assert len(started) == 3, "all batches started"
    assert finished == [], "siblings must be cancelled, not left billing"


@pytest.mark.asyncio
async def test_the_original_provider_error_survives_the_taskgroup():
    """TaskGroup raises an ExceptionGroup; callers still expect the real error."""

    async def create(input, **kwargs):
        raise rate_limit_error()

    async with async_model(create, max_retries=0) as model:
        with pytest.raises(openai.RateLimitError):
            await model.get_embeddings(["a"], model_settings=ModelSettings())


@pytest.mark.asyncio
async def test_async_batches_respect_the_token_budget():
    calls: list = []

    async def create(input, **kwargs):
        calls.append(list(input))
        return build_response(len(input))

    async with async_model(
        create, max_tokens_a_request=100, token_limit_policy="ignore"
    ) as model:
        texts = [f"t{i} " + " ".join(["word"] * 40) for i in range(10)]
        await model.get_embeddings(texts, model_settings=ModelSettings())

    assert len(calls) > 1
    encoding = tiktoken.encoding_for_model("gpt-4o")
    for batch in calls:
        assert sum(len(encoding.encode(t)) for t in batch) <= 100


# --- ordering is preserved through dedup and batching ---


def test_output_order_matches_input_order_across_batches():
    def create(input, **kwargs):
        return CreateEmbeddingResponse.model_construct(
            data=[
                Embedding.model_construct(
                    embedding=py_float_list_to_b64_np32_array(
                        [float(int(t[1:])), 0.0, 0.0]
                    ),
                    index=i,
                    object="embedding",
                )
                for i, t in enumerate(input)
            ],
            model=MODEL,
            object="list",
            usage=OpenAIUsage(prompt_tokens=1, total_tokens=1),
        )

    model = sync_model(create, max_batch_size=3)
    texts = [f"t{i}" for i in range(10)]

    res = model.get_embeddings(texts, model_settings=ModelSettings())

    assert [row[0] for row in res.to_python()] == [float(i) for i in range(10)]


@pytest.mark.asyncio
async def test_output_order_survives_concurrent_batches():
    async def create(input, **kwargs):
        # Later batches finish first, so ordering cannot rely on completion.
        await asyncio.sleep(0.01 * (10 - int(input[0][1:])))
        return CreateEmbeddingResponse.model_construct(
            data=[
                Embedding.model_construct(
                    embedding=py_float_list_to_b64_np32_array(
                        [float(int(t[1:])), 0.0, 0.0]
                    ),
                    index=i,
                    object="embedding",
                )
                for i, t in enumerate(input)
            ],
            model=MODEL,
            object="list",
            usage=OpenAIUsage(prompt_tokens=1, total_tokens=1),
        )

    async with async_model(create, max_batch_size=1) as model:
        texts = [f"t{i}" for i in range(8)]
        res = await model.get_embeddings(texts, model_settings=ModelSettings())

    assert [row[0] for row in res.to_python()] == [float(i) for i in range(8)]


def test_typing_helper_is_exported():
    assert callable(deduplicate_texts)
    assert isinstance(typing.get_type_hints(deduplicate_texts), dict)


# --- ModelResponse immutability (was: cached array went stale) ---


def test_model_response_is_frozen():
    resp = ModelResponse(output=[py_float_list_to_b64_np32_array(DIM3)], usage=Usage())
    resp.to_numpy()  # populate the cached decode

    with pytest.raises(pydantic.ValidationError):
        resp.output = [py_float_list_to_b64_np32_array([9.0, 9.0, 9.0])]


def test_frozen_does_not_break_the_cached_decode():
    resp = ModelResponse(output=[py_float_list_to_b64_np32_array(DIM3)], usage=Usage())

    first = resp.to_numpy()
    second = resp.to_numpy()

    assert first is not second, "each call returns its own writable copy"
    assert (first == second).all()


# --- fork safety (was: child wedged on an inherited sqlite lock) ---


def run_in_fork(child_body, timeout: float = 10.0) -> int:
    """Run `child_body` in a forked child, returning its exit code.

    Times out rather than hanging the suite: before the fix the child blocks
    forever on resources the fork invalidated, and a hung test tells you far
    less than a failed one.
    """
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            child_body()
            code = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(code)  # skip pytest/atexit teardown in the child

    deadline = time.time() + timeout
    while time.time() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)

    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    raise AssertionError(f"forked child did not finish within {timeout}s")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX only")
def test_forked_child_gets_a_working_executor(tmp_path):
    """The gunicorn preload_app shape: build the model, then fork."""
    cache = diskcache.Cache(str(tmp_path / "c"))

    async def create(input, **kwargs):
        return build_response(len(input))

    model = async_model(create, cache=cache)
    parent_executor_id = id(model._executor)
    # Touch the cache so the parent holds an open sqlite connection at fork.
    cache.set("warm", "up")

    def child():
        assert id(model._executor) != parent_executor_id, "executor not rebuilt"
        # Proves the pool has live workers: the parent's did not survive fork.
        assert model._executor.submit(lambda: 21 * 2).result(timeout=5) == 42
        assert asyncio.run(
            model.get_embeddings(["hi"], model_settings=ModelSettings(dimensions=3))
        ).to_numpy().shape == (1, 3)

    assert run_in_fork(child) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX only")
def test_forked_child_can_still_use_the_cache(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "c"))
    calls: list = []
    model = sync_model(recording_create(calls), cache=cache)
    settings = ModelSettings(dimensions=3)

    model.get_embeddings(["parent text"], model_settings=settings)

    def child():
        # Reads a parent-written entry and writes a new one of its own.
        assert (
            model.get_embeddings(
                ["parent text"], model_settings=settings
            ).usage.cache_hits
            == 1
        )
        model.get_embeddings(["child text"], model_settings=settings)

    assert run_in_fork(child) == 0
