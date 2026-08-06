"""Offline regression tests for the bugs fixed in 0.5.2.

Every test here runs without network access or an API key: a real client is
constructed so the library's isinstance checks pass, then `embeddings.create`
is replaced with a fake. Each test fails on 0.5.1 and passes on 0.5.2.
"""

import base64
import math
import typing

import diskcache
import numpy as np
import openai
import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage as OpenAIUsage

from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelResponse,
    ModelSettings,
    OpenAIEmbeddingsModel,
    Usage,
    py_float_list_to_b64_np32_array,
    validate_cached_embedding,
)

MODEL = "text-embedding-3-small"


def build_response(
    vectors: typing.Sequence[typing.Sequence[float]],
    *,
    indices: typing.Sequence[int] | None = None,
    usage: OpenAIUsage | None = None,
) -> CreateEmbeddingResponse:
    """Build a response whose `data` order and `index` values are independent."""
    if indices is None:
        indices = list(range(len(vectors)))
    return CreateEmbeddingResponse.model_construct(
        data=[
            Embedding.model_construct(
                embedding=py_float_list_to_b64_np32_array(list(vec)),
                index=idx,
                object="embedding",
            )
            for vec, idx in zip(vectors, indices)
        ],
        model=MODEL,
        object="list",
        usage=usage,
    )


def sync_model(create, **kwargs) -> OpenAIEmbeddingsModel:
    client = openai.OpenAI(api_key="test", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]
    return OpenAIEmbeddingsModel(model=MODEL, openai_client=client, **kwargs)


def async_model(create, **kwargs) -> AsyncOpenAIEmbeddingsModel:
    client = openai.AsyncOpenAI(api_key="test", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]
    return AsyncOpenAIEmbeddingsModel(model=MODEL, openai_client=client, **kwargs)


def unit_vectors(n: int) -> typing.List[typing.List[float]]:
    """n distinct 3-d vectors, each a clean unit vector."""
    basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return [basis[i % 3] for i in range(n)]


# --- get_similarity with a single document (was: TypeError on a 0-d array) ---


def test_get_similarity_with_single_document_sync():
    def create(input, **kwargs):
        return build_response(
            unit_vectors(len(input)), usage=OpenAIUsage(prompt_tokens=2, total_tokens=2)
        )

    model = sync_model(create)
    res = model.get_similarity("q", ["only one doc"], model_settings=ModelSettings())

    assert len(res.results) == 1
    assert res.results[0].index == 0


@pytest.mark.asyncio
async def test_get_similarity_with_single_document_async():
    async def create(input, **kwargs):
        return build_response(
            unit_vectors(len(input)), usage=OpenAIUsage(prompt_tokens=2, total_tokens=2)
        )

    model = async_model(create)
    try:
        res = await model.get_similarity(
            "q", ["only one doc"], model_settings=ModelSettings()
        )
    finally:
        await model.aclose()

    assert len(res.results) == 1
    assert res.results[0].index == 0


def test_get_similarity_result_count_matches_documents():
    """Guard the general case too, not just the n=1 boundary."""

    def create(input, **kwargs):
        return build_response(
            unit_vectors(len(input)),
            usage=OpenAIUsage(prompt_tokens=len(input), total_tokens=len(input)),
        )

    model = sync_model(create)
    for n_docs in (1, 2, 3, 7):
        res = model.get_similarity(
            "q", [f"doc-{i}" for i in range(n_docs)], model_settings=ModelSettings()
        )
        assert len(res.results) == n_docs
        assert sorted(r.index for r in res.results) == list(range(n_docs))


# --- provider returning data out of order (was: silent misalignment) ---


def test_out_of_order_response_is_realigned_by_index_sync():
    """Provider returns data reversed but labelled with correct `index`."""
    distinct = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]

    def create(input, **kwargs):
        return build_response(
            list(reversed(distinct)),
            indices=[2, 1, 0],
            usage=OpenAIUsage(prompt_tokens=3, total_tokens=3),
        )

    model = sync_model(create)
    res = model.get_embeddings(["a", "b", "c"], model_settings=ModelSettings())

    np.testing.assert_allclose(res.to_python(), distinct)


@pytest.mark.asyncio
async def test_out_of_order_response_is_realigned_by_index_async():
    distinct = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]

    async def create(input, **kwargs):
        return build_response(
            list(reversed(distinct)),
            indices=[2, 1, 0],
            usage=OpenAIUsage(prompt_tokens=3, total_tokens=3),
        )

    model = async_model(create)
    try:
        res = await model.get_embeddings(
            ["a", "b", "c"], model_settings=ModelSettings()
        )
    finally:
        await model.aclose()

    np.testing.assert_allclose(res.to_python(), distinct)


def test_response_without_index_field_keeps_arrival_order():
    """Providers that omit `index` must still work, falling back to order."""
    distinct = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    def create(input, **kwargs):
        resp = build_response(
            distinct, usage=OpenAIUsage(prompt_tokens=2, total_tokens=2)
        )
        for datum in resp.data:
            del datum.index
        return resp

    model = sync_model(create)
    res = model.get_embeddings(["a", "b"], model_settings=ModelSettings())

    np.testing.assert_allclose(res.to_python(), distinct)


# --- provider omitting usage (was: AttributeError on the sync path only) ---


def test_missing_usage_falls_back_to_tiktoken_sync():
    def create(input, **kwargs):
        return build_response(unit_vectors(len(input)), usage=None)

    model = sync_model(create)
    res = model.get_embeddings(["hello world"], model_settings=ModelSettings())

    assert res.usage.input_tokens > 0
    assert res.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_missing_usage_falls_back_to_tiktoken_async():
    async def create(input, **kwargs):
        return build_response(unit_vectors(len(input)), usage=None)

    model = async_model(create)
    try:
        res = await model.get_embeddings(
            ["hello world"], model_settings=ModelSettings()
        )
    finally:
        await model.aclose()

    assert res.usage.input_tokens > 0
    assert res.usage.total_tokens > 0


def test_sync_and_async_report_identical_usage_without_provider_usage():
    """The two paths must not diverge in how they account for tokens."""
    texts = ["alpha beta", "gamma delta epsilon"]

    def create(input, **kwargs):
        return build_response(unit_vectors(len(input)), usage=None)

    async def acreate(input, **kwargs):
        return build_response(unit_vectors(len(input)), usage=None)

    import asyncio

    sync_res = sync_model(create).get_embeddings(texts, model_settings=ModelSettings())

    async def run_async():
        model = async_model(acreate)
        try:
            return await model.get_embeddings(texts, model_settings=ModelSettings())
        finally:
            await model.aclose()

    async_res = asyncio.run(run_async())

    assert sync_res.usage.input_tokens == async_res.usage.input_tokens
    assert sync_res.usage.total_tokens == async_res.usage.total_tokens


# --- NaN scores (was: NaN sorted to the top as the best match) ---


def test_nan_relevance_score_never_ranks_above_a_real_match():
    query = [1.0, 0.0, 0.0]
    docs = [
        [float("nan"), 0.0, 0.0],  # index 0 -> NaN score
        [1.0, 0.0, 0.0],  # index 1 -> perfect match
        [float("nan"), 1.0, 0.0],  # index 2 -> NaN score
    ]
    resp = ModelResponse(
        output=[py_float_list_to_b64_np32_array(v) for v in [query] + docs],
        usage=Usage(),
    )

    results = resp.as_similarity_response().results

    assert results[0].index == 1
    assert not math.isnan(results[0].relevance_score)
    assert all(math.isnan(r.relevance_score) for r in results[-2:])


# --- to_numpy() writability (was: read-only view into the cached buffer) ---


def test_to_numpy_returns_a_writable_array():
    resp = ModelResponse(
        output=[py_float_list_to_b64_np32_array([1.0, 2.0, 3.0])], usage=Usage()
    )

    arr = resp.to_numpy()

    assert arr.flags.writeable
    arr += 1.0  # must not raise


def test_mutating_to_numpy_result_does_not_corrupt_the_response():
    """A caller (or faiss, writing through a raw pointer) must not poison the cache."""
    resp = ModelResponse(
        output=[py_float_list_to_b64_np32_array([3.0, 4.0, 0.0])], usage=Usage()
    )

    first = resp.to_numpy()
    first[:] = 0.0  # simulate an in-place normalisation by a native library

    np.testing.assert_allclose(resp.to_numpy(), [[3.0, 4.0, 0.0]])
    np.testing.assert_allclose(resp.to_python(), [[3.0, 4.0, 0.0]])


# --- corrupt cache entries (was: str() coercion yielding silent garbage) ---


@pytest.mark.parametrize(
    "bad_value, note",
    [
        (1234567890123456, "16-digit int decodes as valid base64 garbage"),
        (b"\x00\x00\x80?", "bytes stringify to a b'...' literal"),
        (b"", "empty bytes are not None but carry no vector"),
        ("not base64 at all!!", "undecodable payload"),
        ("QUJD", "decodes to 3 bytes, not a multiple of 4"),
    ],
)
def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path, bad_value, note):
    cache = diskcache.Cache(str(tmp_path / "cache"))
    calls: typing.List[int] = []

    def create(input, **kwargs):
        calls.append(len(input))
        return build_response(
            [[1.0, 0.0, 0.0]] * len(input),
            usage=OpenAIUsage(prompt_tokens=1, total_tokens=1),
        )

    model = sync_model(create, cache=cache)
    settings = ModelSettings(dimensions=3)

    from openai_embeddings_model import generate_cache_key

    key = generate_cache_key(model=MODEL, dimensions=3, text="hello")
    cache.set(key, bad_value)

    res = model.get_embeddings(["hello"], model_settings=settings)

    assert calls == [1], f"expected a provider call ({note})"
    assert res.usage.cache_hits == 0
    np.testing.assert_allclose(res.to_python(), [[1.0, 0.0, 0.0]])


def test_cache_entry_with_wrong_dimensions_is_rejected():
    good_512 = py_float_list_to_b64_np32_array([0.1] * 512)

    assert validate_cached_embedding("k", good_512, 512) == good_512
    assert validate_cached_embedding("k", good_512, 1024) is None


def test_cache_entry_containing_nan_is_rejected():
    payload = py_float_list_to_b64_np32_array([float("nan"), 1.0, 2.0])

    assert validate_cached_embedding("k", payload, 3) is None


def test_valid_cache_entry_still_hits(tmp_path):
    cache = diskcache.Cache(str(tmp_path / "cache"))
    calls: typing.List[int] = []

    def create(input, **kwargs):
        calls.append(len(input))
        return build_response(
            [[1.0, 0.0, 0.0]] * len(input),
            usage=OpenAIUsage(prompt_tokens=1, total_tokens=1),
        )

    model = sync_model(create, cache=cache)
    settings = ModelSettings(dimensions=3)

    model.get_embeddings(["hello"], model_settings=settings)
    res = model.get_embeddings(["hello"], model_settings=settings)

    assert calls == [1], "second call must be served from cache"
    assert res.usage.cache_hits == 1


def test_base64_roundtrip_survives_validation():
    """Whatever the library writes, it must be able to read back."""
    vec = np.random.default_rng(0).normal(size=256).astype(np.float32)
    payload = base64.b64encode(vec.tobytes()).decode()

    assert validate_cached_embedding("k", payload, 256) == payload


# --- executor lifecycle (was: never shut down, leaking worker threads) ---


@pytest.mark.asyncio
async def test_aclose_shuts_down_the_executor():
    model = async_model(lambda **kw: None)

    await model.aclose()

    assert model._executor._shutdown is True


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    model = async_model(lambda **kw: None)

    await model.aclose()
    await model.aclose()

    assert model._executor._shutdown is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_the_executor():
    async def create(input, **kwargs):
        return build_response(
            unit_vectors(len(input)),
            usage=OpenAIUsage(prompt_tokens=1, total_tokens=1),
        )

    client = openai.AsyncOpenAI(api_key="test", base_url="http://localhost:1")
    client.embeddings.create = create  # type: ignore[method-assign]

    async with AsyncOpenAIEmbeddingsModel(model=MODEL, openai_client=client) as model:
        res = await model.get_embeddings(["hi"], model_settings=ModelSettings())
        assert res.to_numpy().shape[0] == 1
        assert model._executor._shutdown is False

    assert model._executor._shutdown is True


def test_dropped_model_releases_workers_even_if_the_pool_is_referenced():
    """Dropping the model must release its workers.

    CPython already reaps a ThreadPoolExecutor's workers once the executor
    itself becomes unreachable, so a model that is simply dropped was never
    the real problem. The leak shows up when something outlives the model
    still holding the pool — a metrics registry, a debug list, a closure. The
    model's own teardown, not the executor's refcount, has to end the threads.
    """
    import gc
    import threading

    baseline = threading.active_count()
    retained_pools = []

    for _ in range(20):
        model = async_model(lambda **kw: None)
        model._executor.submit(lambda: None).result()  # force a worker to spawn
        retained_pools.append(model._executor)
        del model

    gc.collect()

    assert threading.active_count() - baseline <= 2
    assert all(pool._shutdown for pool in retained_pools)
