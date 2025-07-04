import diskcache
import openai
import pytest
from faker import Faker

from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
    OpenAIEmbeddingsModel,
)

fake = Faker()


def test_get_embeddings(openai_client: openai.OpenAI, cache: diskcache.Cache):
    dim = 512
    sentence = fake.sentence()

    model = OpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client,
        cache=cache,
    )
    res = model.get_embeddings(
        input=sentence,
        model_settings=ModelSettings(dimensions=dim),
    )
    assert res.to_numpy().shape == (1, dim)
    assert len(res.to_python()[0]) == dim
    assert res.usage.input_tokens > 0

    assert (
        model.get_embeddings(
            input=sentence,
            model_settings=ModelSettings(dimensions=dim),
        ).usage.cache_hits
        == 1
    )


def test_get_embeddings_in_batch(openai_client: openai.OpenAI, cache: diskcache.Cache):
    dim = 512
    sentences = [fake.sentence() for _ in range(5)]

    model = OpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client,
        cache=cache,
    )
    res = model.get_embeddings(
        input=sentences,
        model_settings=ModelSettings(dimensions=dim),
    )
    assert res.to_numpy().shape == (len(sentences), dim)
    assert len(res.to_python()) == len(sentences)
    assert len(res.to_python()[0]) == dim
    assert res.usage.input_tokens > 0

    assert model.get_embeddings(
        input=sentences,
        model_settings=ModelSettings(dimensions=dim),
    ).usage.cache_hits == len(sentences)


@pytest.mark.asyncio
async def test_get_embeddings_async(
    openai_client_async: openai.AsyncOpenAI, cache: diskcache.Cache
):
    dim = 512
    sentence = fake.sentence()

    model = AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client_async,
        cache=cache,
    )
    res = await model.get_embeddings(
        input=sentence,
        model_settings=ModelSettings(dimensions=dim),
    )
    assert res.to_numpy().shape == (1, dim)
    assert len(res.to_python()[0]) == dim
    assert res.usage.input_tokens > 0

    assert (
        await model.get_embeddings(
            input=sentence,
            model_settings=ModelSettings(dimensions=dim),
        )
    ).usage.cache_hits == 1


@pytest.mark.asyncio
async def test_get_embeddings_async_in_batch(
    openai_client_async: openai.AsyncOpenAI, cache: diskcache.Cache
):
    dim = 512
    sentences = [fake.sentence() for _ in range(5)]

    model = AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client_async,
        cache=cache,
    )
    res = await model.get_embeddings(
        input=sentences,
        model_settings=ModelSettings(dimensions=dim),
    )
    assert res.to_numpy().shape == (len(sentences), dim)
    assert len(res.to_python()) == len(sentences)
    assert len(res.to_python()[0]) == dim
    assert res.usage.input_tokens > 0

    assert (
        await model.get_embeddings(
            input=sentences,
            model_settings=ModelSettings(dimensions=dim),
        )
    ).usage.cache_hits == len(sentences)
