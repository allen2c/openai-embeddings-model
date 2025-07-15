import diskcache
import openai
import pytest
from faker import Faker

from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

fake = Faker()


@pytest.mark.asyncio
async def test_openai_embeddings(
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
async def test_gemini_embeddings(
    gemini_client_async: openai.AsyncOpenAI, cache: diskcache.Cache
):
    dim = 512
    sentence = fake.sentence()

    model = AsyncOpenAIEmbeddingsModel(
        model="gemini-embedding-001",
        openai_client=gemini_client_async,
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
