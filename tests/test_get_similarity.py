import diskcache
import openai
import pytest

from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
    OpenAIEmbeddingsModel,
)


def test_get_similarity(cache: diskcache.Cache, openai_client: openai.OpenAI):
    emb_model = OpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client,
        cache=cache,
    )

    query = "What is the capital of France?"
    documents = [
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of France is Paris.",
    ]
    response = emb_model.get_similarity(
        query, documents, model_settings=ModelSettings(dimensions=512)
    )
    assert response.usage.cache_hits > 0 or response.usage.total_tokens > 0
    assert response.results[0].relevance_score > 0.0
    assert all(
        response.results[0].relevance_score >= res.relevance_score
        for res in response.results
    )


@pytest.mark.asyncio
async def test_get_similarity_async(
    cache: diskcache.Cache, openai_client_async: openai.AsyncOpenAI
):
    emb_model = AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client_async,
        cache=cache,
    )

    query = "The Apple company is a technology company."
    documents = [
        "Youtube is a video sharing platform.",
        "New show released on Netflix.",
        "The capital of France is Paris.",
        "iOS 26 is the latest version of iPhone.",
    ]
    response = await emb_model.get_similarity(
        query, documents, model_settings=ModelSettings(dimensions=512)
    )
    assert response.usage.cache_hits > 0 or response.usage.total_tokens > 0
    assert response.results[0].relevance_score > 0.0
    assert all(
        response.results[0].relevance_score >= res.relevance_score
        for res in response.results
    )
