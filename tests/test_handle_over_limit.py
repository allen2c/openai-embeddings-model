import base64
from unittest.mock import Mock

import diskcache
import numpy as np
import openai
import pytest
from faker import Faker

from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
    ModelSettings,
    OpenAIEmbeddingsModel,
)

fake = Faker()


def test_handle_over_limit(
    openai_client: openai.OpenAI, cache: diskcache.Cache, monkeypatch
):
    """Test token limit handling with raise and truncate policies."""

    # Monkey patch to avoid API calls
    monkeypatch.setattr(openai_client.embeddings, "create", _mock_embeddings_create)

    # Generate text that exceeds token limits
    long_text = _create_long_text()

    # Test "raise" policy
    model_raise = OpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client,
        cache=cache,
        token_limit_policy="raise",
    )

    with pytest.raises(ValueError, match="Token limit exceeded"):
        model_raise.get_embeddings(
            input=long_text,
            model_settings=ModelSettings(dimensions=512),
        )

    # Test "truncate" policy
    model_truncate = OpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client,
        cache=cache,
        token_limit_policy="truncate",
    )

    # This should succeed by truncating the text
    result = model_truncate.get_embeddings(
        input=long_text,
        model_settings=ModelSettings(dimensions=512),
    )

    assert result.to_numpy().shape == (1, 512)
    assert result.usage.input_tokens > 0


@pytest.mark.asyncio
async def test_handle_over_limit_async(
    openai_client_async: openai.AsyncOpenAI, cache: diskcache.Cache, monkeypatch
):
    """Test async token limit handling with raise and truncate policies."""

    async def async_mock_embeddings_create(input, model, **kwargs):
        """Async version of mock embeddings.create."""
        return _mock_embeddings_create(input, model, **kwargs)

    # Monkey patch to avoid API calls
    monkeypatch.setattr(
        openai_client_async.embeddings, "create", async_mock_embeddings_create
    )

    # Generate text that exceeds token limits
    long_text = _create_long_text()

    # Test "raise" policy
    model_raise = AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client_async,
        cache=cache,
        token_limit_policy="raise",
    )

    with pytest.raises(ValueError, match="Token limit exceeded"):
        await model_raise.get_embeddings(
            input=long_text,
            model_settings=ModelSettings(dimensions=512),
        )

    # Test "truncate" policy
    model_truncate = AsyncOpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai_client_async,
        cache=cache,
        token_limit_policy="truncate",
    )

    # This should succeed by truncating the text
    result = await model_truncate.get_embeddings(
        input=long_text,
        model_settings=ModelSettings(dimensions=512),
    )

    assert result.to_numpy().shape == (1, 512)
    assert result.usage.input_tokens > 0


def _create_long_text() -> str:
    """Generate fake text that exceeds token limits."""
    # Generate a very long text that definitely exceeds the ~6962 token limit
    # Using repeated long sentences to ensure high token count
    sentences = []
    for _ in range(200):  # Generate many sentences
        sentences.append(fake.text(max_nb_chars=500))  # Long sentences
    return " ".join(sentences)


def _mock_embeddings_create(input, model, **kwargs):
    """Mock OpenAI embeddings.create that validates token limits."""
    # Get dimensions from kwargs, default to 512
    dimensions = kwargs.get("dimensions", 512)

    # Mock response structure
    mock_response = Mock()
    mock_response.data = []
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.total_tokens = 100

    # Generate fake base64 embeddings for each input
    input_texts = input if isinstance(input, list) else [input]
    for text in input_texts:
        mock_data = Mock()
        # Generate a fake embedding array with correct dimensions
        fake_embedding = np.random.random(dimensions).astype(np.float32)
        mock_data.embedding = base64.b64encode(fake_embedding.tobytes()).decode("utf-8")
        mock_response.data.append(mock_data)

    return mock_response
