"""
Test all usage examples from README.md to ensure they work correctly.
"""

import tempfile
from unittest.mock import patch

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


class TestSyncUsage:
    """Test synchronous usage examples from README."""

    def test_basic_sync_usage(
        self, openai_client: openai.OpenAI, cache: diskcache.Cache
    ):
        """Test basic synchronous usage example."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        response = model.get_embeddings(
            input="Hello, world!", model_settings=ModelSettings(dimensions=512)
        )

        # Test response structure
        embeddings = response.to_numpy()
        embeddings_list = response.to_python()

        assert embeddings.shape == (1, 512)
        assert len(embeddings_list) == 1
        assert len(embeddings_list[0]) == 512
        assert response.usage.input_tokens > 0
        assert response.usage.total_tokens > 0

    def test_batch_processing(
        self, openai_client: openai.OpenAI, cache: diskcache.Cache
    ):
        """Test batch processing and generator usage."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client, cache=cache
        )

        texts = ["Hello, world!", "How are you?", "This is a test"]

        # Test batch processing
        response = model.get_embeddings(
            input=texts, model_settings=ModelSettings(dimensions=512)
        )

        embeddings = response.to_numpy()
        assert embeddings.shape == (3, 512)
        assert response.usage.input_tokens > 0

        # Test generator usage
        chunks = list(
            model.get_embeddings_generator(
                input=texts, model_settings=ModelSettings(dimensions=512), chunk_size=2
            )
        )

        assert len(chunks) == 2  # 3 texts with chunk_size=2 -> 2 chunks
        assert chunks[0].to_numpy().shape == (2, 512)  # First chunk: 2 texts
        assert chunks[1].to_numpy().shape == (1, 512)  # Second chunk: 1 text

    def test_custom_caching(self, openai_client: openai.OpenAI):
        """Test custom cache and default cache usage."""
        # Test custom cache
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_cache = diskcache.Cache(temp_dir)

            model = OpenAIEmbeddingsModel(
                model="text-embedding-3-small",
                openai_client=openai_client,
                cache=custom_cache,
            )

            text = "Test caching"
            response1 = model.get_embeddings(
                input=text, model_settings=ModelSettings(dimensions=512)
            )
            assert response1.usage.cache_hits == 0

            # Second call should hit cache
            response2 = model.get_embeddings(
                input=text, model_settings=ModelSettings(dimensions=512)
            )
            assert response2.usage.cache_hits == 1

    def test_model_configuration(self, openai_client: openai.OpenAI):
        """Test ModelSettings configuration."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        # Test with custom dimensions and timeout
        settings = ModelSettings(dimensions=1024, timeout=30.0)

        response = model.get_embeddings(
            input="Test configuration", model_settings=settings
        )

        assert response.to_numpy().shape == (1, 1024)
        assert response.usage.input_tokens > 0


class TestAsyncUsage:
    """Test asynchronous usage examples from README."""

    @pytest.mark.asyncio
    async def test_basic_async_usage(self, openai_client_async: openai.AsyncOpenAI):
        """Test basic asynchronous usage example."""
        model = AsyncOpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client_async
        )

        response = await model.get_embeddings(
            input=["Hello, world!", "How are you?"],
            model_settings=ModelSettings(dimensions=512),
        )

        embeddings = response.to_numpy()
        assert embeddings.shape == (2, 512)
        assert response.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_async_generator(self, openai_client_async: openai.AsyncOpenAI):
        """Test async generator usage."""
        model = AsyncOpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client_async
        )

        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        chunks = []
        async for chunk_response in model.get_embeddings_generator(
            input=texts, model_settings=ModelSettings(dimensions=512), chunk_size=2
        ):
            chunks.append(chunk_response)

        assert len(chunks) == 3  # 5 texts with chunk_size=2 -> 3 chunks
        assert chunks[0].to_numpy().shape == (2, 512)
        assert chunks[1].to_numpy().shape == (2, 512)
        assert chunks[2].to_numpy().shape == (1, 512)


class TestClientVariations:
    """Test different OpenAI client configurations from README."""

    def test_azure_openai_client(self):
        """Test Azure OpenAI client usage."""
        # Mock Azure client creation
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_client = openai.OpenAI()  # Use regular client as mock
            mock_azure.return_value = mock_client

            # Azure client setup as shown in README
            azure_client = openai.AzureOpenAI(
                api_key="your-azure-api-key",
                api_version="2023-05-15",
                azure_endpoint="https://your-resource.openai.azure.com/",
            )

            model = OpenAIEmbeddingsModel(
                model="text-embedding-3-small", openai_client=azure_client
            )

            response = model.get_embeddings(
                input="Test Azure client", model_settings=ModelSettings(dimensions=512)
            )

            assert response.to_numpy().shape == (1, 512)

    def test_custom_base_url_client(self):
        """Test custom base URL client (Ollama, self-hosted)."""
        # Test Ollama-style client
        ollama_client = openai.OpenAI(
            base_url="http://localhost:11434/v1", api_key="ollama"
        )

        model = OpenAIEmbeddingsModel(
            model="nomic-embed-text", openai_client=ollama_client
        )

        # Test that model initializes correctly
        assert model.model == "nomic-embed-text"
        assert model._client == ollama_client

        # Test custom endpoint client
        custom_client = openai.OpenAI(
            base_url="https://custom-endpoint.com/v1", api_key="custom-key"
        )

        model = OpenAIEmbeddingsModel(model="custom-model", openai_client=custom_client)

        assert model.model == "custom-model"
        assert model._client == custom_client


class TestErrorHandling:
    """Test error handling examples from README."""

    def test_error_handling_patterns(self, openai_client: openai.OpenAI):
        """Test error handling as shown in README."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        # Test ValueError for invalid input
        with pytest.raises(ValueError):
            model.get_embeddings(
                input="",  # Empty string should raise ValueError
                model_settings=ModelSettings(dimensions=512),
            )

        # Test openai.BadRequestError for API errors (like invalid dimensions)
        with pytest.raises(openai.BadRequestError):
            model.get_embeddings(
                input="Valid text",
                model_settings=ModelSettings(dimensions=5000),  # Too large
            )

        # Test TypeError for invalid input type
        with pytest.raises(TypeError):
            model.get_embeddings(
                input=123,  # type: ignore  # Intentionally testing invalid type
                model_settings=ModelSettings(dimensions=512),
            )


class TestResponseUsage:
    """Test response usage patterns from README."""

    def test_response_methods(
        self, openai_client: openai.OpenAI, cache: diskcache.Cache
    ):
        """Test all response methods and properties."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client, cache=cache
        )

        response = model.get_embeddings(
            input=["First text", "Second text"],
            model_settings=ModelSettings(dimensions=512),
        )

        # Test to_numpy() method
        numpy_embeddings = response.to_numpy()
        assert numpy_embeddings.shape == (2, 512)

        # Test to_python() method
        python_embeddings = response.to_python()
        assert len(python_embeddings) == 2
        assert len(python_embeddings[0]) == 512
        assert isinstance(python_embeddings[0][0], float)

        # Test usage statistics
        assert response.usage.input_tokens > 0
        assert response.usage.total_tokens >= response.usage.input_tokens
        assert response.usage.cache_hits >= 0  # Should be 0 for first call

        # Test caching behavior
        response2 = model.get_embeddings(
            input=["First text", "Second text"],
            model_settings=ModelSettings(dimensions=512),
        )
        assert response2.usage.cache_hits == 2  # Should hit cache for both texts
