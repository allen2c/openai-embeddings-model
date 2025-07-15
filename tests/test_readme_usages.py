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


class TestBasicUsage:
    """Test basic usage examples from README."""

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


class TestSupportedProviders:
    """Test supported provider examples from README."""

    def test_openai_provider(self, openai_client: openai.OpenAI):
        """Test OpenAI provider example."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        response = model.get_embeddings(
            input="Test OpenAI provider", model_settings=ModelSettings(dimensions=512)
        )

        assert response.to_numpy().shape == (1, 512)
        assert response.usage.input_tokens > 0

    def test_gemini_provider(self):
        """Test Gemini provider example."""
        # Mock Gemini client
        gemini_client = openai.OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="test-gemini-key",
        )

        model = OpenAIEmbeddingsModel(
            model="text-embedding-004", openai_client=gemini_client
        )

        # Test model initialization
        assert model.model == "text-embedding-004"
        assert model._client == gemini_client

    def test_azure_openai_provider(self):
        """Test Azure OpenAI provider example."""
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_client = openai.OpenAI()
            mock_azure.return_value = mock_client

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

    def test_self_hosted_provider(self):
        """Test self-hosted provider example (Ollama)."""
        ollama_client = openai.OpenAI(
            base_url="http://localhost:11434/v1", api_key="ollama"
        )

        model = OpenAIEmbeddingsModel(
            model="nomic-embed-text", openai_client=ollama_client
        )

        assert model.model == "nomic-embed-text"
        assert model._client == ollama_client


class TestAdvancedFeatures:
    """Test advanced features from README."""

    def test_batch_processing(
        self, openai_client: openai.OpenAI, cache: diskcache.Cache
    ):
        """Test batch processing examples."""
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
        """Test custom caching examples."""
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
        """Test model configuration examples."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        settings = ModelSettings(dimensions=1024, timeout=30.0)

        response = model.get_embeddings(
            input="Test configuration", model_settings=settings
        )

        assert response.to_numpy().shape == (1, 1024)
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


class TestApiReference:
    """Test API reference examples from README."""

    def test_model_classes(
        self, openai_client: openai.OpenAI, openai_client_async: openai.AsyncOpenAI
    ):
        """Test model classes mentioned in API reference."""
        # Test synchronous model
        sync_model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )
        assert sync_model is not None

        # Test asynchronous model
        async_model = AsyncOpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client_async
        )
        assert async_model is not None

    def test_model_settings_configuration(self, openai_client: openai.OpenAI):
        """Test ModelSettings configuration options."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        # Test default settings
        default_settings = ModelSettings()
        assert default_settings.dimensions is None
        assert default_settings.timeout is None

        # Test custom settings
        custom_settings = ModelSettings(dimensions=1024, timeout=30.0)
        assert custom_settings.dimensions == 1024
        assert custom_settings.timeout == 30.0

        # Test with custom settings
        response = model.get_embeddings(
            input="Test settings", model_settings=custom_settings
        )
        assert response.to_numpy().shape == (1, 1024)

    def test_response_properties(self, openai_client: openai.OpenAI):
        """Test ModelResponse properties from API reference."""
        model = OpenAIEmbeddingsModel(
            model="text-embedding-3-small", openai_client=openai_client
        )

        response = model.get_embeddings(
            input="Test response", model_settings=ModelSettings(dimensions=512)
        )

        # Test response methods
        numpy_result = response.to_numpy()
        python_result = response.to_python()

        assert numpy_result.shape == (1, 512)
        assert len(python_result) == 1
        assert len(python_result[0]) == 512

        # Test usage properties
        assert hasattr(response.usage, "input_tokens")
        assert hasattr(response.usage, "total_tokens")
        assert hasattr(response.usage, "cache_hits")
        assert response.usage.input_tokens > 0
        assert response.usage.total_tokens >= response.usage.input_tokens
        assert response.usage.cache_hits >= 0
