import base64
import functools
import hashlib
import typing

import diskcache
import numpy as np
import openai
import openai.types.create_embedding_response as openai_emb_resp
import pydantic

from .embedding_model import EmbeddingModel

__all__ = ["ModelSettings", "OpenAIEmbeddingsModel"]


class ModelSettings(pydantic.BaseModel):
    dimensions: int | None = None
    timeout: float | None = None


class Usage(pydantic.BaseModel):
    input_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(pydantic.BaseModel):
    output: list[typing.Text]
    usage: Usage

    @functools.cached_property
    def _decoded_bytes(self) -> memoryview:
        """
        Lazily decode *all* embeddings in one pass and expose them
        as a zero-copy memoryview to avoid duplicating data.
        """
        return memoryview(b"".join(base64.b64decode(s) for s in self.output))

    @functools.cached_property
    def _ndarray(self) -> np.ndarray:
        """
        Materialize the NumPy array once and cache it.  Later calls to
        `to_numpy()` or `to_python()` return the cached view.
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
    def __init__(
        self,
        model: str | EmbeddingModel,
        openai_client: openai.OpenAI | openai.AzureOpenAI,
        cache: diskcache.Cache | None = None,
    ) -> None:
        self.model = model
        self._client = openai_client
        self._cache = cache

    def get_embeddings(
        self,
        input: str | typing.List[str],
        model_settings: ModelSettings,
    ) -> ModelResponse:
        """
        Get embeddings for the input text.
        """

        _input = [input] if isinstance(input, str) else input

        _output: typing.List[typing.Text | None] = [None] * len(_input)
        _missing_idx: typing.List[int] = []
        if self._cache is not None:
            for i, item in enumerate(_input):
                _cached_item = self._cache.get(
                    hashlib.sha256(item.encode()).hexdigest()
                )
                if _cached_item is None:
                    _missing_idx.append(i)
                else:
                    _output[i] = _cached_item  # type: ignore
        else:
            _missing_idx = list(range(len(_input)))

        response: openai_emb_resp.CreateEmbeddingResponse | None = None
        if len(_missing_idx) > 0:
            response = self._client.embeddings.create(
                input=[_input[i] for i in _missing_idx],
                model=self.model,
                dimensions=(
                    openai.NOT_GIVEN
                    if model_settings.dimensions is None
                    else model_settings.dimensions
                ),
                encoding_format="base64",
            )

        for i, item_idx in enumerate(_missing_idx):
            _embedding: str = response.data[i].embedding  # type: ignore
            if self._cache is not None:
                self._cache.set(
                    hashlib.sha256(_input[item_idx].encode()).hexdigest(),
                    _embedding,
                )
            _output[item_idx] = _embedding

        return ModelResponse.model_validate(
            {
                "output": _output,
                "usage": Usage(
                    input_tokens=response.usage.prompt_tokens if response else 0,
                    total_tokens=response.usage.total_tokens if response else 0,
                ),
            }
        )


def get_default_cache() -> diskcache.Cache:
    return diskcache.Cache(directory="./.cache/embeddings.cache")
