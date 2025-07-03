import base64
import functools
import typing

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
    ) -> None:
        self.model = model
        self._client = openai_client

    def get_embeddings(
        self,
        input: str | typing.List[str],
        model_settings: ModelSettings,
    ) -> ModelResponse:
        """
        Get embeddings for the input text.
        """

        _input = [input] if isinstance(input, str) else input

        response: "openai_emb_resp.CreateEmbeddingResponse" = (
            self._client.embeddings.create(
                input=_input,
                model=self.model,
                dimensions=(
                    openai.NOT_GIVEN
                    if model_settings.dimensions is None
                    else model_settings.dimensions
                ),
                encoding_format="base64",
            )
        )

        return ModelResponse.model_validate(
            {
                "output": [item.embedding for item in response.data],
                "usage": Usage(
                    input_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
            }
        )
