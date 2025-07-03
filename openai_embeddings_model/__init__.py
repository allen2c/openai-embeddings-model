import base64
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

    def to_numpy(self) -> np.typing.NDArray[np.float32]:
        decoded_bytes = [base64.b64decode(s) for s in self.output]
        embedding_array = [
            np.frombuffer(decoded_byte, dtype=np.float32)
            for decoded_byte in decoded_bytes
        ]
        return np.array(embedding_array)

    def to_python(self) -> list[list[float]]:
        return [embedding.tolist() for embedding in self.to_numpy()]


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
