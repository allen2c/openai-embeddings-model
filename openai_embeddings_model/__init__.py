import typing

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
    ):
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

        return response
