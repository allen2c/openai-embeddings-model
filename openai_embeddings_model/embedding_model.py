from typing_extensions import Literal, TypeAlias

__all__ = ["EmbeddingModel"]

EmbeddingModel: TypeAlias = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
