import logging
import os
import sys
import tempfile

import diskcache
import openai
import pytest
from str_or_none import str_or_none

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


@pytest.fixture(scope="module")
def cache():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield diskcache.Cache(temp_dir)


@pytest.fixture(scope="module")
def openai_client():
    return openai.OpenAI()


@pytest.fixture(scope="module")
def openai_client_async():
    return openai.AsyncOpenAI()


@pytest.fixture(scope="module")
def gemini_client_async():
    GEMINI_API_KEY = str_or_none(os.getenv("GEMINI_API_KEY"))
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    return openai.AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY,
    )
