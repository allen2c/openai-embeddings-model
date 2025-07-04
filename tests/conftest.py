import logging
import sys
import tempfile

import diskcache
import openai
import pytest

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
