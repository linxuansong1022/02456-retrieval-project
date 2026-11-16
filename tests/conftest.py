import pytest

from tests.tokenizer_util import make_tiny_tokenizer


@pytest.fixture(scope="module")
def tiny_tokenizer():
    return make_tiny_tokenizer()
