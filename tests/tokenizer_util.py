"""A download-free WordLevel tokenizer for tests."""
from __future__ import annotations

from tokenizers import Tokenizer as TokenizersTokenizer
from tokenizers import models
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

_BASE_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "hello": 3,
    "world": 4,
    "the": 5,
    "quick": 6,
    "brown": 7,
    "fox": 8,
    "jumps": 9,
    "over": 10,
    "lazy": 11,
    "dog": 12,
    "a": 13,
    "is": 14,
}


def make_tiny_tokenizer(extra_tokens: tuple[str, ...] = ()) -> PreTrainedTokenizerFast:
    vocab = dict(_BASE_VOCAB)
    for token in extra_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    tokenizer = TokenizersTokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
