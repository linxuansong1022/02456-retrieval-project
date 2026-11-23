import torch

def cls_pooling(hidden_states):
    # Take the first token (CLS): [B,T,D] -> [B,D]
    return hidden_states[:, 0, :]

def mean_pooling(hidden_states, attention_mask):
    # Average the token vectors, ignoring padding.
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    total = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    return total / count


def max_pooling(hidden_states, attention_mask):
    # Take the maximum in each feature dimension, ignoring padding.
    padding = attention_mask.unsqueeze(-1) == 0
    hidden_states = hidden_states.masked_fill(padding, torch.finfo(hidden_states.dtype).min)
    return hidden_states.max(dim=1).values


def pool(hidden_states, attention_mask, pooling_type="cls"):
    # Choose the pooling method from the config.
    if pooling_type == "cls":
        return cls_pooling(hidden_states)
    elif pooling_type == "mean":
        return mean_pooling(hidden_states, attention_mask)
    elif pooling_type == "max":
        return max_pooling(hidden_states, attention_mask)
    raise ValueError(f"unknown pooling type {pooling_type!r}, expected cls, mean or max")
