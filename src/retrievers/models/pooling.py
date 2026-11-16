import torch

def cls_pooling(hidden_states):
    # Take the first token (CLS): [B,T,D] -> [B,D]
    return hidden_states[:, 0, :]

def mean_pooling(hidden_states, attention_mask):
    # Average the token vectors, ignoring padding.
    batch_size = hidden_states.shape[0]
    sequence_length = hidden_states.shape[1]
    results = []

    for b in range(batch_size):
        valid_vectors = []
        for t in range(sequence_length):
            if attention_mask[b, t] == 1:
                vector = hidden_states[b, t, :]
                valid_vectors.append(vector)
        vectors = torch.stack(valid_vectors)
        mean_vector = torch.mean(vectors, dim=0)
        results.append(mean_vector)

    return torch.stack(results)


def max_pooling(hidden_states, attention_mask):
    # Take the maximum in each feature dimension, ignoring padding.
    batch_size = hidden_states.shape[0]
    sequence_length = hidden_states.shape[1]
    results = []
    for b in range(batch_size):
        valid_vectors = []
        for t in range(sequence_length):
            if attention_mask[b, t] == 1:
                vector = hidden_states[b, t, :]
                valid_vectors.append(vector)
        vectors = torch.stack(valid_vectors)
        max_result = torch.max(vectors, dim=0)
        max_vector = max_result.values
        results.append(max_vector)

    return torch.stack(results)


def pool(hidden_states, attention_mask, pooling_type="cls"):
    # Choose the pooling method from the config.
    if pooling_type == "cls":
        return cls_pooling(hidden_states)
    elif pooling_type == "mean":
        return mean_pooling(hidden_states, attention_mask)
    elif pooling_type == "max":
        return max_pooling(hidden_states, attention_mask)
