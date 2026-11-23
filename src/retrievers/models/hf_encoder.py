"""Hugging Face encoder tower behind the shared forward(input_ids, attention_mask) interface."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.retrievers.models.pooling import cls_pooling, max_pooling, mean_pooling


class HFEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling_type: str = "cls",
        freeze: bool = False,
        collect_attention: bool = False,
    ):
        super().__init__()
        if pooling_type not in ("cls", "mean", "max"):
            raise ValueError(f"pooling_type must be one of 'cls'/'mean'/'max', got {pooling_type!r}")
        self.model_name = model_name
        self.pooling_type = pooling_type
        self.collect_attention = collect_attention
        # sdpa attention does not return weights.
        from_pretrained_kwargs = {"attn_implementation": "eager"} if collect_attention else {}
        self.model = AutoModel.from_pretrained(model_name, **from_pretrained_kwargs)
        self._attention_maps: list[torch.Tensor] | None = None
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Recompute activations so two trainable BERT towers fit in 6 GB.
            self.model.gradient_checkpointing_enable()

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=self.collect_attention,
        )
        hidden_states = outputs.last_hidden_state

        if self.collect_attention:
            # one (B, heads, T, T) tensor per layer
            self._attention_maps = list(outputs.attentions)
        else:
            self._attention_maps = None

        if self.pooling_type == "cls":
            return cls_pooling(hidden_states)
        if self.pooling_type == "mean":
            return mean_pooling(hidden_states, attention_mask)
        return max_pooling(hidden_states, attention_mask)

    def attention_maps(self) -> list[torch.Tensor] | None:
        return self._attention_maps
