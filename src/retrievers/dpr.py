import torch
import torch.nn.functional as F

from src.retrievers.base_retriever import Retriever
from src.training.build import load_towers


def encode_texts(encoder, tokenizer, texts, max_len, batch_size, device):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=max_len, return_tensors="pt")
            output = encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            embeddings.append(F.normalize(output, dim=-1))
    return torch.cat(embeddings)


class DPR(Retriever):
    def __init__(self, documents, query_encoder, document_encoder, tokenizer,
                 query_max_len=64, passage_max_len=256, batch_size=128, device="cpu"):
        super().__init__(documents)
        self.query_encoder = query_encoder.to(device).eval()
        self.document_encoder = document_encoder.to(device).eval()
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.batch_size = batch_size
        self.device = device

        texts = [document.GetText() for document in self.index.GetDocuments()]
        self.document_embeddings = encode_texts(self.document_encoder, tokenizer, texts,
                                                passage_max_len, batch_size, device)

    @classmethod
    def from_checkpoint(cls, documents, checkpoint_dir, batch_size=128, device="cpu"):
        query_encoder, document_encoder, meta, tokenizer = load_towers(checkpoint_dir, device)
        return cls(documents, query_encoder, document_encoder, tokenizer,
                   meta["query_max_len"], meta["passage_max_len"], batch_size, device)

    def CalculateScores(self, queries: list[str]):
        query_embeddings = encode_texts(self.query_encoder, self.tokenizer, queries,
                                        self.query_max_len, self.batch_size, self.device)
        return (query_embeddings @ self.document_embeddings.T).cpu().numpy()
