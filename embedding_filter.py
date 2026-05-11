"""bge-m3 embedding wrapper: encode questions, encode ontology labels, and produce
top-K candidate labels per chunk for the LLM to choose from."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


class EmbeddingFilter:
    """Loads bge-m3, holds label embeddings, ranks candidates for chunks."""

    def __init__(self, model_name: str, device: str = "cuda",
                 batch_size: int = 256, max_length: int = 512):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.labels: List[str] = []
        self.label_descs: List[str] = []
        self.label_matrix: np.ndarray | None = None  # (L, D) normalized

    # ---------------- encode ----------------

    def encode_texts(self, texts: List[str], max_length: int | None = None) -> np.ndarray:
        """L2-normalized dense embeddings, fp32, shape (N, D)."""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length or self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        mat = np.asarray(out["dense_vecs"], dtype=np.float32)
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        return mat / n

    # ---------------- fit + query ----------------

    def fit_labels(self, labels: List[str], descriptions: List[str] | None = None) -> None:
        self.labels = list(labels)
        self.label_descs = list(descriptions) if descriptions else [l.replace("_", " ") for l in labels]
        # combine name + description for a richer label embedding
        texts = [f"ethical concept: {l.replace('_', ' ')}. {d}".strip()
                 for l, d in zip(self.labels, self.label_descs)]
        self.label_matrix = self.encode_texts(texts, max_length=128)

    def topk_for_texts(self, texts: List[str], k: int = 8,
                       threshold: float = 0.0) -> List[Tuple[List[int], List[float]]]:
        """For each text return (indices_into_labels, scores). Sorted descending."""
        assert self.label_matrix is not None, "call fit_labels first"
        embs = self.encode_texts(texts)
        sims = embs @ self.label_matrix.T          # (B, L)
        L = sims.shape[1]
        k = max(1, min(k, L))
        out: List[Tuple[List[int], List[float]]] = []
        # argsort each row descending (L is tiny, fast)
        order = np.argsort(-sims, axis=1)
        for i in range(sims.shape[0]):
            idx = order[i, :k]
            sc = sims[i, idx]
            mask = sc >= threshold
            if not mask.any():
                # always keep best one as a safety net
                idx = order[i, :1]
                sc = sims[i, idx]
                mask = np.array([True])
            out.append((idx[mask].tolist(), sc[mask].astype(float).tolist()))
        return out
