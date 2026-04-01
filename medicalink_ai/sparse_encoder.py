"""Sparse vectors (BM25-style) qua FastEmbed — dùng cho hybrid search trên Qdrant."""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_sparse_model(model_name: str):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model_name)


def text_to_sparse_vector(text: str, model_name: str) -> SparseVector:
    model = _get_sparse_model(model_name)
    chunk = text.replace("\n", " ").strip()[:8000] or " "
    emb = next(model.embed([chunk]))
    obj = emb.as_object()
    idx = obj["indices"]
    val = obj["values"]
    indices = [int(x) for x in idx.tolist()]
    values = [float(x) for x in val.tolist()]
    return SparseVector(indices=indices, values=values)
