import sys
from pathlib import Path

# Ensure repo root is on sys.path for test discovery (e.g. app imports)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain_core.embeddings import DeterministicFakeEmbedding
import numpy as np
import pytest


class UnitFakeEmbedding(DeterministicFakeEmbedding):
    """Deterministic, L2-normalized (inner product == cosine, like the real model)."""

    def _get_embedding(self, seed=None):
        v = np.asarray(super()._get_embedding(seed=seed))
        return list(v / np.linalg.norm(v))


@pytest.fixture
def fake_emb():
    return UnitFakeEmbedding(size=16)
