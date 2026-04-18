"""
Standalone multilingual embedding utility.
Used by qdrant_service internally, but exposed here for CLI use.

Usage:
    python ml/embeddings/embed.py "Some text to embed"
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import sys

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """Returns a normalised 384-dim vector for the input text."""
    model = load_model()
    return model.encode(text, normalize_embeddings=True)


def embed_batch(texts: list) -> np.ndarray:
    """Returns a (N, 384) array for a list of texts."""
    model = load_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Test embedding"
    vec = embed(text)
    print(f"Text   : {text}")
    print(f"Shape  : {vec.shape}")
    print(f"Norm   : {np.linalg.norm(vec):.4f}")
    print(f"First 5: {vec[:5]}")
