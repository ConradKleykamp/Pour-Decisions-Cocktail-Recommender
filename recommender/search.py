import sys
from pathlib import Path

# Adding the project root to sys.path so config.py is importable when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, EMBEDDINGS_PATH, METADATA_PATH

# Declaring module-level state for lazy-loaded index and model
_model = None
_embeddings = None
_metadata = None


def _load_index() -> None:
    # Loading model, embeddings, and metadata on the first search call
    global _model, _embeddings, _metadata

    if _model is not None:
        return

    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {EMBEDDINGS_PATH.parent}. "
            "Run [bold]python embeddings/build_index.py[/bold] first."
        )

    _model = SentenceTransformer(EMBEDDING_MODEL)
    _embeddings = np.load(EMBEDDINGS_PATH)
    _metadata = joblib.load(METADATA_PATH)


def search(query: str, top_k: int) -> list[dict]:
    # Ensuring the index is loaded before running a query
    _load_index()

    # Embedding the query string using the same model as the index
    query_vector = _model.encode([query])

    # Computing cosine similarity between the query and all cocktail embeddings
    scores = cosine_similarity(query_vector, _embeddings)[0]

    # Selecting the top-k indices sorted by descending similarity score
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Building result dicts by merging metadata with similarity scores
    results = []
    for idx in top_indices:
        entry = dict(_metadata[idx])
        entry["score"] = float(scores[idx])
        results.append(entry)

    return results
