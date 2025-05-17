import numpy as np
import pandas as pd
from typing import List, Tuple

DATASET_PATH = "data/mock_wealthy_dataset.csv"
EMBEDDING_DIM = 512

df = pd.read_csv(DATASET_PATH)

embedding_columns = [f"embedding_{i}" for i in range(EMBEDDING_DIM)]
embeddings_matrix = df[embedding_columns].values.astype(np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def find_similar_wealthy_individuals(input_embedding: np.ndarray,top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Compute cosine similarities and return top_k similar individuals.
    """
    norm_input = input_embedding / np.linalg.norm(input_embedding)
    norm_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

    similarities = np.dot(norm_embeddings, norm_input)

    top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
    sorted_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

    return [(df.iloc[i]["name"], float(similarities[i])) for i in sorted_indices]


def estimate_net_worth_from_similars(top_similar: List[Tuple[str, float]]) -> float:
    """
    Estimate net worth based on precomputed top-k similar individuals.
    """
    filtered = [(name, score) for name, score in top_similar if score > 0]
    if not filtered:
        return float(df["net_worth"].mean())

    names, scores = zip(*filtered)
    weights = np.array(scores) / sum(scores)
    net_worths = df[df["name"].isin(names)]["net_worth_million"].values

    return float(np.dot(weights, net_worths))
