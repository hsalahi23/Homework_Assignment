import numpy as np
import pandas as pd
import random

def generate_mock_wealthy_dataset(num_samples=100, embedding_dim=512, seed=42,
                                  output_csv_path='../data/mock_wealthy_dataset.csv'):

    random.seed(seed)
    np.random.seed(seed)

    names = [f"Wealthy Person {i}" for i in range(num_samples)]

    embeddings = np.random.randn(num_samples, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    net_worths = np.random.uniform(low=10, high=400000, size=num_samples)  # In millions

    data = {
        'name': names,
    }

    for i in range(embedding_dim):
        data[f'embedding_{i}'] = embeddings[:, i]

    data['net_worth_million'] = net_worths
    df = pd.DataFrame(data)

    df.to_csv(output_csv_path, index=False)
    print(f"Mock dataset saved to {output_csv_path}")

    return df
