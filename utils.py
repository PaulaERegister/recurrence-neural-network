import numpy as np
import pandas as pd

def get_embeddings(model):
    embeddings = model.get_layer(index=0).get_weights()[0]
    embeddings /= np.linalg.norm(embeddings, axis=1).reshape((-1,1))
    return np.nan_to_num(embeddings)

def find_closest(query, embedding_matrix, word_idx, idx_word, n=10):
    idx = word_idx.get(query, None)
    if idx is None:
        print(f'{query} not found in vocab')
        return
    vec = embedding_matrix[idx]
    if np.all(vec==0):
        print(f'{query} has no pre-trained embedding')
        return
    dists = np.dot(embedding_matrix, vec)
    idxs = np.argsort(dists)[::-1][:n]
    sorted_dists = dists[idxs]
    closest = [idx_word[i] for i in idxs]
    print(f'Query: {query}\n')
    for word, dist in zip(closest, sorted_dists):
        print(f'Word: {word::15} Cosine Similarity: {round(dist, 4)}')

def get_data(file, filters='!"%;[\\]^_`{|}~\t\n', training_len=50, lower=False):
    data = pd.read_csv(file, parse_dates=)