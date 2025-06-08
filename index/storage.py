import numpy as np
import os

class DiskVectorStore:
    def __init__(self, path: str, dim: int, max_vectors: int):
        self.path = path
        self.dim = dim
        self.max_vectors = max_vectors
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.memmap = np.memmap(path, dtype='float32', mode='w+', shape=(max_vectors, dim))
        self.index = 0
        self.id_map = {}

    def add(self, vec_id: str, vector: np.ndarray) -> int:
        assert vector.shape == (self.dim,)
        self.memmap[self.index] = vector
        self.id_map[vec_id] = self.index
        self.index += 1
        return self.index - 1

    def get(self, vec_id: str) -> np.ndarray:
        return self.memmap[self.id_map[vec_id]]