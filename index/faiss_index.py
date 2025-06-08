import faiss
import numpy as np

class FaissIVFPQIndex:
    def __init__(self, dim, nlist=100, nprobe=32):
        self.dim = dim
        self.index = faiss.index_factory(dim, f"IVF{nlist},PQ64")
        self.trained = False
        self.nprobe = nprobe

    def train(self, train_data: np.ndarray):
        self.index.train(train_data)
        self.trained = True

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        self.index.add_with_ids(vectors, ids)

    def search(self, query: np.ndarray, k: int):
        self.index.nprobe = self.nprobe
        distances, indices = self.index.search(query, k)
        return indices, distances
