import numpy as np
from coordinator import insert, query_topk, train_all

DIM = 128

# Train shards
print(train_all())

# Insert a few vectors
for i in range(10):
    vec = np.random.rand(DIM).astype("float32")
    print(insert(f"vec_{i}", vec))

# Query a new vector
query_vec = np.random.rand(DIM).astype("float32")
print(query_topk(query_vec, k=5))
