import grpc
import numpy as np
from concurrent import futures
from typing import List
from prometheus_client import Counter, Summary, start_http_server
from common import vectordb_pb2, vectordb_pb2_grpc

SHARD_PORTS = [50051, 50052, 50053]
DIM = 128

# -------------------- Load all shard stubs --------------------
stubs = []
for port in SHARD_PORTS:
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = vectordb_pb2_grpc.VectorDBStub(channel)
    stubs.append(stub)

# -------------------- Insert (vec_id hash → shard) --------------------
def insert(vec_id: str, vector: np.ndarray):
    shard_idx = hash(vec_id) % len(stubs)
    vec = vector.astype("float32").flatten().tolist()

    req = vectordb_pb2.InsertRequest(vec_id=vec_id, vector=vec)
    res = stubs[shard_idx].Insert(req)

    return {"status": res.status, "shard": shard_idx}

# -------------------- Train all shards --------------------
def train_all():
    for stub in stubs:
        stub.Train(vectordb_pb2.TrainRequest())
    return {"status": "all shards trained"}

# -------------------- Query top-k from all shards --------------------
def query_topk(vector: np.ndarray, k: int = 5):
    vec = vector.astype("float32").flatten().tolist()

    results = []
    for stub in stubs:
        res = stub.Query(vectordb_pb2.QueryRequest(vector=vec, k=k))
        results.extend(zip(res.ids, res.distances))

    # Sort merged results and return top-k globally
    top_k = sorted(results, key=lambda x: x[1])[:k]
    return [{"id": int(i), "distance": float(d)} for i, d in top_k]
