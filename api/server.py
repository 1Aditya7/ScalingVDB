from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from index.faiss_index import FaissIVFPQIndex
from index.storage import DiskVectorStore
import threading

app = FastAPI()
dim = 128
store = DiskVectorStore("data/vectors.memmap", dim, max_vectors=100_000)
index = FaissIVFPQIndex(dim)

class InsertRequest(BaseModel):
    vec_id: str

@app.post("/train")
def train():
    def train_async():
        print("Training started...")
        train_data = np.random.rand(2000, dim).astype('float32')
        index.train(train_data)        # internally sets trained = True
        print("Training complete.")
        index.trained = True
    
    threading.Thread(target=train_async).start()
    return {"status": "training started"}

index.trained = True

@app.post("/insert")
def insert(req: InsertRequest):
    if not index.trained:
        return {"error": "Index not trained yet. Please wait."}
    vector = np.random.rand(dim).astype('float32')
    idx = store.add(req.vec_id, vector)
    index.add(np.expand_dims(vector, axis=0), np.array([idx]))
    return {"status": "inserted", "index_id": idx}


@app.get("/query")
def query(k: int = 5):
    q = np.random.rand(1, dim).astype('float32')
    indices, distances = index.search(q, k)
    return {"results": list(zip(indices[0].tolist(), distances[0].tolist()))}
