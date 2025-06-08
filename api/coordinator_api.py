from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np

from coordinator import insert, train_all, query_topk

app = FastAPI()

DIM = 128  

class InsertInput(BaseModel):
    vec_id: str
    vector: list[float]

class QueryInput(BaseModel):
    vector: list[float]
    k: int = 5

@app.post("/insert")
def insert_vector(input: InsertInput):
    vec = np.array(input.vector, dtype=np.float32)
    result = insert(input.vec_id, vec)
    return result

@app.post("/train")
def train_vectors():
    return train_all()

@app.post("/query")
def query_vector(input: QueryInput):
    vec = np.array(input.vector, dtype=np.float32)
    return query_topk(vec, k=input.k)
