import grpc
from concurrent import futures
import numpy as np
import faiss

from common import vectordb_pb2, vectordb_pb2_grpc

DIM = 128
NPROBE = 16

class VectorDBServicer(vectordb_pb2_grpc.VectorDBServicer):
    def __init__(self):
        self.index = faiss.index_factory(DIM, "IVF100,PQ64")
        self.index.nprobe = NPROBE
        self.trained = False
        self.next_id = 0

    def Train(self, request, context):
        train_data = np.random.rand(2000, DIM).astype('float32')
        self.index.train(train_data)
        self.trained = True
        print("Shard trained.")
        return vectordb_pb2.TrainReply(status="trained")

    def Insert(self, request, context):
        if not self.trained:
            return vectordb_pb2.InsertReply(status="index not trained", shard_index=-1)

        vec = np.array(request.vector, dtype='float32').reshape(1, -1)
        vec_id = self.next_id
        self.index.add_with_ids(vec, np.array([vec_id]))
        self.next_id += 1

        return vectordb_pb2.InsertReply(status="inserted", shard_index=vec_id)

    def Query(self, request, context):
        q = np.array(request.vector, dtype='float32').reshape(1, -1)
        distances, ids = self.index.search(q, request.k)

        return vectordb_pb2.QueryReply(
            ids=ids[0].tolist(),
            distances=distances[0].tolist()
        )

def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vectordb_pb2_grpc.add_VectorDBServicer_to_server(VectorDBServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Shard running on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    import sys
    serve(port=int(sys.argv[1]))
