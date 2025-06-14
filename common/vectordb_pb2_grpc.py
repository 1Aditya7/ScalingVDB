# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import vectordb_pb2 as vectordb__pb2

GRPC_GENERATED_VERSION = '1.72.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in vectordb_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class VectorDBStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Insert = channel.unary_unary(
                '/vector.VectorDB/Insert',
                request_serializer=vectordb__pb2.InsertRequest.SerializeToString,
                response_deserializer=vectordb__pb2.InsertReply.FromString,
                _registered_method=True)
        self.Query = channel.unary_unary(
                '/vector.VectorDB/Query',
                request_serializer=vectordb__pb2.QueryRequest.SerializeToString,
                response_deserializer=vectordb__pb2.QueryReply.FromString,
                _registered_method=True)
        self.Train = channel.unary_unary(
                '/vector.VectorDB/Train',
                request_serializer=vectordb__pb2.TrainRequest.SerializeToString,
                response_deserializer=vectordb__pb2.TrainReply.FromString,
                _registered_method=True)


class VectorDBServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Insert(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Query(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Train(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_VectorDBServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Insert': grpc.unary_unary_rpc_method_handler(
                    servicer.Insert,
                    request_deserializer=vectordb__pb2.InsertRequest.FromString,
                    response_serializer=vectordb__pb2.InsertReply.SerializeToString,
            ),
            'Query': grpc.unary_unary_rpc_method_handler(
                    servicer.Query,
                    request_deserializer=vectordb__pb2.QueryRequest.FromString,
                    response_serializer=vectordb__pb2.QueryReply.SerializeToString,
            ),
            'Train': grpc.unary_unary_rpc_method_handler(
                    servicer.Train,
                    request_deserializer=vectordb__pb2.TrainRequest.FromString,
                    response_serializer=vectordb__pb2.TrainReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'vector.VectorDB', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('vector.VectorDB', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class VectorDB(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Insert(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/vector.VectorDB/Insert',
            vectordb__pb2.InsertRequest.SerializeToString,
            vectordb__pb2.InsertReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Query(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/vector.VectorDB/Query',
            vectordb__pb2.QueryRequest.SerializeToString,
            vectordb__pb2.QueryReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Train(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/vector.VectorDB/Train',
            vectordb__pb2.TrainRequest.SerializeToString,
            vectordb__pb2.TrainReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
