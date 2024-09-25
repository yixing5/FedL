'''
Author: Sasha
Date: 2023-02-24 16:02:50
LastEditors: Sasha
LastEditTime: 2023-03-13 18:05:56
Description: 
FilePath: /FederatedLearning/FedL/federatedml/proto/data_pb2_grpc.py
'''
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from FedL.federatedml.proto import data_pb2 as data__pb2


class DataTransferStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.get_data = channel.unary_unary(
                '/data.DataTransfer/get_data',
                request_serializer=data__pb2.Point.SerializeToString,
                response_deserializer=data__pb2.ReturnCode.FromString,
                )


class DataTransferServicer(object):
    """Missing associated documentation comment in .proto file."""

    def get_data(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DataTransferServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'get_data': grpc.unary_unary_rpc_method_handler(
                    servicer.get_data,
                    request_deserializer=data__pb2.Point.FromString,
                    response_serializer=data__pb2.ReturnCode.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'data.DataTransfer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DataTransfer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def get_data(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/data.DataTransfer/get_data',
            data__pb2.Point.SerializeToString,
            data__pb2.ReturnCode.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
