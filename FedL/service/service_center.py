'''
Author: Sasha
Date: 2023-02-27 10:53:46
LastEditors: Sasha
LastEditTime: 2023-03-29 16:28:29
Description: 
FilePath: /FederatedLearning/FedL/service/service_center.py
'''
#%%
from concurrent import futures
import grpc
from FedL.database.fed_lmdb import feddb_center
from FedL.federatedml.proto import data_pb2, data_pb2_grpc
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
class DataTransferService(data_pb2_grpc.DataTransferServicer):
    def __init__(self, ):
        pass
    def get_data(self, request,content):
        log.info("get key %s" % request.key)
        with feddb_center.db.begin(write=True) as db:
            db.put(request.key,request.value)
        return data_pb2.ReturnCode(code='0',msg='sucess')

MAX_MESSAGE_LENGTH = 256*1024*1024  # 可根据具体需求设置，此处设为256M
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
               ]
        )
    data_pb2_grpc.add_DataTransferServicer_to_server(DataTransferService(), server)
    server.add_insecure_port('[::]:%s' % ServiceConf.port_center)
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
