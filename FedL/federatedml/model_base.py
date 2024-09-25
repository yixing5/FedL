'''
Author: Sasha
Date: 2023-02-27 09:31:51
LastEditors: Sasha
LastEditTime: 2023-04-07 12:17:41
Description: 
FilePath: /FederatedLearning/FedL/federatedml/model_base.py
'''
import grpc
import pickle
from FedL.federatedml.proto import data_pb2, data_pb2_grpc
import time
from abc import ABCMeta,abstractmethod
from FedL.util.log import log
from FedL.federatedml.secureprotol.encrypt import PaillierEncrypt,RsaEncrypt
from Cryptodome.PublicKey import RSA

class ModelBase(metaclass=ABCMeta):
    def __init__(self,fed_lmdb) -> None:
        self.fed_lmdb=fed_lmdb
    def get(self,key):
        log.info('get %s' % key)
        count = 0
        key = key.encode('utf8')
        while 1:
            with self.fed_lmdb.db.begin(write=True) as db:
                value = db.get(key)
                if value:
                    db.delete(key)
                    break
                time.sleep(0.0001)
                count +=1
                if count % 10000==0:
                    log.info('waiting ... %s' % key)
        log.info('get %s seccuss' % key)
        return pickle.loads(value)
                    
    def remote(self,data,key,part,ip="localhost"):
        log.info("remote data : %s" % key)
        channel = grpc.insecure_channel('%s:%s' % (ip,part))
        stub = data_pb2_grpc.DataTransferStub(channel)
        request = data_pb2.Point(key=key.encode('utf8'),value=pickle.dumps(data))
        response = stub.get_data(request)
        log.info("%s %s" % (key,response))
        return response
    
    def _generate_key(self,types="paillier"):
        map_dict = {'paillier':PaillierEncrypt,'rsa':RsaEncrypt}
        if types not in map_dict:
            raise ValueError("%s Encrypt method not support" % types)
        cipher = map_dict.get(types)()
        cipher.generate_key(1024)
        self.public_key ,self.private_key = cipher.get_public_key(),cipher.get_privacy_key()


        
    @abstractmethod
    def run(self,n):
        pass

if __name__ == '__main__':
    cipher = PaillierEncrypt()
    cipher.generate_key(1024)
    public_key ,private_key = cipher.get_public_key(),cipher.get_privacy_key()
    a = public_key.encrypt(1.2)
    b = private_key.decrypt(a)